import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
from models import ActorCritic
from environment import RayVecEnv, create_env_creator
from utils import RolloutBuffer, to_tensor, Logger

class ParallelPPO:
    """
    并行版本的PPO算法实现，使用多个环境并行收集经验
    针对Pendulum环境特别优化的版本
    """
    def __init__(self, 
                 env_name,
                 num_envs=8,
                 num_steps=512,  # 减小步数，增加更新频率
                 epochs=4,       # 减少训练轮数，防止过拟合
                 mini_batch_size=64,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 seed=0,
                 device="cpu",
                 log_dir="./logs",
                 hidden_dim=64):  # 使用更小的网络
        """
        初始化ParallelPPO代理
        
        参数:
            env_name: 环境名称
            num_envs: 并行环境数量
            num_steps: 每个环境采集的步数
            epochs: 每批数据的训练轮数
            mini_batch_size: 小批量大小
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_param: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪范数
            seed: 随机种子
            device: 计算设备
            log_dir: 日志目录
            hidden_dim: 神经网络隐藏层维度
        """
        # 保存环境名称
        self.env_name = env_name
        
        # 为Pendulum环境专门优化参数
        if 'Pendulum' in env_name:
            print("检测到Pendulum环境，使用专用优化参数")
            lr = 5e-4  # 更高的学习率
            epochs = 4  # 减少训练轮数
            num_steps = 512  # 更短的轨迹长度
            hidden_dim = 64  # 更小的网络
            entropy_coef = 0.005  # 更小的熵系数
            value_coef = 0.25  # 更小的价值系数
            max_grad_norm = 0.1  # 更严格的梯度裁剪
            gae_lambda = 0.92  # 更平滑的GAE
        
        # 确保Ray已初始化
        if not ray.is_initialized():
            ray.init()
            
        # 初始化并行环境
        env_creator = create_env_creator(env_name)
        self.env = RayVecEnv(env_creator, num_envs=num_envs, seed=seed)
        
        # 获取环境信息
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.is_discrete = getattr(self.env, 'is_discrete', False)  # 获取动作空间类型
        
        # 初始化参数
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络和优化器
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, hidden_dim=hidden_dim, discrete=self.is_discrete).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 初始化rollout缓冲区
        buffer_size = num_steps * num_envs
        self.rollout_buffer = RolloutBuffer(
            buffer_size, 
            self.state_dim, 
            self.action_dim, 
            self.device, 
            is_discrete=self.is_discrete,
            mini_batch_size=mini_batch_size
        )
        
        # 初始化日志记录器
        self.logger = Logger(log_dir)
        
        # 计数器
        self.total_steps = 0
        
    def generate_rollout(self):
        """采集轨迹数据，优化Pendulum环境的数据收集过程"""
        self.rollout_buffer.reset()
        
        # 重置所有环境，获取初始观察
        states = self.env.reset()
        if isinstance(states, tuple):
            states = states[0]  # 适应新版gym格式
            
        dones = [False] * self.env.num_envs
        
        # 记录每次采样的整体奖励
        episode_rewards = []
        total_rewards = np.zeros(self.env.num_envs)
        episode_lengths = np.zeros(self.env.num_envs)
        
        # 每个worker收集self.num_steps步数据
        for step in range(self.num_steps):
            # 加载环境状态到缓冲区 - 为每个并行环境的状态分配空间
            for i in range(self.num_envs):
                idx = step * self.num_envs + i
                if idx < self.rollout_buffer.capacity:
                    self.rollout_buffer.states[idx] = states[i]
            
            # 从策略网络获取动作和值
            with torch.no_grad():
                # 转换状态为张量
                state_tensor = torch.FloatTensor(states).to(self.device)
                
                # 使用当前策略采样动作
                actions, log_probs, values = self.actor_critic.get_action(state_tensor)
                
                # 将Pendulum环境的动作从[-1,1]空间放大到实际动作空间
                if 'Pendulum' in self.env_name:
                    env_actions = actions.cpu().numpy() * 2.0  # 扩大到[-2,2]范围
                else:
                    env_actions = actions.cpu().numpy()
                
                # 确保动作值合法且不包含NaN
                if np.isnan(env_actions).any():
                    env_actions = np.nan_to_num(env_actions, nan=0.0)
                    
            # 使用动作与环境交互 - 处理新旧API兼容性
            try:
                # 尝试新版gymnasium API (返回5个值)
                step_result = self.env.step(env_actions)
                if len(step_result) == 5:
                    next_states, rewards, terminated, truncated, infos = step_result
                    # 合并terminated和truncated为done标志
                    dones = np.logical_or(terminated, truncated)
                else:
                    # 旧版gym API (返回4个值)
                    next_states, rewards, dones, infos = step_result
            except ValueError:
                # 如果解包失败，说明使用的是旧版API
                next_states, rewards, dones, infos = self.env.step(env_actions)
            
            # 新版gym接口适配
            if isinstance(next_states, tuple):
                next_states = next_states[0]
            if isinstance(rewards, tuple):
                rewards = rewards[0]
                
            # 记录分数并处理回合结束情况
            if isinstance(infos, dict) and 'episode' in infos:
                for item in infos['episode']:
                    if 'r' in item:
                        episode_rewards.append(item['r'])
            else:
                # 累积奖励并检查回合结束
                total_rewards += rewards
                episode_lengths += 1
                
                # 如果回合结束，记录总奖励并重置
                for i, done in enumerate(dones):
                    if done and episode_lengths[i] > 0:
                        episode_rewards.append(total_rewards[i])
                        total_rewards[i] = 0
                        episode_lengths[i] = 0
            
            # 存储数据到缓冲区 - 一次性存储所有并行环境的数据
            self.rollout_buffer.insert(
                actions=actions.cpu().numpy(),
                values=values.cpu().numpy().flatten(),
                log_probs=log_probs.cpu().numpy(),
                rewards=np.array(rewards).flatten(),
                dones=np.array(dones).flatten()
            )
            
            # 更新状态
            states = next_states
            
        # 计算最后状态的值作为bootstrap值
        with torch.no_grad():
            last_state = torch.FloatTensor(states).to(self.device)
            last_value = self.actor_critic.get_value(last_state).cpu().numpy().flatten()
            
        # 计算GAE和回报
        self.rollout_buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        return episode_rewards
    
    def update_policy(self):
        """
        使用收集的轨迹数据更新策略 (针对Pendulum环境优化版本)
        
        返回:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 熵
        """
        # 获取所有数据
        states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get_all()
        
        # 安全处理returns和advantages中的NaN或Inf值
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            print("警告: returns中包含NaN或Inf值，已纠正")
            returns = torch.where(torch.isnan(returns) | torch.isinf(returns),
                                 torch.zeros_like(returns),
                                 returns)
        
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("警告: advantages中包含NaN或Inf值，已纠正")
            advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages),
                                    torch.zeros_like(advantages),
                                    advantages)
        
        # 简单归一化，避免过度处理
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            # 极端值裁剪
            advantages = torch.clamp(advantages, -3.0, 3.0)
        
        # 跟踪损失
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        # 减少训练轮数，避免过拟合
        actual_epochs = min(self.epochs, 5)  # 最多5轮
        
        # 缓存旧值函数预测，用于值函数裁剪
        with torch.no_grad():
            # 根据动作空间类型获取值函数
            if hasattr(self.env, 'is_discrete') and self.env.is_discrete:
                # 离散动作空间
                _, old_values = self.actor_critic(states)
            else:
                # 连续动作空间
                _, _, old_values = self.actor_critic(states)
        
        # 多轮训练
        batch_size = states.size(0)
        
        for epoch in range(actual_epochs):
            # 生成随机排列的索引
            indices = torch.randperm(batch_size).to(self.device)
            
            # 分批处理数据
            for start in range(0, batch_size, self.mini_batch_size):
                update_count += 1
                end = min(start + self.mini_batch_size, batch_size)
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # 前向传播计算新的动作概率、熵和状态值
                try:
                    new_log_probs, entropy, values = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                    
                    # 安全检查
                    if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                        print("警告: new_log_probs包含NaN或Inf值，跳过此批次")
                        continue
                    
                    if torch.isnan(values).any() or torch.isinf(values).any():
                        print("警告: values包含NaN或Inf值，跳过此批次")
                        continue
                    
                    # 计算策略损失 (使用简化版本，减少计算复杂度)
                    ratio = torch.exp(torch.clamp(new_log_probs - batch_old_log_probs, -10, 10))
                    
                    # 裁剪比率，避免异常值
                    ratio = torch.clamp(ratio, 0.1, 10.0)
                    
                    # 使用PPO裁剪目标
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值函数裁剪 (PPO论文中推荐的做法)
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.clip_param,
                        self.clip_param
                    )
                    value_loss1 = F.mse_loss(values, batch_returns, reduction='none')
                    value_loss2 = F.mse_loss(values_clipped, batch_returns, reduction='none')
                    value_loss = torch.max(value_loss1, value_loss2).mean()
                    
                    # 熵损失
                    entropy_loss = -entropy.mean() 
                    
                    # 总损失
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    # 梯度更新
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 简单梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    # 累加损失
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.mean().item()
                    
                except Exception as e:
                    print(f"更新过程中出错: {e}")
                    self.optimizer.zero_grad()
                    continue
        
        # 防止更新计数为0
        if update_count == 0:
            update_count = 1
        
        # 计算平均损失
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def train(self, total_timesteps, log_interval=10, save_interval=50, progress_callback=None):
        """
        训练PPO代理
        
        参数:
            total_timesteps: 总训练步数
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
            progress_callback: 训练进度回调函数，用于动态调整参数
        """
        # 计算训练的轮次数
        num_updates = total_timesteps // (self.num_steps * self.num_envs)
        
        # 跟踪总步数
        total_steps = 0
        best_mean_reward = float('-inf')
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # 收集轨迹数据
            collect_start = time.time()
            mean_reward = self.generate_rollout()
            collect_end = time.time()
            
            # 更新策略
            update_start = time.time()
            policy_loss, value_loss, entropy = self.update_policy()
            update_end = time.time()
            
            # 更新总步数
            total_steps += self.num_steps * self.num_envs
            
            # 计算每秒步数和剩余时间
            steps_per_sec = total_steps / (time.time() - start_time)
            remaining_updates = num_updates - update
            remaining_time = remaining_updates * (time.time() - start_time) / max(1, update)
            
            # 获取平均回合奖励
            avg_reward = self.logger.get_avg_reward()
            
            # 创建信息字典
            info_dict = {
                'policy_loss': float(policy_loss),
                'value_loss': float(value_loss),
                'entropy': float(entropy),
                'total_steps': total_steps,
                'avg_reward': avg_reward,
                'collect_time': collect_end - collect_start,
                'update_time': update_end - update_start,
                'steps_per_sec': steps_per_sec
            }
            
            # 如果有进度回调函数，则调用它
            if progress_callback is not None:
                info_dict = progress_callback(
                    total_steps, 
                    total_timesteps,
                    info_dict
                )
            
            # 定期打印和记录训练信息
            if update % log_interval == 0 or update == 1:
                # 计算平均奖励
                avg_reward_100 = avg_reward  # 最近100回合的平均奖励
                
                # 记录训练信息到日志
                self.logger.log_training_info(
                    update, num_updates, total_steps, total_timesteps,
                    avg_reward, avg_reward_100, policy_loss, value_loss,
                    entropy, steps_per_sec, remaining_time
                )
                
                # 输出到控制台
                print(f"\r更新: {update}/{num_updates}, 步数: {total_steps}/{total_timesteps}, "
                      f"奖励: {avg_reward:.2f}, 近100均值: {avg_reward_100:.2f}, 速率: {steps_per_sec:.1f}步/秒, "
                      f"剩余时间: {int(remaining_time//60)}分 {int(remaining_time%60)}秒", end="")
                
                # 记录度量信息
                self.logger.log_step(total_steps, {
                    'policy_loss': float(policy_loss),
                    'value_loss': float(value_loss),
                    'entropy': float(entropy),
                    'avg_reward': float(avg_reward),
                    'avg_reward_100': float(avg_reward_100),
                    'steps_per_sec': float(steps_per_sec)
                })
                
            # 定期保存模型
            if (update % save_interval == 0 or update == num_updates):
                model_path = os.path.join(self.logger.log_dir, f"model_{total_steps}.pt")
                self.save(model_path)
                self.logger.log_model_saved(model_path)
                
                # 如果当前奖励是最好的，保存一个best模型
                if avg_reward > best_mean_reward:
                    best_mean_reward = avg_reward
                    best_model_path = os.path.join(self.logger.log_dir, "best_model.pt")
                    self.save(best_model_path)
                    self.logger.log_model_saved(f"最佳模型: {best_model_path} (奖励: {best_mean_reward:.2f})")
                
        # 训练结束，保存最终模型
        final_model_path = os.path.join(self.logger.log_dir, "final_model.pt")
        self.save(final_model_path)
        self.logger.log_model_saved(f"最终模型: {final_model_path}")
        
        # 保存日志数据
        self.logger.dump()
        
        # 关闭日志
        self.logger.close()
        
        print("\n训练完成！")
        return self

    def save(self, path):
        """
        保存模型到指定路径
        
        参数:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path):
        """
        从指定路径加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # 更新工作器权重（如果适用）
        if hasattr(self, '_update_worker_weights'):
            self._update_worker_weights()


@ray.remote
class RayWorker:
    """
    Ray工作器，用于在分布式环境中采样轨迹
    """
    def __init__(self, env_creator, policy_class, policy_kwargs, worker_index, seed=0):
        """
        初始化Ray工作器
        
        参数:
            env_creator: 环境创建器函数
            policy_class: 策略类
            policy_kwargs: 策略初始化参数
            worker_index: 工作器索引
            seed: 随机种子
        """
        from ray.rllib.env.env_context import EnvContext
        
        # 保存种子值供后续使用
        self.seed = seed + worker_index
        
        # 创建环境
        env_config = {"env_id": worker_index}
        # 打印调试信息
        print(f"RayWorker-{worker_index} 初始化: 创建环境, seed={self.seed}")
        
        # 使用EnvContext，确保worker_index正确传递
        env_context = EnvContext(env_config, worker_index=worker_index)
        self.env = env_creator(env_context)
        
        # 获取环境信息
        self.state_dim = self.env.observation_space.shape[0]
        
        # 判断是否为离散动作空间
        self.is_discrete = hasattr(self.env.action_space, 'n')
        if self.is_discrete:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
        
        # 更新策略参数以包含离散动作空间信息
        policy_kwargs.update({'discrete': self.is_discrete})
        
        # 创建策略
        self.policy = policy_class(self.state_dim, self.action_dim, **policy_kwargs)
        
        # 初始化状态
        self.state = None
        self.ep_reward = 0
        self.ep_length = 0
        
    def set_weights(self, weights):
        """设置策略权重"""
        self.policy.load_state_dict(weights)
        
    def sample_episode(self, max_steps=1000):
        """
        采样一个完整的回合
        
        参数:
            max_steps: 最大步数
            
        返回:
            trajectory: 轨迹字典
            ep_reward: 回合奖励
            ep_length: 回合长度
        """
        # 重置环境
        try:
            # 新版gymnasium API支持在reset时传入seed
            self.state = self.env.reset(seed=self.seed)[0]  # 返回(obs, info)，只取obs
        except (TypeError, ValueError):
            # 对于不支持seed参数的环境，使用旧版本API
            self.state = self.env.reset()
            
        self.ep_reward = 0
        self.ep_length = 0
        
        # 存储轨迹
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        
        # 开始回合
        done = False
        while not done and self.ep_length < max_steps:
            # 获取动作
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
                
            # 执行动作
            action_np = action.squeeze().numpy()
            
            # 处理离散和连续动作空间
            if self.is_discrete:
                # 离散动作空间 - 使用整数动作
                if isinstance(action_np, np.ndarray):
                    if action_np.size > 0:  # 如果是非空数组
                        if action_np.ndim > 0:  # 多维数组
                            action_to_env = int(action_np[0])
                        else:  # 0维数组（标量数组）
                            action_to_env = int(action_np.item())
                    else:
                        action_to_env = 0  # 默认值，避免空数组错误
                else:
                    action_to_env = int(action_np)  # 非数组情况
            else:
                # 连续动作空间 - 确保动作形状正确
                if action_np.size == 1 and not isinstance(action_np, np.ndarray):
                    action_np = np.array([action_np])
                elif len(action_np.shape) == 0:  # 如果是标量
                    action_np = np.array([action_np])
                action_to_env = action_np
            
            try:
                # 新版gymnasium API的step返回(obs, reward, terminated, truncated, info)
                step_result = self.env.step(action_to_env)
                if len(step_result) == 5:  # 新版API
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:  # 旧版API
                    next_state, reward, done, _ = step_result
            except ValueError:
                # 旧版API
                next_state, reward, done, _ = self.env.step(action_to_env)
            
            # 存储步骤
            states.append(self.state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # 更新状态
            self.state = next_state
            self.ep_reward += reward
            self.ep_length += 1
            
        # 构建轨迹字典
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'last_state': self.state,
            'ep_reward': self.ep_reward,
            'ep_length': self.ep_length
        }
        
        return trajectory, self.ep_reward, self.ep_length
    
    def sample_steps(self, num_steps):
        """
        采样指定步数的轨迹
        
        参数:
            num_steps: 采样步数
            
        返回:
            trajectory: 轨迹字典
            ep_rewards: 回合奖励列表
            ep_lengths: 回合长度列表
        """
        # 如果状态为空，则重置环境
        if self.state is None:
            try:
                # 新版gymnasium API支持在reset时传入seed
                self.state = self.env.reset(seed=self.seed)[0]  # 返回(obs, info)，只取obs
            except (TypeError, ValueError):
                # 对于不支持seed参数的环境，使用旧版本API
                self.state = self.env.reset()
                
            self.ep_reward = 0
            self.ep_length = 0
            
        # 存储轨迹
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        ep_rewards, ep_lengths = [], []
        
        # 采样指定步数
        steps = 0
        while steps < num_steps:
            # 获取动作
            state_tensor = torch.FloatTensor(self.state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor)
                
            # 执行动作
            action_np = action.squeeze().numpy()
            
            # 处理离散和连续动作空间
            if self.is_discrete:
                # 离散动作空间 - 使用整数动作
                if isinstance(action_np, np.ndarray):
                    if action_np.size > 0:  # 如果是非空数组
                        if action_np.ndim > 0:  # 多维数组
                            action_to_env = int(action_np[0])
                        else:  # 0维数组（标量数组）
                            action_to_env = int(action_np.item())
                    else:
                        action_to_env = 0  # 默认值，避免空数组错误
                else:
                    action_to_env = int(action_np)  # 非数组情况
            else:
                # 连续动作空间 - 确保动作形状正确
                if action_np.size == 1 and not isinstance(action_np, np.ndarray):
                    action_np = np.array([action_np])
                elif len(action_np.shape) == 0:  # 如果是标量
                    action_np = np.array([action_np])
                action_to_env = action_np
            
            try:
                # 新版gymnasium API的step返回(obs, reward, terminated, truncated, info)
                step_result = self.env.step(action_to_env)
                if len(step_result) == 5:  # 新版API
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:  # 旧版API
                    next_state, reward, done, _ = step_result
            except ValueError:
                # 旧版API
                next_state, reward, done, _ = self.env.step(action_to_env)
            
            # 存储步骤
            states.append(self.state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())
            
            # 更新状态
            self.state = next_state
            self.ep_reward += reward
            self.ep_length += 1
            steps += 1
            
            # 如果回合结束
            if done:
                ep_rewards.append(self.ep_reward)
                ep_lengths.append(self.ep_length)
                
                # 重置环境
                try:
                    # 新版gymnasium API支持在reset时传入seed
                    self.state = self.env.reset(seed=self.seed)[0]  # 返回(obs, info)，只取obs
                except (TypeError, ValueError):
                    # 对于不支持seed参数的环境，使用旧版本API
                    self.state = self.env.reset()
                    
                self.ep_reward = 0
                self.ep_length = 0
                
        # 构建轨迹字典
        trajectory = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'last_state': self.state
        }
        
        return trajectory, ep_rewards, ep_lengths
    
    def close(self):
        """关闭环境"""
        self.env.close()


class DistributedPPO:
    """
    分布式PPO实现，使用Ray进行并行化
    """
    def __init__(self, 
                 env_name,
                 num_workers=4,
                 steps_per_worker=512,
                 epochs=10,
                 mini_batch_size=64,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 seed=0,
                 device="cpu",
                 log_dir="./logs",
                 hidden_dim=256):
        """
        初始化分布式PPO
        
        参数:
            env_name: 环境名称
            num_workers: 工作器数量
            steps_per_worker: 每个工作器采样的步数
            epochs: 每批数据的训练轮数
            mini_batch_size: 小批量大小
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_param: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪范数
            seed: 随机种子
            device: 计算设备
            log_dir: 日志目录
            hidden_dim: 神经网络隐藏层维度
        """
        # 确保Ray已初始化
        if not ray.is_initialized():
            ray.init()
            
        # 初始化环境创建器
        self.env_creator = create_env_creator(env_name)
        
        # 创建一个临时环境获取state_dim和action_dim
        from ray.rllib.env.env_context import EnvContext
        temp_env = self.env_creator(EnvContext({"env_id": 0}, worker_index=0))
        self.state_dim = temp_env.observation_space.shape[0]
        
        # 判断是否为离散动作空间
        self.is_discrete = hasattr(temp_env.action_space, 'n')
        if self.is_discrete:
            self.action_dim = temp_env.action_space.n
        else:
            self.action_dim = temp_env.action_space.shape[0]
            
        temp_env.close()
        
        # 初始化参数
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.num_workers = num_workers
        self.steps_per_worker = steps_per_worker
        self.total_steps_per_update = num_workers * steps_per_worker
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络和优化器
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim, hidden_dim=hidden_dim, discrete=self.is_discrete).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 创建Ray工作器
        policy_kwargs = {'hidden_dim': hidden_dim}
        print(f"创建{num_workers}个Ray工作器")
        self.workers = [
            RayWorker.remote(
                self.env_creator,
                ActorCritic,
                policy_kwargs,
                i,
                seed + i
            )
            for i in range(num_workers)
        ]
        
        # 初始化策略权重
        self._update_worker_weights()
        
        # 初始化rollout缓冲区
        buffer_size = self.total_steps_per_update
        self.rollout_buffer = RolloutBuffer(
            buffer_size, 
            self.state_dim, 
            self.action_dim, 
            self.device,
            is_discrete=self.is_discrete,
            mini_batch_size=mini_batch_size
        )
        
        # 初始化日志记录器
        self.logger = Logger(log_dir)
        
        # 计数器
        self.total_steps = 0
        
    def _update_worker_weights(self):
        """更新所有工作器的策略权重"""
        weights = self.actor_critic.state_dict()
        weight_futures = [worker.set_weights.remote(weights) for worker in self.workers]
        ray.get(weight_futures)
        
    def collect_rollouts(self):
        """
        并行收集轨迹数据
        
        返回:
            avg_reward: 平均回合奖励
        """
        # 清空缓冲区
        self.rollout_buffer.clear()
        
        # 每个工作器并行采样
        sample_futures = [worker.sample_steps.remote(self.steps_per_worker) for worker in self.workers]
        worker_results = ray.get(sample_futures)
        
        # 处理结果
        trajectories, rewards_list, lengths_list = zip(*worker_results)
        
        # 计算平均回合奖励
        all_rewards = [r for rewards in rewards_list for r in rewards]
        if all_rewards:
            avg_reward = np.mean(all_rewards)
            for reward in all_rewards:
                self.logger.log_episode(reward)
        else:
            avg_reward = 0
            
        # 处理轨迹数据
        for trajectory in trajectories:
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            dones = trajectory['dones']
            log_probs = trajectory['log_probs']
            values = trajectory['values']
            
            for i in range(len(states)):
                self.rollout_buffer.add(
                    states[i],
                    actions[i],
                    rewards[i],
                    dones[i],
                    log_probs[i],
                    values[i]
                )
        
        # 计算每个工作器最后一个状态的价值
        last_states = np.array([t['last_state'] for t in trajectories])
        with torch.no_grad():
            last_states_tensor = to_tensor(last_states, self.device)
            _, _, last_values = self.actor_critic.get_action(last_states_tensor)
            
            # 确保last_values是numpy数组
            if isinstance(last_values, torch.Tensor):
                last_values = last_values.cpu().numpy()
            
        # 计算回报和优势
        self.rollout_buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)
        
        # 更新总步数
        self.total_steps += self.total_steps_per_update
        
        return avg_reward
    
    def update_policy(self):
        """
        使用收集的轨迹数据更新策略
        
        返回:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 熵
        """
        # 获取所有数据
        states, actions, old_log_probs, returns, advantages = self.rollout_buffer.get_all()
        
        # 归一化优势，增强数值稳定性
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print("警告: 优势函数中存在NaN或Inf值，已纠正")
            # 移除NaN值
            advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                                    torch.zeros_like(advantages), 
                                    advantages)
        
        # 使用更稳健的归一化，防止极端值
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
        # 检查统计量是否有效
        if not torch.isnan(adv_mean) and not torch.isnan(adv_std) and adv_std > 0:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            # 裁剪极端优势值，提高稳定性
            advantages = torch.clamp(advantages, -10.0, 10.0)
        else:
            print("警告: 优势函数统计量无效，跳过归一化")
        
        # 跟踪损失
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # 多轮训练
        batch_size = states.size(0)
        
        for _ in range(self.epochs):
            # 生成随机排列的索引
            indices = torch.randperm(batch_size).to(self.device)
            
            # 分批处理数据
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                batch_indices = indices[start:end]
                
                # 获取批次数据
                batch_data = self.rollout_buffer.get_batch(batch_indices.cpu().numpy())
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch_data
                
                # 前向传播计算新的动作概率、熵和状态值
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(batch_states, batch_actions)
                
                # 确保形状匹配
                if new_log_probs.shape != batch_old_log_probs.shape:
                    if new_log_probs.shape[0] == batch_old_log_probs.shape[0]:
                        new_log_probs = new_log_probs.view_as(batch_old_log_probs)
                
                # 计算比率，并防止数值溢出
                try:
                    # 裁剪log_probs差异，防止exp计算爆炸
                    log_ratio = new_log_probs - batch_old_log_probs
                    log_ratio = torch.clamp(log_ratio, -20, 20)  # 避免exp计算时溢出
                    ratio = torch.exp(log_ratio)
                    
                    # 检查是否存在NaN值
                    if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                        print("警告: 比率计算中存在NaN或Inf值，使用安全值")
                        ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio),
                                          torch.ones_like(ratio),
                                          ratio)
                except Exception as e:
                    print(f"计算比率时出错 {str(e)}，使用默认值1")
                    ratio = torch.ones_like(batch_advantages)
                
                # 确保比率在合理范围内
                ratio = torch.clamp(ratio, 0.0, 10.0)
                
                # 确保比率和优势的形状匹配
                if ratio.shape != batch_advantages.shape:
                    if ratio.dim() > batch_advantages.dim():
                        batch_advantages = batch_advantages.unsqueeze(-1)
                    elif ratio.dim() < batch_advantages.dim():
                        ratio = ratio.squeeze(-1)
                
                # 计算裁剪的优势和策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                
                # 检查中间结果是否有效
                if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                    print("警告: PPO目标计算中存在NaN值")
                    # 使用安全的目标函数
                    policy_loss = -torch.mean(torch.where(
                        torch.isnan(surr1) | torch.isnan(surr2),
                        torch.zeros_like(surr1),
                        torch.min(surr1, surr2)
                    ))
                else:
                    # 正常计算策略损失
                    policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                # 确保values和batch_returns维度匹配
                if values.shape != batch_returns.shape:
                    if values.dim() > batch_returns.dim():
                        batch_returns = batch_returns.unsqueeze(-1)
                    elif values.dim() < batch_returns.dim():
                        values = values.unsqueeze(-1)
                        
                value_loss = F.mse_loss(values, batch_returns)
                
                # 计算熵损失
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 检查损失是否有效
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("警告: 总损失包含NaN或Inf值，跳过此批次更新")
                    continue
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                
                try:
                    loss.backward()
                    
                    # 梯度裁剪，同时检查梯度是否有效
                    if self.max_grad_norm > 0:
                        # 先检查梯度是否存在NaN值
                        has_nan_grad = False
                        for param in self.actor_critic.parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    has_nan_grad = True
                                    break
                        
                        if has_nan_grad:
                            print("警告: 梯度中存在NaN值，跳过此次参数更新")
                            # 重置梯度
                            self.optimizer.zero_grad()
                            continue
                        
                        # 如果梯度有效，应用梯度裁剪
                        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                    # 更新参数
                    self.optimizer.step()
                    
                except Exception as e:
                    print(f"反向传播或优化时出错 {str(e)}，跳过此批次更新")
                    self.optimizer.zero_grad()
                    continue
                
                # 累加损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # 计算平均损失
        num_updates = self.epochs * (batch_size // self.mini_batch_size)
        total_policy_loss /= num_updates
        total_value_loss /= num_updates
        total_entropy /= num_updates
        
        # 更新工作器权重
        self._update_worker_weights()
        
        return total_policy_loss, total_value_loss, total_entropy
    
    def train(self, total_timesteps, log_interval=10, save_interval=50, progress_callback=None):
        """
        训练分布式PPO代理
        
        参数:
            total_timesteps: 总训练步数
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
            progress_callback: 训练进度回调函数，用于动态调整参数
        """
        # 计算训练的轮次数
        num_updates = total_timesteps // (self.steps_per_worker * self.num_workers)
        
        # 跟踪总步数
        total_steps = 0
        best_mean_reward = float('-inf')
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            # 同步最新权重到所有工作器
            self._update_worker_weights()
            
            # 收集轨迹数据
            collect_start = time.time()
            mean_reward = self.collect_rollouts()
            collect_end = time.time()
            
            # 更新策略
            update_start = time.time()
            policy_loss, value_loss, entropy = self.update_policy()
            update_end = time.time()
            
            # 更新总步数
            total_steps += self.steps_per_worker * self.num_workers
            
            # 计算每秒步数和剩余时间
            steps_per_sec = total_steps / (time.time() - start_time)
            remaining_updates = num_updates - update
            remaining_time = remaining_updates * (time.time() - start_time) / max(1, update)
            
            # 获取平均回合奖励
            avg_reward = self.logger.get_avg_reward()
            
            # 创建信息字典
            info_dict = {
                'policy_loss': float(policy_loss),
                'value_loss': float(value_loss),
                'entropy': float(entropy),
                'total_steps': total_steps,
                'avg_reward': avg_reward,
                'collect_time': collect_end - collect_start,
                'update_time': update_end - update_start,
                'steps_per_sec': steps_per_sec
            }
            
            # 如果有进度回调函数，则调用它
            if progress_callback is not None:
                info_dict = progress_callback(
                    total_steps, 
                    total_timesteps,
                    info_dict
                )
            
            # 定期打印和记录训练信息
            if update % log_interval == 0 or update == 1:
                # 计算平均奖励
                avg_reward_100 = avg_reward  # 最近100回合的平均奖励
                
                # 记录训练信息到日志
                self.logger.log_training_info(
                    update, num_updates, total_steps, total_timesteps,
                    avg_reward, avg_reward_100, policy_loss, value_loss,
                    entropy, steps_per_sec, remaining_time
                )
                
                # 输出到控制台
                print(f"\r更新: {update}/{num_updates}, 步数: {total_steps}/{total_timesteps}, "
                      f"奖励: {avg_reward:.2f}, 近100均值: {avg_reward_100:.2f}, 速率: {steps_per_sec:.1f}步/秒, "
                      f"剩余时间: {int(remaining_time//60)}分 {int(remaining_time%60)}秒", end="")
                
                # 记录度量信息
                self.logger.log_step(total_steps, {
                    'policy_loss': float(policy_loss),
                    'value_loss': float(value_loss),
                    'entropy': float(entropy),
                    'avg_reward': float(avg_reward),
                    'avg_reward_100': float(avg_reward_100),
                    'steps_per_sec': float(steps_per_sec)
                })
                
            # 定期保存模型
            if (update % save_interval == 0 or update == num_updates):
                model_path = os.path.join(self.logger.log_dir, f"model_{total_steps}.pt")
                self.save(model_path)
                self.logger.log_model_saved(model_path)
                
                # 如果当前奖励是最好的，保存一个best模型
                if avg_reward > best_mean_reward:
                    best_mean_reward = avg_reward
                    best_model_path = os.path.join(self.logger.log_dir, "best_model.pt")
                    self.save(best_model_path)
                    self.logger.log_model_saved(f"最佳模型: {best_model_path} (奖励: {best_mean_reward:.2f})")
            
        # 训练结束，保存最终模型
        final_model_path = os.path.join(self.logger.log_dir, "final_model.pt")
        self.save(final_model_path)
        self.logger.log_model_saved(f"最终模型: {final_model_path}")
        
        # 保存日志数据
        self.logger.dump()
        
        # 关闭日志
        self.logger.close()
        
        print("\n训练完成！")
        return self

    def save(self, path):
        """
        保存模型到指定路径
        
        参数:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path):
        """
        从指定路径加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # 更新工作器权重
        self._update_worker_weights() 