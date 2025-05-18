import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from collections import deque

def to_tensor(x, device="cpu"):
    """将numpy数组转换为torch张量"""
    if isinstance(x, np.ndarray):
        return torch.FloatTensor(x).to(device)
    return x.to(device)

class RolloutBuffer:
    """
    滚动缓冲区，用于存储和批处理环境交互的轨迹数据
    支持离散和连续动作空间，以及多环境并行
    """
    def __init__(self, capacity, state_dim, action_dim, device="cpu", is_discrete=False, mini_batch_size=64, num_envs=1):
        self.capacity = capacity
        self.device = device
        self.is_discrete = is_discrete
        self.mini_batch_size = mini_batch_size
        self.num_envs = num_envs
        
        # 初始化缓冲区，考虑多环境
        if isinstance(state_dim, (tuple, list)):
            self.states = np.zeros((capacity, num_envs, *state_dim), dtype=np.float32)
        else:
            self.states = np.zeros((capacity, num_envs, state_dim), dtype=np.float32)
        
        # 根据动作空间类型初始化动作缓冲区
        if is_discrete:
            # 离散动作空间 - 存储整数动作
            self.actions = np.zeros((capacity, num_envs), dtype=np.int64)
        else:
            # 连续动作空间 - 存储浮点向量
            if isinstance(action_dim, (tuple, list)):
                self.actions = np.zeros((capacity, num_envs, *action_dim), dtype=np.float32)
            else:
                self.actions = np.zeros((capacity, num_envs, action_dim), dtype=np.float32)
            
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs), dtype=bool)
        self.log_probs = np.zeros((capacity, num_envs), dtype=np.float32)
        self.values = np.zeros((capacity, num_envs), dtype=np.float32)
        self.returns = np.zeros((capacity, num_envs), dtype=np.float32)
        self.advantages = np.zeros((capacity, num_envs), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def is_full(self):
        """检查缓冲区是否已满"""
        return self.size >= self.capacity
        
    def add(self, states, actions, rewards, dones, log_probs, values):
        """
        添加一个转换到缓冲区，支持多环境数据
        """
        # 确保输入是numpy数组
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.cpu().numpy()
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
            
        # 存储数据
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.log_probs[self.ptr] = log_probs.reshape(-1)  # 确保是一维的
        self.values[self.ptr] = values.reshape(-1)  # 确保是一维的
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        计算每个环境的返回值和优势估计
        
        参数:
            last_value: 最后状态的值函数估计，形状为 [num_envs,] 或 [num_envs, 1]
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
        """
        # 确保last_value是numpy数组并且形状正确
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.cpu().detach().numpy()
        
        if last_value.ndim == 2:
            last_value = last_value.reshape(-1)
            
        # 初始化数组
        advantages = np.zeros((self.size, self.num_envs), dtype=np.float32)
        returns = np.zeros((self.size, self.num_envs), dtype=np.float32)
        
        # 对每个环境分别计算
        next_values = last_value
        next_advantages = np.zeros(self.num_envs, dtype=np.float32)
        
        # 从后向前计算
        for t in reversed(range(self.size)):
            # 非终止状态标志
            if t == self.size - 1:
                next_non_terminals = 1.0 - self.dones[t]
            else:
                next_non_terminals = 1.0 - self.dones[t]
                
            # 计算时序差分误差
            delta = (self.rewards[t] +
                    gamma * next_values * next_non_terminals -
                    self.values[t])
            
            # 计算GAE
            advantages[t] = delta + gamma * gae_lambda * next_non_terminals * next_advantages
            
            # 更新下一步的值
            next_advantages = advantages[t]
            next_values = self.values[t]
            
        # 计算回报
        returns = advantages + self.values[:self.size]
            
        # 数值安全检查和归一化
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # 对每个环境分别进行归一化
        if self.size > 1:
            adv_mean = advantages.mean(0, keepdims=True)
            adv_std = advantages.std(0, keepdims=True)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # 限制极端值
        advantages = np.clip(advantages, -5.0, 5.0)
        
        # 更新缓冲区
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = returns
        
    def get_all(self):
        """获取所有数据"""
        if self.is_discrete:
            # 离散动作 - 需要转换为LongTensor并确保是二维[batch_size, 1]
            actions = torch.LongTensor(self.actions[:self.size]).view(-1, 1).to(self.device)
        else:
            # 连续动作 - 转换为FloatTensor
            actions = to_tensor(self.actions[:self.size], self.device)
            
        # 确保advantages和returns是二维的 [batch_size, 1]
        advantages = to_tensor(self.advantages[:self.size], self.device)
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
            
        returns = to_tensor(self.returns[:self.size], self.device)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
            
        return (
            to_tensor(self.states[:self.size], self.device),
            actions,
            to_tensor(self.log_probs[:self.size], self.device),
            returns,
            advantages
        )
        
    def get_batch(self, batch_indices=None):
        """
        获取指定索引的批次数据，如果未指定索引则获取随机批次
        
        参数:
            batch_indices: 可选，指定的数据索引
        
        返回:
            批次数据元组：(states, actions, log_probs, returns, advantages)
        """
        if batch_indices is None:
            batch_size = min(self.mini_batch_size, self.size)
            batch_indices = np.random.randint(0, self.size, size=batch_size)
            
        if self.is_discrete:
            # 离散动作 - 需要转换为LongTensor并确保是二维[batch_size, 1]，除非后续处理需要一维
            actions = torch.LongTensor(self.actions[batch_indices]).to(self.device)
            # 注意：不再强制reshape，因为evaluate_actions会根据维度做不同处理
        else:
            # 连续动作 - 转换为FloatTensor
            actions = to_tensor(self.actions[batch_indices], self.device)
            
        # 确保advantages和returns是二维的 [batch_size, 1]
        advantages = to_tensor(self.advantages[batch_indices], self.device)
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
            
        returns = to_tensor(self.returns[batch_indices], self.device)
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
            
        return (
            to_tensor(self.states[batch_indices], self.device),
            actions,
            to_tensor(self.log_probs[batch_indices], self.device),
            returns,
            advantages
        )
        
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
        
    def reset(self):
        """重置缓冲区，与clear方法功能相同"""
        self.clear()
        
    def insert(self, actions, values, log_probs, rewards, dones):
        """
        批量插入数据到缓冲区
        
        参数:
            actions: 动作数组
            values: 值函数预测数组
            log_probs: 动作概率对数数组 
            rewards: 奖励数组
            dones: 回合结束标志数组
        """
        # 注意：状态已经在rollout_buffer.states中正确设置
        # 这个方法只需要处理其他数据
        
        batch_size = len(rewards)
        
        # 确保我们不会超出缓冲区容量
        if self.ptr + batch_size > self.capacity:
            # 如果数据超出容量，只存储能容纳的部分
            overflow = (self.ptr + batch_size) - self.capacity
            batch_size -= overflow
        
        # 批量存储数据
        if batch_size > 0:
            # 直接使用切片赋值，提高效率
            end_idx = self.ptr + batch_size
            
            # 根据动作空间类型处理动作
            if self.is_discrete:
                # 离散动作 - 确保是整数
                for i in range(batch_size):
                    if isinstance(actions[i], np.ndarray):
                        self.actions[self.ptr + i] = int(actions[i].item()) if actions[i].size == 1 else int(actions[i][0])
                    else:
                        self.actions[self.ptr + i] = int(actions[i])
            else:
                # 连续动作 - 直接存储
                self.actions[self.ptr:end_idx] = actions[:batch_size]
            
            # 批量赋值其他数据
            self.values[self.ptr:end_idx] = values[:batch_size]
            self.log_probs[self.ptr:end_idx] = log_probs[:batch_size].reshape(-1, 1)
            self.rewards[self.ptr:end_idx] = rewards[:batch_size]
            self.dones[self.ptr:end_idx] = dones[:batch_size]
            
            # 更新指针和大小
            self.ptr = (self.ptr + batch_size) % self.capacity
            self.size = min(self.size + batch_size, self.capacity)

    def get_batches(self):
        """
        生成小批次数据的迭代器
        
        返回:
            迭代器，每次返回一个批次的数据：(states, actions, log_probs, returns, advantages)
        """
        indices = np.arange(self.size * self.num_envs)
        np.random.shuffle(indices)
        
        # 重塑数据以合并时间步和环境维度
        states = self.states[:self.size].reshape(-1, *self.states.shape[2:])
        actions = self.actions[:self.size].reshape(-1, *self.actions.shape[2:]) if self.actions.ndim > 2 else self.actions[:self.size].reshape(-1)
        log_probs = self.log_probs[:self.size].reshape(-1)
        returns = self.returns[:self.size].reshape(-1)
        advantages = self.advantages[:self.size].reshape(-1)
        
        # 生成小批次
        start_idx = 0
        batch_size = self.mini_batch_size
        while start_idx < len(indices):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            if self.is_discrete:
                batch_actions = torch.LongTensor(actions[batch_indices]).to(self.device)
            else:
                batch_actions = to_tensor(actions[batch_indices], self.device)
                
            # 确保返回值和优势估计是二维的
            batch_returns = to_tensor(returns[batch_indices], self.device).unsqueeze(1)
            batch_advantages = to_tensor(advantages[batch_indices], self.device).unsqueeze(1)
            
            yield (
                to_tensor(states[batch_indices], self.device),
                batch_actions,
                to_tensor(log_probs[batch_indices], self.device),
                batch_returns,
                batch_advantages
            )
            
            start_idx += batch_size


class Logger:
    """日志记录器，用于记录训练过程中的各种信息"""
    
    def __init__(self, log_dir):
        """初始化日志记录器"""
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.data = {}
        # 存储所有回合的奖励
        self.episode_rewards = []
        # 存储最近100个回合的奖励
        self.episode_rewards_100 = deque(maxlen=100)
        
        # 创建文本日志文件
        self.log_file_path = os.path.join(log_dir, 'training_log.txt')
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        self.log_to_file(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_to_file(f"日志目录: {log_dir}")
        self.log_to_file("-" * 50)
        
    def log_episode(self, episode_reward):
        """记录回合奖励"""
        # 添加到全部奖励列表
        self.episode_rewards.append(episode_reward)
        # 添加到最近100回合的队列
        self.episode_rewards_100.append(episode_reward)
        
    def log_step(self, step, info_dict):
        """记录每一步的信息"""
        for key, value in info_dict.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append((step, value))
    
    def log_to_file(self, message):
        """将消息写入日志文件"""
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.write(f"{message}\n")
            self.log_file.flush()  # 确保立即写入文件
            
    def log_training_info(self, update, num_updates, total_steps, total_timesteps, 
                          avg_reward, avg_reward_100, policy_loss, value_loss, 
                          entropy, steps_per_sec, remaining_time):
        """记录训练信息到日志文件"""
        message = f"\n更新: {update}/{num_updates}, 步数: {total_steps}/{total_timesteps}\n"
        message += f"平均奖励: {avg_reward:.2f}, 最近100回合平均: {avg_reward_100:.2f}\n"
        message += f"策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}, 熵: {entropy:.4f}\n"
        message += f"FPS: {steps_per_sec:.1f}, 预计剩余时间: {int(remaining_time//60)}分 {int(remaining_time%60)}秒"
        
        self.log_to_file(message)

    def log_model_saved(self, model_path):
        """记录模型保存信息"""
        self.log_to_file(f"模型已保存: {model_path}")
        
    def get_avg_reward(self):
        """获取最近100回合的平均奖励"""
        if not self.episode_rewards_100:
            return 0.0
        return np.mean(self.episode_rewards_100)
        
    def dump(self, filename='progress.csv'):
        """将数据保存到CSV文件"""
        import pandas as pd
        
        all_data = []
        for key, values in self.data.items():
            steps, vals = zip(*values)
            df = pd.DataFrame({
                'Step': steps,
                key: vals
            })
            all_data.append(df)
            
        if all_data:
            result = all_data[0]
            for df in all_data[1:]:
                result = pd.merge(result, df, on='Step', how='outer')
                
            result.to_csv(os.path.join(self.log_dir, filename), index=False)
    
    def close(self):
        """关闭日志文件"""
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_to_file(f"\n训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.log_file.close()
            
    def plot_rewards(self, save=True, show=False):
        """绘制回合奖励曲线"""
        steps, rewards = zip(*self.data.get('episode_reward', []))
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards)
        plt.title('回合奖励')
        plt.xlabel('步数')
        plt.ylabel('奖励')
        
        if save:
            plt.savefig(os.path.join(self.log_dir, 'rewards.png'))
        if show:
            plt.show()
        plt.close()
            
    def plot_metrics(self, metrics=None, save=True, show=False):
        """绘制指定的指标"""
        if metrics is None:
            metrics = [key for key in self.data.keys() if key != 'episode_reward']
            
        for metric in metrics:
            if metric in self.data:
                steps, values = zip(*self.data[metric])
                
                plt.figure(figsize=(10, 5))
                plt.plot(steps, values)
                plt.title(f'{metric}')
                plt.xlabel('步数')
                plt.ylabel(metric)
                
                if save:
                    plt.savefig(os.path.join(self.log_dir, f'{metric}.png'))
                if show:
                    plt.show()
                plt.close() 

