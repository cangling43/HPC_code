import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import queue
import time
import os
from typing import Dict, List, Optional
from models import ActorCritic
from async_environment import AsyncVecEnv
from utils import RolloutBuffer, Logger

@ray.remote
class ModelReplica:
    """模型副本，用于分布式训练"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, device="cpu"):
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim)
        self.version = 0
        self.device = torch.device(device)
        self.actor_critic.to(self.device)
        
    def get_action(self, state):
        """获取动作"""
        with torch.no_grad():
            # 确保输入张量在正确的设备上
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            state = state.to(self.device)
            
            # 获取动作
            action, log_prob, value = self.actor_critic.get_action(state)
            
            # 将结果移回CPU
            return (action.cpu(),
                    log_prob.cpu(),
                    value.cpu())
            
    def update_weights(self, weights, version):
        """更新模型权重"""
        if version > self.version:
            # 确保权重在正确的设备上
            weights = {k: v.to(self.device) for k, v in weights.items()}
            self.actor_critic.load_state_dict(weights)
            self.version = version
            
    def get_version(self):
        """获取模型版本"""
        return self.version

class AsyncPPO:
    """异步PPO算法实现"""
    def __init__(self,
                 env_creator,
                 num_workers: int = 8,
                 num_model_replicas: int = 4,
                 steps_per_worker: int = 128,
                 epochs: int = 4,
                 mini_batch_size: int = 64,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_param: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 seed: int = 0,
                 device: str = None,
                 log_dir: str = "./logs"):
                 
        # 初始化Ray（如果尚未初始化）
        if not ray.is_initialized():
            ray.init()
            
        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 创建异步环境
        self.env = AsyncVecEnv(env_creator, num_workers, seed)
        
        # 获取环境信息
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = (self.env.action_space.n 
                          if hasattr(self.env.action_space, 'n')
                          else self.env.action_space.shape[0])
        
        # 创建主模型
        self.actor_critic = ActorCritic(
            self.state_dim,
            self.action_dim,
            hidden_dim=64
        ).to(self.device)
        
        # 创建模型副本，确保使用相同的设备
        self.model_replicas = [
            ModelReplica.remote(self.state_dim, self.action_dim, device=self.device.type)
            for _ in range(num_model_replicas)
        ]
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # 保存超参数
        self.num_workers = num_workers
        self.steps_per_worker = steps_per_worker
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建经验回放缓冲区，添加 num_workers 参数
        buffer_size = steps_per_worker
        self.rollout_buffer = RolloutBuffer(
            buffer_size,
            self.state_dim,
            self.action_dim,
            self.device,
            mini_batch_size=mini_batch_size,
            num_envs=num_workers  # 添加这个参数
        )
        
        # 创建日志记录器
        self.logger = Logger(log_dir)
        
        # 异步更新相关
        self.model_version = 0
        self.update_event = threading.Event()
        self.sample_queue = queue.Queue(maxsize=num_workers * 2)
        self.stop_flag = threading.Event()
        
        # 启动异步采样线程
        self.sample_thread = threading.Thread(
            target=self._async_sample_loop,
            daemon=True
        )
        self.sample_thread.start()
        
        # 启动模型更新线程
        self.update_thread = threading.Thread(
            target=self._async_update_loop,
            daemon=True
        )
        self.update_thread.start()
        
    def _async_sample_loop(self):
        """异步采样循环"""
        states = self.env.reset()
        
        while not self.stop_flag.is_set():
            # 使用随机选择的模型副本进行采样
            replica = np.random.choice(self.model_replicas)
            
            # 获取动作
            actions, log_probs, values = ray.get(
                replica.get_action.remote(states)
            )
            
            # 执行环境步进
            next_states, rewards, dones, infos = self.env.step(actions.numpy())
            
            # 存储采样数据 - 现在传入整个批次的数据
            self.rollout_buffer.add(
                states,
                actions.numpy(),
                rewards,
                dones,
                log_probs.numpy(),
                values.numpy()
            )
            
            # 更新状态
            states = next_states
            
            # 检查是否需要更新
            if self.rollout_buffer.is_full():
                self.update_event.set()
                self.rollout_buffer = RolloutBuffer(
                    self.rollout_buffer.capacity,
                    self.state_dim,
                    self.action_dim,
                    self.device,
                    mini_batch_size=self.mini_batch_size,
                    num_envs=self.num_workers  # 添加这个参数
                )
                
    def _async_update_loop(self):
        """异步更新循环"""
        while not self.stop_flag.is_set():
            # 等待采样完成
            self.update_event.wait()
            self.update_event.clear()
            
            # 计算优势估计
            with torch.no_grad():
                # 将字典值转换为列表，然后转换为张量
                last_states = list(self.env.state_buffer.values())
                last_states = np.array(last_states)  # 确保是numpy数组
                last_values = self.actor_critic.get_value(
                    torch.FloatTensor(last_states).to(self.device)
                )
                
            self.rollout_buffer.compute_returns_and_advantages(
                last_values.cpu().numpy(),
                self.gamma,
                self.gae_lambda
            )
            
            # 执行多轮训练
            for _ in range(self.epochs):
                for batch in self.rollout_buffer.get_batches():
                    states, actions, old_log_probs, returns, advantages = batch
                    
                    # 计算策略损失
                    new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                        states, actions
                    )
                    
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                      1.0 + self.clip_param) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 计算价值损失
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                    
                    # 计算熵损失
                    entropy_loss = -entropy.mean()
                    
                    # 总损失
                    loss = (policy_loss +
                           self.value_coef * value_loss +
                           self.entropy_coef * entropy_loss)
                           
                    # 更新模型
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(),
                        self.max_grad_norm
                    )
                    self.optimizer.step()
                    
            # 更新模型版本并同步到副本
            self.model_version += 1
            # 确保权重在CPU上
            weights = {k: v.cpu() for k, v in self.actor_critic.state_dict().items()}
            futures = [
                replica.update_weights.remote(weights, self.model_version)
                for replica in self.model_replicas
            ]
            ray.get(futures)
            
    def train(self, total_timesteps: int, log_interval: int = 10, save_interval: int = 50):
        """训练入口
        
        参数:
            total_timesteps: 总训练步数
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
        """
        num_updates = total_timesteps // (self.num_workers * self.steps_per_worker)
        start_time = time.time()
        
        print(f"\n开始训练 - 总步数: {total_timesteps}, 预计更新次数: {num_updates}\n")
        
        for update in range(num_updates):
            # 等待一次更新完成
            self.update_event.wait()
            self.update_event.clear()
            
            current_steps = (update + 1) * self.num_workers * self.steps_per_worker
            progress = current_steps / total_timesteps * 100
            elapsed_time = time.time() - start_time
            steps_per_sec = current_steps / elapsed_time
            remaining_steps = total_timesteps - current_steps
            remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # 记录日志
            if update % log_interval == 0:
                print(f"\r进度: {progress:.1f}% [{current_steps}/{total_timesteps}] "
                      f"更新: {update}/{num_updates} "
                      f"FPS: {steps_per_sec:.1f} "
                      f"预计剩余时间: {int(remaining_time//60)}分{int(remaining_time%60)}秒", 
                      end="", flush=True)
                
                self.logger.log_step(
                    current_steps,
                    {
                        'model_version': self.model_version,
                        'fps': steps_per_sec,
                        'progress': progress,
                        'remaining_time': remaining_time
                    }
                )
                
            # 保存模型
            if save_interval > 0 and update % save_interval == 0:
                save_path = os.path.join(self.logger.log_dir, f'model_{update}.pt')
                self.save(save_path)
                print(f"\n模型已保存: {save_path}")
                
        # 停止训练
        self.stop_flag.set()
        self.sample_thread.join()
        self.update_thread.join()
        self.env.close()
        
        # 保存最终模型
        final_path = os.path.join(self.logger.log_dir, 'model_final.pt')
        self.save(final_path)
        print(f"\n训练完成 - 最终模型已保存: {final_path}")
        print(f"总训练时间: {int((time.time() - start_time)//60)}分{int((time.time() - start_time)%60)}秒")
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_version': self.model_version
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_version = checkpoint['model_version']
        
        # 同步到所有副本
        weights = self.actor_critic.state_dict()
        futures = [
            replica.update_weights.remote(weights, self.model_version)
            for replica in self.model_replicas
        ]
        ray.get(futures) 