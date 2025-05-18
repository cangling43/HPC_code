import ray
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Any, Tuple
from gymnasium.spaces import Box, Discrete

@ray.remote
class RemoteEnvWorker:
    """远程环境工作器，用于异步环境交互"""
    def __init__(self, env_creator, worker_id: int, seed: int):
        # 使用新版本 Gymnasium 的方式设置随机种子
        self.env = env_creator()
        # 新版本 Gymnasium 在 reset 时设置随机种子
        self.worker_id = worker_id
        self.seed = seed + worker_id
        self.state = None
        self.episode_reward = 0
        self.episode_length = 0
        
    def reset(self):
        """重置环境"""
        # 在 reset 时设置随机种子
        self.state, _ = self.env.reset(seed=self.seed)  # 适配新版gym返回格式
        self.episode_reward = 0
        self.episode_length = 0
        return self.state
        
    def step(self, action):
        """异步执行环境步进"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.state = next_state
        self.episode_reward += reward
        self.episode_length += 1
        
        if done:
            # 重置环境
            self.state, _ = self.env.reset(seed=self.seed)
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
            self.episode_reward = 0
            self.episode_length = 0
            
        return self.state, reward, done, info
        
    def get_state(self):
        """获取当前环境状态"""
        return self.state

    def close(self):
        """关闭环境"""
        if hasattr(self, 'env'):
            self.env.close()

class AsyncVecEnv:
    """异步向量化环境"""
    def __init__(self, env_creator, num_workers: int, seed: int):
        # 创建远程环境工作器
        self.workers = [
            RemoteEnvWorker.remote(env_creator, i, seed)
            for i in range(num_workers)
        ]
        self.num_workers = num_workers
        
        # 获取环境信息
        temp_env = env_creator()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()
        
        # 状态缓冲区
        self.state_buffer = {}
        
    def reset(self):
        """重置所有环境"""
        # 异步重置所有环境
        futures = [worker.reset.remote() for worker in self.workers]
        states = ray.get(futures)
        
        # 更新状态缓冲区
        self.state_buffer = {i: state for i, state in enumerate(states)}
        
        # 确保返回 numpy 数组
        return np.array(states)
        
    def step(self, actions):
        """异步执行环境步进
        
        参数:
            actions: numpy数组，每个环境的动作
            
        返回:
            states: 下一个状态列表
            rewards: 奖励列表
            dones: 完成标志列表
            infos: 信息字典列表
        """
        # 异步执行动作
        futures = [
            worker.step.remote(action)
            for worker, action in zip(self.workers, actions)
        ]
        results = ray.get(futures)
        
        # 解包结果
        states, rewards, dones, infos = zip(*results)
        
        # 更新状态缓冲区
        self.state_buffer = {i: state for i, state in enumerate(states)}
        
        # 确保返回 numpy 数组
        return (
            np.array(states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos
        )
        
    def close(self):
        """关闭所有环境"""
        ray.get([worker.close.remote() for worker in self.workers])    