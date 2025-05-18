import argparse
import time
import os
import ray
import torch
import numpy as np
import gym
import gymnasium
from parallel_ppo import ParallelPPO, DistributedPPO
from collections import deque
from async_ppo import AsyncPPO
from async_environment import AsyncVecEnv

def get_env_info(env_name):
    """获取环境信息"""
    try:
        # 尝试使用gymnasium
        env = gymnasium.make(env_name)
    except (ImportError, ModuleNotFoundError):
        # 使用gym
        env = gym.make(env_name)
    
    # 获取状态和动作空间信息
    state_dim = env.observation_space.shape[0]
    
    # 更准确地检查动作空间类型
    if hasattr(env.action_space, 'shape') and len(env.action_space.shape) > 0:
        # 连续动作空间
        action_dim = env.action_space.shape[0]
        action_bounds = [float(env.action_space.low[0]), float(env.action_space.high[0])]
        is_discrete = False
    else:
        # 离散动作空间
        action_dim = env.action_space.n
        action_bounds = None
        is_discrete = True
    
    # 关闭环境
    env.close()
    
    return {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'action_bounds': action_bounds,
        'is_discrete': is_discrete,
        'env_name': env_name
    }

def auto_select_hyperparams(env_info, cpu_count=None, available_memory_gb=None):
    """
    根据环境特性自动选择超参数
    
    参数:
        env_info: 环境信息字典
        cpu_count: CPU核心数，如果为None则自动检测
        available_memory_gb: 可用内存(GB)，如果为None则使用默认值
    
    返回:
        hyperparams: 超参数字典
    """
    import multiprocessing
    
    # 如果未指定CPU核心数，则自动检测
    if cpu_count is None:
        cpu_count = multiprocessing.cpu_count()
    
    # 如果未指定可用内存，则使用默认值
    if available_memory_gb is None:
        available_memory_gb = 4  # 假设至少有4GB内存可用
        
    # 获取环境信息
    state_dim = env_info['state_dim']
    action_dim = env_info['action_dim']
    is_discrete = env_info['is_discrete']
    env_name = env_info['env_name']
    
    # 根据环境名称设置特定参数
    env_specific_params = {}
    if 'Pendulum' in env_name:
        # 摆锤环境优化参数 - 这些参数在实践中表现良好
        env_specific_params = {
            'lr': 3e-4,  # 更保守的学习率
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.01,  # 增加熵以促进更好的探索
            'max_grad_norm': 0.75,  # 更宽松的梯度裁剪
            'gae_lambda': 0.95,
            'epochs': 4,  # 减少训练轮数，防止过拟合
        }
    elif 'CartPole' in env_name:
        env_specific_params = {
            'lr': 2.5e-4,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        }
    elif 'MountainCar' in env_name:
        env_specific_params = {
            'lr': 2e-4,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        }
    elif 'Acrobot' in env_name:
        env_specific_params = {
            'lr': 3e-4,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        }
    elif 'LunarLander' in env_name:
        env_specific_params = {
            'lr': 1e-4,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        }
    elif 'BipedalWalker' in env_name:
        env_specific_params = {
            'lr': 5e-5,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.001,
            'max_grad_norm': 0.5,
        }
    elif any(key in env_name for key in ['Humanoid', 'Ant', 'HalfCheetah', 'Hopper', 'Walker']):
        # MuJoCo环境
        env_specific_params = {
            'lr': 3e-5,
            'gamma': 0.99,
            'clip_param': 0.2,
            'entropy_coef': 0.001,
            'max_grad_norm': 0.5,
        }
    
    # 根据状态和动作空间大小调整网络参数
    # 状态空间维度越高，网络应该越大
    if state_dim <= 10:
        hidden_dim = 256
    elif state_dim <= 50:
        hidden_dim = 512
    else:
        hidden_dim = 1024
    
    # 根据动作空间类型调整策略参数
    if is_discrete:
        # 离散动作空间 - 动作空间越大，熵系数应该越小
        if action_dim < 5:
            entropy_coef = env_specific_params.get('entropy_coef', 0.01)
        elif action_dim < 20:
            entropy_coef = env_specific_params.get('entropy_coef', 0.005)
        else:
            entropy_coef = env_specific_params.get('entropy_coef', 0.001)
        
        value_coef = 0.5
    else:
        # 连续动作空间 - 需要更合适的熵系数和价值系数
        entropy_coef = env_specific_params.get('entropy_coef', 0.01)  # 对于摆锤等环境，保持更高的探索度
        value_coef = 0.5  # 提高价值估计的权重，帮助稳定学习
    
    # 根据CPU核心数调整并行参数
    # 保留1-2个核心给操作系统
    num_workers = min(max(2, cpu_count - 1), 16)  # 使用更多核心，保留1个给系统
    
    # 根据环境和可用资源调整步数和批次大小
    if 'Pendulum' in env_name:
        # 摆锤环境需要更多样本才能学习有效策略
        steps_per_worker = 512  # 增加步数
        # 较大的批次有助于更稳定的学习
        mini_batch_size = min(256, max(64, num_workers * steps_per_worker // 8))
    else:
        # 其他环境使用通用设置
        steps_per_worker = 256
        total_steps = num_workers * steps_per_worker
        # 根据总steps数调整mini_batch_size
        mini_batch_size = min(128, max(32, total_steps // 16))
    
    # 根据环境复杂度和并行度调整总训练步数
    if 'Pendulum' in env_name:
        # 摆锤环境通常需要更长的训练
        total_timesteps = 2000000
    elif 'CartPole' in env_name:
        total_timesteps = 1000000
    elif 'MountainCar' in env_name or 'Acrobot' in env_name:
        total_timesteps = 2000000
    elif 'LunarLander' in env_name:
        total_timesteps = 2000000
    elif 'BipedalWalker' in env_name:
        total_timesteps = 5000000
    elif any(key in env_name for key in ['Humanoid', 'Ant', 'HalfCheetah', 'Hopper', 'Walker']):
        total_timesteps = 10000000
    else:
        # 默认值
        total_timesteps = 2000000
    
    # 增加随机性，使不同运行有所区别
    random_seed = int(time.time()) % 10000
    
    # 默认使用并行模式
    mode = "parallel"  # 始终使用并行模式，通常更稳定
    
    # 根据批次大小和训练长度调整更新频率
    epochs = env_specific_params.get('epochs', 10)
    
    # 组合所有参数
    hyperparams = {
        # 并行参数
        'mode': mode,
        'num_workers': num_workers,
        'num_gpus': 0,  # 默认不使用GPU
        
        # 算法超参数
        'total_timesteps': total_timesteps,
        'lr': env_specific_params.get('lr', 3e-4),
        'gamma': env_specific_params.get('gamma', 0.99),
        'gae_lambda': env_specific_params.get('gae_lambda', 0.95),
        'clip_param': env_specific_params.get('clip_param', 0.2),
        'value_coef': value_coef,
        'entropy_coef': entropy_coef,
        'max_grad_norm': env_specific_params.get('max_grad_norm', 0.5),
        'steps_per_worker': steps_per_worker,
        'epochs': epochs,
        'mini_batch_size': mini_batch_size,
        
        # 其他参数
        'hidden_dim': hidden_dim,
        'seed': random_seed,
    }
    
    return hyperparams

class AutoAdjustParams:
    """参数自动调整器，训练过程中动态调整超参数"""
    
    def __init__(self, initial_params, enable=True, env_name=None):
        """
        初始化参数调整器
        
        参数:
            initial_params: 初始超参数字典
            enable: 是否启用自动调整
            env_name: 环境名称，用于特定环境的参数优化
        """
        self.params = initial_params.copy()
        self.initial_params = initial_params.copy()
        self.enable = enable
        self.updates = 0
        self.last_mean_reward = float('-inf')
        self.best_mean_reward = float('-inf')
        self.stagnation_count = 0
        # 连续停滞计数
        self.consecutive_stagnation = 0
        # 记录最近N次的奖励，用于检测训练停滞
        self.recent_mean_rewards = deque(maxlen=10)
        # 调整间隔控制
        self.adjust_interval = 0
        self.adjustment_history = []
        
        # 记录环境名称用于特定环境的参数优化
        self.env_name = env_name if env_name else "Unknown"
        
        # 针对不同环境的初始参数调整
        if self.env_name:
            self.adjustment_history.append(("初始化", f"环境: {self.env_name}", ""))
            
            # Pendulum环境特殊处理
            if "Pendulum" in self.env_name:
                # Pendulum环境通常需要更高的学习率和较低的熵系数
                if self.initial_params['lr'] < 3e-4:
                    self.params['lr'] = 3e-4
                    self.initial_params['lr'] = 3e-4
                    self.adjustment_history.append(("Pendulum优化", "初始学习率调整", self.params['lr']))
                
                # 调整熵系数为Pendulum较优值
                optimal_entropy = 0.01
                if abs(self.params['entropy_coef'] - optimal_entropy) > 0.005:
                    self.params['entropy_coef'] = optimal_entropy
                    self.adjustment_history.append(("Pendulum优化", "初始熵系数调整", self.params['entropy_coef']))
            
            # 对于其他环境可以添加类似的优化逻辑
            elif "CartPole" in self.env_name:
                # CartPole环境参数优化
                pass
                
        # 确保参数范围合理
        self._validate_params()
    
    def _validate_params(self):
        """验证参数是否在合理范围内，并修正不合理值"""
        # 学习率边界
        if self.params['lr'] < 1e-6 or self.params['lr'] > 1e-1:
            default_lr = 3e-4
            self.params['lr'] = default_lr
            self.initial_params['lr'] = default_lr
            self.adjustment_history.append(("参数校正", "学习率超出范围，重置为默认值", default_lr))
        
        # 熵系数边界
        if self.params['entropy_coef'] < 0 or self.params['entropy_coef'] > 0.2:
            default_entropy = 0.01
            self.params['entropy_coef'] = default_entropy
            self.initial_params['entropy_coef'] = default_entropy
            self.adjustment_history.append(("参数校正", "熵系数超出范围，重置为默认值", default_entropy))
        
        # 梯度裁剪参数边界
        if self.params['max_grad_norm'] <= 0 or self.params['max_grad_norm'] > 10:
            default_grad_norm = 0.5
            self.params['max_grad_norm'] = default_grad_norm
            self.initial_params['max_grad_norm'] = default_grad_norm
            self.adjustment_history.append(("参数校正", "梯度裁剪参数超出范围，重置为默认值", default_grad_norm))
    
    def adjust_params(self, mean_reward, recent_rewards, step_progress):
        """
        根据训练进度和奖励调整参数
        
        参数:
            mean_reward: 当前平均奖励
            recent_rewards: 最近N个回合的奖励列表
            step_progress: 训练进度 (0.0 ~ 1.0)
            
        返回:
            adjusted_params: 调整后的参数字典
        """
        if not self.enable:
            return self.params
        
        self.updates += 1
        
        # 减小调整频率，每5次更新才考虑调整一次参数
        self.adjust_interval += 1
        if self.adjust_interval < 5:
            return self.params
        
        self.adjust_interval = 0
        
        # 添加当前奖励到历史记录
        self.recent_mean_rewards.append(mean_reward)
        
        # 判断是否有改进
        improvement = mean_reward > self.last_mean_reward + 1.0  # 添加一个阈值，避免微小改进就认为有进展
        
        # 如果有足够多的历史记录，检测训练是否停滞
        is_stagnant = False
        if len(self.recent_mean_rewards) >= 5:
            # 计算最近5次奖励的标准差，如果很小则说明训练停滞
            reward_std = np.std(list(self.recent_mean_rewards)[-5:])
            # 如果标准差小于1.0且平均奖励没有达到目标，认为训练停滞
            target_reward = -200  # Pendulum环境的目标奖励约为-200左右
            current_avg = np.mean(list(self.recent_mean_rewards)[-5:])
            is_stagnant = reward_std < 1.0 and current_avg < target_reward
        
        # 记录最佳奖励
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.stagnation_count = 0
            self.consecutive_stagnation = 0
            # 如果有明显改进，记录一次成功的参数组合
            if improvement and mean_reward > -300:
                self.adjustment_history.append(("成功配置", f"奖励={mean_reward:.2f}", 
                                              f"lr={self.params['lr']:.6f}, ent={self.params['entropy_coef']:.6f}"))
        else:
            self.stagnation_count += 1
            if is_stagnant:
                self.consecutive_stagnation += 1
            else:
                self.consecutive_stagnation = 0
        
        # 记录奖励变化
        reward_improved = improvement
        self.last_mean_reward = mean_reward
        
        # 计算奖励的标准差，用于判断稳定性
        if len(recent_rewards) > 1:
            reward_std = np.std(recent_rewards)
            reward_stability = reward_std / (abs(mean_reward) + 1e-8)
        else:
            reward_stability = 1.0
        
        # ===== 修改策略：更智能的参数调整 =====
        
        # 1. 学习率调整策略
        
        # 训练初期保持较高学习率
        if step_progress < 0.3:
            # 训练前30%保持接近初始学习率
            min_lr = self.initial_params['lr'] * 0.5
            if self.params['lr'] < min_lr:
                self.params['lr'] = min_lr
                self.adjustment_history.append(("学习率", "重置", self.params['lr']))
        
        # 训练中后期正常衰减
        elif step_progress > 0.5:
            # 训练过半后缓慢减小学习率
            decay_factor = max(0.3, 1.0 - (step_progress - 0.5) * 1.0)
            target_lr = self.initial_params['lr'] * decay_factor
            # 平滑调整，不要一次性大幅变化
            self.params['lr'] = self.params['lr'] * 0.9 + target_lr * 0.1
        
        # 2. 训练停滞时的应对策略
        
        # 如果连续多次检测到停滞且奖励很差，采取激进措施
        if self.consecutive_stagnation >= 3:
            # 根据当前平均奖励决定采取何种恢复策略
            if mean_reward < -300:
                # 严重停滞情况：比较激进的重置
                # 策略1：恢复到接近初始学习率，但不超过初始值的两倍
                self.params['lr'] = min(self.initial_params['lr'] * 2.0, 
                                       max(self.initial_params['lr'], self.params['lr'] * 3.0))
                
                # 策略2：增大熵系数促进更多探索，但避免设置过高
                self.params['entropy_coef'] = min(0.05, 
                                               max(self.initial_params['entropy_coef'] * 2.0, 
                                                  self.params['entropy_coef'] * 3.0))
                
                # 策略3：重置梯度裁剪参数
                self.params['max_grad_norm'] = self.initial_params['max_grad_norm'] * 1.5
                
                self.adjustment_history.append(("严重停滞", "重置参数", 
                                              f"lr={self.params['lr']:.6f}, ent={self.params['entropy_coef']:.6f}"))
            else:
                # 轻微停滞情况：温和调整
                # 适度提高学习率，增加探索
                self.params['lr'] = max(self.params['lr'] * 1.5, self.initial_params['lr'] * 0.5)
                self.params['entropy_coef'] = min(0.03, self.params['entropy_coef'] * 1.5)
                
                self.adjustment_history.append(("轻微停滞", "温和调整", 
                                              f"lr={self.params['lr']:.6f}, ent={self.params['entropy_coef']:.6f}"))
            
            # 重置停滞计数
            self.consecutive_stagnation = 0
            
        # 处理长期停滞情况（长时间训练但奖励仍然很差）
        if step_progress > 0.7 and mean_reward < -300 and self.updates % 20 == 0:
            # 训练已经进行了70%但效果仍然很差，尝试周期性参数脉冲
            self.params['lr'] = self.initial_params['lr']  # 重置到初始学习率
            self.params['entropy_coef'] = self.initial_params['entropy_coef'] * 1.5  # 略高于初始熵系数
            self.adjustment_history.append(("训练后期停滞", "参数脉冲", 
                                          f"lr={self.params['lr']:.6f}, ent={self.params['entropy_coef']:.6f}"))
        
        # 针对Pendulum环境的特殊优化
        # Pendulum环境往往需要较高学习率和适度熵系数才能获得好的性能
        if step_progress > 0.5 and "Pendulum" in self.env_name:
            # 确保学习率不会太低
            min_pendulum_lr = 3e-4  # Pendulum环境推荐的最小学习率
            if self.params['lr'] < min_pendulum_lr:
                self.params['lr'] = min_pendulum_lr
                self.adjustment_history.append(("Pendulum优化", "调整学习率", self.params['lr']))
            
            # Pendulum环境中熵系数的推荐值
            optimal_pendulum_entropy = 0.01
            # 逐渐调整熵系数接近最优值
            self.params['entropy_coef'] = self.params['entropy_coef'] * 0.8 + optimal_pendulum_entropy * 0.2
            
        # 对于所有环境通用的学习率恢复机制
        # 如果奖励超过一定阈值但学习率过低，适当提高学习率以加速收敛
        if mean_reward > -250 and self.params['lr'] < self.initial_params['lr'] * 0.2:
            # 奖励不错但学习率太低，适度提高学习率
            self.params['lr'] = min(self.initial_params['lr'] * 0.5, self.params['lr'] * 2.0)
            self.adjustment_history.append(("学习率过低", "适度提高", self.params['lr']))
        
        # 3. 熵系数随训练进度自然衰减
        # 但即使在训练末期也保持最小熵促进一定探索
        min_entropy = self.initial_params['entropy_coef'] * 0.1
        natural_entropy = self.initial_params['entropy_coef'] * max(0.1, 1.0 - step_progress * 0.8)
        self.params['entropy_coef'] = max(min_entropy, min(self.params['entropy_coef'], natural_entropy))
        
        # 确保参数在合理范围内
        # 学习率设置更合理的下限，避免降至过低导致训练停滞
        self.params['lr'] = max(1e-4, min(1e-2, self.params['lr']))
        # 确保学习率不会降到太低，导致训练停滞
        if self.params['lr'] < 1e-4 and mean_reward < -250:
            self.params['lr'] = 1e-4
            self.adjustment_history.append(("学习率", "设置最小值", self.params['lr']))
            
        # 熵系数也设置更合理的上限和下限
        self.params['entropy_coef'] = max(1e-5, min(0.05, self.params['entropy_coef']))
        
        # 梯度裁剪参数的范围更加合理
        self.params['max_grad_norm'] = max(0.1, min(5.0, self.params['max_grad_norm']))
        
        return self.params
    
    def get_adjustment_summary(self):
        """获取参数调整历史摘要"""
        if not self.adjustment_history:
            return "未进行参数调整"
            
        summary = "参数自动调整历史:\n"
        for param, direction, value in self.adjustment_history[-15:]:  # 显示最近15次调整
            if isinstance(value, float):
                summary += f"- {param}: {direction}到 {value:.6f}\n"
            else:
                summary += f"- {param}: {direction} {value}\n"
                
        # 添加当前参数状态
        summary += f"\n当前参数状态:\n"
        summary += f"- 学习率: {self.params['lr']:.6f}\n"
        summary += f"- 熵系数: {self.params['entropy_coef']:.6f}\n"
        summary += f"- 梯度裁剪: {self.params['max_grad_norm']:.2f}\n"
        summary += f"- 最佳奖励: {self.best_mean_reward:.2f}\n"
        
        return summary

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="并行PPO算法训练脚本")
    
    # 环境参数
    parser.add_argument("--env_name", type=str, default="Pendulum-v1",
                        help="环境名称 (默认: Pendulum-v1)")
    
    # 并行化参数
    parser.add_argument("--mode", type=str, default=None, choices=["parallel", "distributed", "auto"],
                        help="并行模式: 'parallel'(单机多进程), 'distributed'(分布式), 或 'auto'(自动选择) (默认: auto)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="并行工作器数量 (默认: 自动选择)")
    parser.add_argument("--num_gpus", type=int, default=0,
                        help="使用的GPU数量 (默认: 0)")
    
    # 算法超参数
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="总训练步数 (默认: 自动选择)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (默认: 自动选择)")
    parser.add_argument("--gamma", type=float, default=None,
                        help="折扣因子 (默认: 自动选择)")
    parser.add_argument("--gae_lambda", type=float, default=None,
                        help="GAE lambda参数 (默认: 自动选择)")
    parser.add_argument("--clip_param", type=float, default=None,
                        help="PPO裁剪参数 (默认: 自动选择)")
    parser.add_argument("--value_coef", type=float, default=None,
                        help="价值损失系数 (默认: 自动选择)")
    parser.add_argument("--entropy_coef", type=float, default=None,
                        help="熵正则化系数 (默认: 自动选择)")
    parser.add_argument("--max_grad_norm", type=float, default=None,
                        help="梯度裁剪范数 (默认: 自动选择)")
    parser.add_argument("--steps_per_worker", type=int, default=None,
                        help="每个工作器采样的步数 (默认: 自动选择)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="每批数据的训练轮数 (默认: 自动选择)")
    parser.add_argument("--mini_batch_size", type=int, default=None,
                        help="小批量大小 (默认: 自动选择)")
    
    # 日志参数
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="日志目录 (默认: ./logs)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="日志记录间隔 (默认: 10)")
    parser.add_argument("--save_interval", type=int, default=25,
                        help="模型保存间隔 (默认: 25)")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子 (默认: 自动选择)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备 (默认: cuda)")
    parser.add_argument("--auto_params", action="store_true",
                        help="启用超参数自动选择和动态调整")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="神经网络隐藏层维度 (默认: 自动选择)")
    parser.add_argument('--async_mode', action='store_true',
                      help='是否使用异步训练模式')
    parser.add_argument('--num_model_replicas', type=int, default=4,
                      help='异步模式下的模型副本数量')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 获取环境信息
    env_info = get_env_info(args.env_name)
    print(f"环境信息: {env_info}")
    
    # 自动选择超参数或使用命令行参数
    if args.auto_params or (args.mode is None or
                           args.num_workers is None or 
                           args.total_timesteps is None or
                           args.lr is None):
        print("启用超参数自动选择...")
        auto_params = auto_select_hyperparams(env_info)
        
        # 命令行参数覆盖自动选择的参数
        for key, value in vars(args).items():
            if value is not None and key in auto_params:
                auto_params[key] = value
        
        # 更新args
        for key, value in auto_params.items():
            if hasattr(args, key):
                setattr(args, key, value)
            elif key == 'hidden_dim':  # 特别处理hidden_dim
                setattr(args, 'hidden_dim', value)
        
        print("自动选择的超参数:")
        for key, value in auto_params.items():
            print(f"  {key}: {value}")
    
    # 创建日志目录
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 时间戳作为实验ID
    experiment_id = int(time.time())
    log_dir = os.path.join(log_dir, f"{args.env_name}_{args.mode}_{experiment_id}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 创建一个总体训练日志文件
    with open(os.path.join(log_dir, 'experiment_info.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=========== 实验信息 ===========\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"环境: {args.env_name}\n")
        f.write(f"自动参数: {'启用' if args.auto_params else '禁用'}\n")
        f.write(f"并行模式: {args.mode}\n")
        f.write(f"工作器数量: {args.num_workers}\n")
        f.write(f"计算设备: {args.device}\n")
        f.write(f"总训练步数: {args.total_timesteps}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"折扣因子: {args.gamma}\n")
        f.write(f"GAE lambda: {args.gae_lambda}\n")
        f.write(f"PPO裁剪参数: {args.clip_param}\n")
        f.write(f"价值损失系数: {args.value_coef}\n")
        f.write(f"熵正则化系数: {args.entropy_coef}\n")
        f.write(f"梯度裁剪范数: {args.max_grad_norm}\n")
        f.write(f"每批数据的训练轮数: {args.epochs}\n")
        f.write(f"小批量大小: {args.mini_batch_size}\n")
        f.write(f"每个工作器采样的步数: {args.steps_per_worker}\n")
        f.write(f"随机种子: {args.seed}\n")
        f.write(f"神经网络隐藏层维度: {getattr(args, 'hidden_dim', 256)}\n")
        f.write(f"日志目录: {log_dir}\n")
        f.write(f"环境信息: {env_info}\n")
        f.write(f"================================\n\n")
    
    # 初始化参数自动调整器
    param_adjuster = AutoAdjustParams(
        initial_params={
            'lr': args.lr,
            'entropy_coef': args.entropy_coef,
            'max_grad_norm': args.max_grad_norm
        },
        enable=args.auto_params,
        env_name=args.env_name
    )
    
    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            num_gpus=args.num_gpus,
            object_store_memory=2 * 10**9,  # 2GB对象存储
            _memory=4 * 10**9,  # 4GB堆内存
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False
        )
    
    print(f"=========== 开始训练 ===========")
    print(f"环境: {args.env_name}")
    print(f"并行模式: {args.mode}")
    print(f"工作器数量: {args.num_workers}")
    print(f"计算设备: {args.device}")
    print(f"日志目录: {log_dir}")
    print(f"================================")
    
    # 创建环境创建器函数
    def env_creator():
        try:
            import gymnasium
            env = gymnasium.make(args.env_name)
        except (ImportError, ModuleNotFoundError):
            import gym
            env = gym.make(args.env_name)
        return env
    
    # 根据模式选择不同的PPO实现
    if args.async_mode:
        # 使用异步PPO
        agent = AsyncPPO(
            env_creator=env_creator,
            num_workers=args.num_workers,
            num_model_replicas=args.num_model_replicas,
            steps_per_worker=args.steps_per_worker,
            epochs=args.epochs,
            mini_batch_size=args.mini_batch_size,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_param=args.clip_param,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            log_dir=log_dir
        )
    else:
        # 使用原始PPO
        agent = ParallelPPO(
            env_name=args.env_name,
            num_envs=args.num_workers,
            num_steps=args.steps_per_worker,
            epochs=args.epochs,
            mini_batch_size=args.mini_batch_size,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_param=args.clip_param,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            log_dir=log_dir
        )
    
    # 训练进度回调函数，用于动态调整参数
    def progress_callback(current_step, total_steps, info_dict):
        # 控制参数更新频率，避免过于频繁的调整
        if current_step % (10 * args.steps_per_worker) != 0:
            return info_dict
            
        # 计算训练进度
        progress = current_step / total_steps
        
        # 获取最近的奖励
        try:
            recent_rewards = agent.logger.episode_rewards if hasattr(agent.logger, 'episode_rewards') else []
            # 如果episode_rewards不存在或为空，尝试从info_dict获取
            if not recent_rewards and 'avg_reward' in info_dict:
                recent_rewards = [info_dict['avg_reward']]
        except (AttributeError, IndexError):
            # 如果发生任何错误，使用空列表
            recent_rewards = []
        
        # 获取平均奖励
        try:
            mean_reward = agent.logger.get_avg_reward()
        except (AttributeError, ValueError):
            # 如果获取平均奖励失败，使用info_dict中的值或默认值
            mean_reward = info_dict.get('avg_reward', 0.0)
        
        # 动态调整参数
        new_params = param_adjuster.adjust_params(mean_reward, recent_rewards, progress)
        
        # 安全地更新优化器学习率
        if hasattr(agent, 'optimizer'):
            try:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_params['lr']
            except (AttributeError, IndexError) as e:
                print(f"更新学习率失败: {e}")
        
        # 更新熵系数和梯度裁剪参数
        try:
            agent.entropy_coef = new_params['entropy_coef']
            agent.max_grad_norm = new_params['max_grad_norm']
            
            # 对于分布式模式，确保参数同步
            if hasattr(agent, 'sync_parameters'):
                agent.sync_parameters()
        except Exception as e:
            print(f"更新神经网络参数失败: {e}")
        
        # 每50次迭代记录一次参数调整情况
        if current_step % (50 * args.steps_per_worker * args.num_workers) == 0:
            adjustment_summary = param_adjuster.get_adjustment_summary()
            try:
                agent.logger.log_to_file(adjustment_summary)
            except (AttributeError, IOError):
                # 如果日志记录失败，只打印到控制台
                pass
            print(adjustment_summary)
        
        # 将当前参数添加到日志
        info_dict.update({
            'current_lr': new_params['lr'],
            'current_entropy_coef': new_params['entropy_coef'],
            'current_max_grad_norm': new_params['max_grad_norm']
        })
        
        return info_dict
    
    # 训练
    start_time = time.time()
    
    # 如果启用自动参数调整，传入回调函数
    if args.auto_params:
        agent.train(
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            progress_callback=progress_callback
        )
    else:
        agent.train(
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            save_interval=args.save_interval
        )
    
    # 输出总训练时间
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"=========== 训练完成 ===========")
    print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"日志和模型保存在: {log_dir}")
    print(f"================================")
    
    # 记录训练完成信息到日志
    with open(os.path.join(log_dir, 'experiment_info.txt'), 'a', encoding='utf-8') as f:
        f.write(f"=========== 训练完成 ===========\n")
        f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒\n")
        if args.auto_params:
            f.write(f"参数调整历史:\n{param_adjuster.get_adjustment_summary()}\n")
        f.write(f"================================\n")
    
if __name__ == "__main__":
    main() 