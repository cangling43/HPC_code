import os
import time
import numpy as np
import gymnasium
try:
    from gymnasium.wrappers import TimeLimit
except ImportError:
    # 如果无法导入，创建一个简单的TimeLimit包装器
    class TimeLimit:
        """简单的TimeLimit包装器"""
        def __init__(self, env, max_episode_steps=1000):
            self.env = env
            self.max_episode_steps = max_episode_steps
            self.observation_space = env.observation_space if hasattr(env, 'observation_space') else None
            self.action_space = env.action_space if hasattr(env, 'action_space') else None
            self._elapsed_steps = 0
            
        def reset(self, **kwargs):
            self._elapsed_steps = 0
            return self.env.reset(**kwargs)
            
        def step(self, action):
            observation, reward, done, info = self.env.step(action)
            self._elapsed_steps += 1
            if self._elapsed_steps >= self.max_episode_steps:
                done = True
                info['TimeLimit.truncated'] = True
            return observation, reward, done, info
            
        def close(self):
            return self.env.close()
            
        def __getattr__(self, attr):
            return getattr(self.env, attr)

try:
    import gym
except:
    pass
import ray
from ray.rllib.env.env_context import EnvContext
from typing import Callable, Dict, List, Optional, Union, Any, Tuple

# 导入torch以供RayVecEnv使用
try:
    import torch
except ImportError:
    print("警告: 无法导入torch，如果使用RayVecEnv的step方法可能会出现问题")

class VecEnv:
    """
    向量化环境包装器，用于并行运行多个环境实例
    """
    def __init__(self, env_name, num_envs=1, seed=0):
        """
        初始化向量化环境
        
        参数:
            env_name: 环境名称（gym环境ID）
            num_envs: 并行环境的数量
            seed: 随机种子
        """
        self.envs = []
        self.seeds = []
        
        for i in range(num_envs):
            try:
                # 首先尝试使用gymnasium
                import gymnasium
                env = gymnasium.make(env_name)
            except (ImportError, ModuleNotFoundError):
                # 如果gymnasium不可用，则使用gym
                env = gym.make(env_name)
                
            env = TimeLimit(env, max_episode_steps=1000)  # 限制最大步数
            
            # 存储种子
            env_seed = seed + i
            self.seeds.append(env_seed)
            
            # 尝试设置种子（旧版gym方式）
            try:
                env.seed(env_seed)
            except (AttributeError, TypeError):
                # 新版gymnasium不再使用seed方法
                pass
                
            self.envs.append(env)
            
        self.num_envs = num_envs
        
        # 获取环境的动作空间和观察空间维度
        # 判断动作空间类型
        if hasattr(self.envs[0].action_space, 'shape'):
            # 连续动作空间
            self.action_dim = self.envs[0].action_space.shape[0]
            self.is_discrete = False
        else:
            # 离散动作空间
            self.action_dim = self.envs[0].action_space.n
            self.is_discrete = True
            
        self.state_dim = self.envs[0].observation_space.shape[0]
        
        # 存储环境状态
        self.states = None
        self.dones = np.zeros(self.num_envs, dtype=bool)
        
    def reset(self):
        """
        重置所有环境并返回初始状态
        
        返回:
            states: numpy数组，形状为[num_envs, state_dim]
        """
        states = []
        for i, env in enumerate(self.envs):
            try:
                # 新版gymnasium的reset方法接受seed参数
                state = env.reset(seed=self.seeds[i])
                # 新版gymnasium可能返回(state, info)元组
                if isinstance(state, tuple):
                    state = state[0]
            except (TypeError, ValueError):
                # 旧版gym不接受seed参数
                state = env.reset()
                
            states.append(state)
        
        self.states = np.stack(states)
        self.dones = np.zeros(self.num_envs, dtype=bool)
        
        return self.states
    
    def step(self, actions):
        """
        在所有环境中执行动作
        
        参数:
            actions: numpy数组，形状为[num_envs, action_dim]
            
        返回:
            next_states: 下一个状态
            rewards: 奖励
            dones: 终止标志
            infos: 额外信息
        """
        next_states, rewards, dones, infos = [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # 处理动作格式
            if self.is_discrete:
                # 离散动作空间 - 使用整数
                action_to_env = int(action) if np.isscalar(action) else int(action.item())
            else:
                # 连续动作空间 - 使用数组
                action_to_env = action
                
            result = env.step(action_to_env)
            
            # 处理新版gymnasium返回的五元组
            if len(result) == 5:
                state, reward, terminated, truncated, info = result
                # 合并terminated和truncated为done
                done = terminated or truncated
            else:
                # 旧版gym返回四元组
                state, reward, done, info = result
            
            # 如果环境完成则重置
            if done:
                try:
                    # 新版gymnasium的reset方法接受seed参数
                    reset_result = env.reset(seed=self.seeds[i])
                    # 新版gymnasium可能返回(state, info)元组
                    if isinstance(reset_result, tuple) and len(reset_result) == 2:
                        state = reset_result[0]
                    else:
                        state = reset_result
                except (TypeError, ValueError):
                    # 旧版gym不接受seed参数
                    state = env.reset()
                
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            
        self.states = np.stack(next_states)
        self.dones = np.array(dones, dtype=bool)
        
        return np.array(next_states), np.array(rewards), np.array(dones), infos
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()
            
            
class RayVecEnv:
    """
    基于Ray的向量化环境包装器，用于分布式并行环境采样
    """
    def __init__(self, env_creator, num_envs=1, seed=0):
        """
        初始化Ray向量化环境
        
        参数:
            env_creator: 函数，用于创建环境
            num_envs: 并行环境的数量
            seed: 随机种子
        """
        from ray.rllib.env.env_context import EnvContext
        
        # 确保Ray已初始化
        if not ray.is_initialized():
            ray.init()
                    
        @ray.remote
        class RemoteEnv:
            """远程环境的Ray Actor"""
            def __init__(self, env_creator, env_config, seed):
                """
                初始化远程环境
                
                参数:
                    env_creator: 环境创建函数
                    env_config: 环境配置
                    seed: 随机种子
                """
                self.seed = seed
                
                # 对于环境，使用标准方式创建
                if isinstance(env_config, dict):
                    self.env_config = env_config.copy()
                    self.env_config['env_id'] = self.env_config.get('env_id', 0)
                    self.env_config['worker_index'] = self.env_config.get('worker_index', 0)
                else:
                    self.env_config = EnvContext({'env_id': 0}, worker_index=0)
                    
                # 创建环境
                self.env = env_creator(self.env_config)
                
                # 检查环境是否需要包装
                if hasattr(self.env, 'observation_space') and hasattr(self.env, 'action_space'):
                    print(f"环境创建成功: {getattr(self.env, 'name', '未命名环境')}")
                else:
                    # 环境可能需要包装
                    print("环境没有标准空间属性，应用包装")
                    class SimpleWrapper:
                        """简单环境包装器，确保环境具有标准属性"""
                        def __init__(self, env):
                            self.env = env
                            self.observation_space = getattr(env, 'observation_space', None)
                            self.action_space = getattr(env, 'action_space', None)
                            
                        def reset(self, seed=None):
                            """重置环境"""
                            if hasattr(self.env, 'reset'):
                                # 处理不同版本reset API
                                try:
                                    if seed is not None:
                                        return self.env.reset(seed=seed)
                                    else:
                                        return self.env.reset()
                                except TypeError:
                                    return self.env.reset()
                            
                        def step(self, action):
                            """执行动作"""
                            return self.env.step(action)
                            
                        def close(self):
                            """关闭环境"""
                            if hasattr(self.env, 'close'):
                                return self.env.close()
                                
                        def render(self):
                            """渲染环境"""
                            if hasattr(self.env, 'render'):
                                return self.env.render()
                                
                        def seed(self, seed=None):
                            """设置随机种子"""
                            if hasattr(self.env, 'seed'):
                                return self.env.seed(seed)
                                
                    # 应用包装
                    self.env = SimpleWrapper(self.env)
                
                # 检查环境类型并设置空间
                self.is_discrete = hasattr(self.env.action_space, 'n')
                
                # 尝试设置随机种子
                try:
                    if hasattr(self.env, 'seed'):
                        self.env.seed(seed)
                except (AttributeError, TypeError):
                    print(f"无法为环境设置种子 {seed}")
                    
                # 存储当前环境状态
                self.state = None
                self.ep_reward = 0
                self.ep_length = 0
                
                # 重置环境获取初始状态
                try:
                    # 新版gymnasium API返回(state, info)
                    reset_result = self.env.reset(seed=seed)
                    if isinstance(reset_result, tuple):
                        self.state = reset_result[0]  # 只取state部分
                    else:
                        self.state = reset_result
                except (TypeError, ValueError):
                    # 旧版gym API
                    self.state = self.env.reset()
                except Exception as e:
                    print(f"环境reset失败: {e}")
                    raise
                
            def reset(self):
                try:
                    # 新版gymnasium的reset方法接受seed参数
                    result = self.env.reset(seed=self.seed)
                    # 新版gymnasium返回(state, info)元组
                    if isinstance(result, tuple) and len(result) == 2:
                        state = result[0]
                    else:
                        state = result
                    # 确保返回的是float32类型的numpy数组
                    return np.array(state, dtype=np.float32)
                except TypeError:
                    # 旧版gym不接受seed参数
                    state = self.env.reset()
                    return np.array(state, dtype=np.float32)
                
            def step(self, action):
                # 确保action是正确类型
                # 检查环境的动作空间类型
                is_discrete = hasattr(self.env.action_space, 'n')
                
                if is_discrete:
                    # 离散动作空间 - 需要整数
                    if isinstance(action, np.ndarray):
                        action = int(action.item()) if action.size == 1 else int(action[0])
                    else:
                        action = int(action)
                else:
                    # 连续动作空间 - 需要正确形状的数组
                    if isinstance(action, np.ndarray):
                        action = action.astype(np.float32)
                    
                    # 确保动作形状正确 - 对于Pendulum-v1，动作需要是形状为(1,)的数组
                    if np.isscalar(action) or (isinstance(action, np.ndarray) and action.size == 1 and action.ndim == 1):
                        # 已经是正确形状
                        pass
                    elif isinstance(action, np.ndarray) and action.size == 1:
                        # 将单元素数组转为形状为(1,)的数组
                        action = np.array([float(action.flatten()[0])], dtype=np.float32)
                    elif not isinstance(action, np.ndarray):
                        # 将非数组转为数组
                        action = np.array([float(action)], dtype=np.float32)
                
                result = self.env.step(action)
                # 新版gymnasium返回(state, reward, terminated, truncated, info)五元组
                if len(result) == 5:
                    state, reward, terminated, truncated, info = result
                    # 合并terminated和truncated为done
                    done = terminated or truncated
                    # 返回与旧版gym兼容的四元组
                    return np.array(state, dtype=np.float32), reward, done, info
                else:
                    # 旧版gym返回(state, reward, done, info)四元组
                    state, reward, done, info = result
                    return np.array(state, dtype=np.float32), reward, done, info
                
            def get_spaces(self):
                """获取动作空间和观察空间"""
                return self.env.action_space, self.env.observation_space
                
            def close(self):
                self.env.close()
                
        # 创建远程环境，确保传递env_creator函数
        print(f"创建{num_envs}个远程环境实例")
        self.envs = []
        for i in range(num_envs):
            env_config = {
                "env_id": i
            }
            self.envs.append(RemoteEnv.remote(env_creator, env_config, seed + i))
            
        self.num_envs = num_envs
        
        # 获取空间信息
        action_space, observation_space = ray.get(self.envs[0].get_spaces.remote())
        
        # 打印动作空间信息，帮助调试
        print(f"Action space type: {type(action_space)}")
        print(f"Action space: {action_space}")
        
        # 根据动作空间类型设置action_dim
        if hasattr(action_space, 'n'):  # 离散动作空间(Discrete)
            self.action_dim = action_space.n
            self.is_discrete = True
            print(f"检测到离散动作空间，动作维度: {self.action_dim}")
        elif hasattr(action_space, 'shape') and action_space.shape:  # 连续动作空间(Box)
            self.action_dim = action_space.shape[0]
            self.is_discrete = False
            print(f"检测到连续动作空间，动作维度: {self.action_dim}")
        else:
            # 无法确定动作空间类型，设置默认值
            self.action_dim = 1
            self.is_discrete = False
            print(f"无法确定动作空间类型，使用默认值: {self.action_dim}")
            
        # 设置状态维度
        self.state_dim = observation_space.shape[0]
        
    def reset(self):
        """
        以分布式方式重置所有环境并返回初始状态
        
        返回:
            states: numpy数组，形状为[num_envs, state_dim]
        """
        # 异步发送重置命令
        reset_futures = [env.reset.remote() for env in self.envs]
        
        # 等待所有环境重置完成并获取结果
        states = ray.get(reset_futures)
        
        # 确保返回的是正确类型的numpy数组
        return np.array(states, dtype=np.float32)
    
    def step(self, actions):
        """
        以分布式方式在所有环境中执行动作
        
        参数:
            actions: numpy数组，形状为[num_envs, action_dim]
            
        返回:
            next_states: 下一个状态
            rewards: 奖励
            dones: 终止标志
            infos: 额外信息
        """
        # 确保actions是numpy数组且数据类型正确
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
            
        # 根据动作空间类型处理动作
        if self.is_discrete:
            # 离散动作空间 - 确保是整数
            # 如果actions是多维数组，转换为一维
            if isinstance(actions, np.ndarray) and actions.ndim > 1:
                actions = actions.flatten()
            actions = actions.astype(np.int32)
        else:
            # 连续动作空间 - 确保是浮点数
            actions = actions.astype(np.float32)
        
        
        # 异步发送步进命令
        step_futures = [env.step.remote(action) for env, action in zip(self.envs, actions)]
        
        # 等待所有环境步进完成并获取结果
        results = ray.get(step_futures)
        
        # 解析结果
        next_states, rewards, dones, infos = zip(*results)
        
        # 确保所有返回的数组使用正确的数据类型
        next_states_array = np.array(next_states, dtype=np.float32)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=bool)
        
        return next_states_array, rewards_array, dones_array, infos
    
    def close(self):
        """关闭所有远程环境"""
        close_futures = [env.close.remote() for env in self.envs]
        ray.get(close_futures)


def create_env_creator(env_name):
    """
    创建环境创建器函数
    
    参数:
        env_name: 环境名称（gym环境ID）
        
    返回:
        env_creator: 环境创建器函数
    """
    # 使用闭包捕获传入的env_name
    env_name_closure = env_name
    
    def _env_creator(env_config=None, env_name=None):
        """
        创建环境的函数
        
        参数:
            env_config: 环境配置
            env_name: 环境名称，如果提供则覆盖外部传入的环境名称
        
        返回:
            env: 创建的环境
        """
        if env_config is None:
            env_config = {}
        
        # 优先使用直接传入的env_name参数
        actual_env_name = env_name
        
        # 如果直接传入的env_name为None，则尝试使用env_config中的env_name
        if actual_env_name is None and 'env_name' in env_config:
            actual_env_name = env_config['env_name']
        
        # 如果仍然为None，则使用外部传入的env_name
        if actual_env_name is None:
            actual_env_name = env_name_closure
        
        # 确保env_name不为None
        assert actual_env_name is not None, "环境名称不能为None"
        
        # 使用标准方法创建环境
        try:
            # 优先尝试使用gymnasium
            print(f"尝试使用gymnasium创建环境: {actual_env_name}")
            import gymnasium
            env = gymnasium.make(actual_env_name)
            print(f"成功使用gymnasium创建环境: {actual_env_name}")
        except (ImportError, ModuleNotFoundError, ValueError) as e:
            try:
                # 如果gymnasium失败，尝试使用gym
                print(f"gymnasium创建失败: {e}")
                print(f"尝试使用gym创建环境: {actual_env_name}")
                import gym
                env = gym.make(actual_env_name)
                print(f"成功使用gym创建环境: {actual_env_name}")
            except Exception as e2:
                raise ValueError(f"无法创建环境 {actual_env_name}: gymnasium错误: {e}, gym错误: {e2}")
        
        # 返回环境
        return env
    
    return _env_creator 