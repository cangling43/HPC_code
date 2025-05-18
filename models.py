import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ActorCritic(nn.Module):
    """
    演员评论家网络，包含策略网络（Actor）和价值网络（Critic）
    用于PPO算法的策略表示和状态价值估计
    支持连续动作空间和离散动作空间
    
    完全重构的版本，针对Pendulum环境特别优化
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, discrete=False):
        super(ActorCritic, self).__init__()
        
        # 记录动作空间类型
        self.discrete = discrete
        self.action_dim = action_dim
        
        # 简化网络架构，使用更小的隐藏层和更保守的激活函数
        
        # Actor网络 - 不再共享特征提取器，增加网络稳定性
        if self.discrete:
            # 离散动作空间
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # 连续动作空间 - 均值网络
            self.actor_mean = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()  # 限制动作范围在[-1,1]
            )
            
            # Pendulum环境动作范围是[-2,2]，所以我们输出后会乘以2
            self.action_scale = 2.0
            
            # 使用固定的标准差而非学习的标准差，增加稳定性
            # 初始值设为0.5，适合Pendulum环境
            self.action_log_std = nn.Parameter(torch.zeros(action_dim) - 0.7)
        
        # Critic网络 - 独立网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 使用正确的初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重，使用小而保守的数值"""
        if isinstance(module, nn.Linear):
            # 使用较小的标准差初始化
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
            
            # 对最后一层特殊处理
            if self.discrete:
                # 离散动作空间
                if hasattr(self, 'actor') and isinstance(self.actor, nn.Sequential) and module == self.actor[-2]:
                    # 离散动作网络倒数第二层使用较小初始化
                    nn.init.orthogonal_(module.weight, gain=0.01)
            else:
                # 连续动作空间
                if hasattr(self, 'actor_mean') and isinstance(self.actor_mean, nn.Sequential) and module == self.actor_mean[-2]:
                    # 连续动作均值层使用很小的初始化，保证初始动作接近零
                    nn.init.orthogonal_(module.weight, gain=0.001)
            
            # 价值网络最后一层使用较小初始化
            if isinstance(self.critic, nn.Sequential) and module == self.critic[-1]:
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def forward(self, state):
        """前向传播，返回动作分布参数和状态值估计"""
        if self.discrete:
            # 离散动作空间 - 输出动作概率
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            return probs, self.critic(state)
        else:
            # 连续动作空间 - 输出均值和标准差
            mean = self.actor_mean(state) * self.action_scale
            # 固定标准差，增加稳定性
            log_std = self.action_log_std.expand_as(mean) if mean.dim() > 1 else self.action_log_std
            std = torch.exp(log_std)
            return mean, std, self.critic(state)
            
    def get_action(self, state, deterministic=False):
        """采样动作"""
        with torch.no_grad():
            if self.discrete:
                probs, value = self.forward(state)
                if deterministic:
                    action = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().unsqueeze(-1)
                log_prob = torch.log(probs.gather(-1, action) + 1e-8)
                return action, log_prob, value
            else:
                mean, std, value = self.forward(state)
                
                if deterministic:
                    return mean, None, value
                
                # 创建正态分布并采样
                try:
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                    
                    # 计算对数概率
                    log_prob = normal.log_prob(action).sum(-1, keepdim=True)
                    
                    # 裁剪动作到有效范围
                    action = torch.clamp(action, -self.action_scale, self.action_scale)
                    
                    return action, log_prob, value
                except:
                    # 发生错误时使用均值作为动作
                    print("警告：采样动作时出错，使用均值")
                    return mean, torch.zeros_like(mean[:, 0:1]), value
    
    def evaluate_actions(self, state, action):
        """评估动作的对数概率和熵"""
        if self.discrete:
            probs, value = self.forward(state)
            
            # 处理action形状
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)  # 从[batch, 1]变为[batch]
            
            # 确保action是一维张量，与gather操作兼容
            if action.dim() == 0:
                action = action.unsqueeze(0)  # 单个标量值变为一维张量
            
            # 确保action和probs维度匹配
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)  # 单样本情况，增加batch维度
            
            # 确保action是LongTensor
            action = action.long()
            
            # 打印调试信息，帮助诊断
            try:
                # 安全地执行gather操作
                gathered_probs = probs.gather(-1, action.unsqueeze(-1))
                log_prob = torch.log(gathered_probs + 1e-8)
                
                # 计算熵
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
                
                return log_prob, entropy, value
            except Exception as e:
                print(f"离散动作空间评估出错: {e}")
                print(f"probs shape: {probs.shape}, action shape: {action.shape}")
                print(f"value shape: {value.shape}")
                
                # 提供回退方案
                batch_size = state.shape[0] if state.dim() > 1 else 1
                return torch.zeros(batch_size, 1).to(state.device), torch.zeros(batch_size, 1).to(state.device), value
        else:
            mean, std, value = self.forward(state)
            
            # 创建正态分布
            try:
                normal = torch.distributions.Normal(mean, std)
                
                # 计算对数概率
                log_prob = normal.log_prob(action).sum(-1, keepdim=True)
                
                # 计算熵
                entropy = normal.entropy().sum(-1, keepdim=True)
                
                return log_prob, entropy, value
            except Exception as e:
                print(f"评估连续动作时出错: {e}")
                # 手动计算对数概率和熵
                var = std.pow(2) + 1e-8
                log_prob = -0.5 * ((action - mean).pow(2) / var + torch.log(2 * math.pi * var)).sum(-1, keepdim=True)
                entropy = 0.5 * (1.0 + torch.log(2 * math.pi * var)).sum(-1, keepdim=True)
                
                return log_prob, entropy, value
                
    def get_value(self, state):
        """获取状态值估计"""
        with torch.no_grad():
            if self.discrete:
                _, value = self.forward(state)
                return value
            else:
                _, _, value = self.forward(state)
                return value 