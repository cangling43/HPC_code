# Pendulum-v1 训练结果可视化说明

## 图表说明

1. rewards.png - 奖励变化趋势和回合长度变化
   - 左图：显示平均奖励及其变化趋势
   - 右图：显示回合长度或训练速度变化

2. losses.png - 损失函数和熵变化
   - 左图：策略损失(Policy Loss)变化
   - 中图：价值损失(Value Loss)变化
   - 右图：策略熵(Entropy)变化

3. summary_dashboard.png - 训练总结仪表盘
   - 综合显示所有关键指标
   - 包含奖励曲线、损失变化、奖励分布和熵变化
   - 自动评级和训练结果评估

平滑窗口大小: 10（自动选择）

## 数据列

- Step
- policy_loss
- value_loss
- entropy
- avg_reward
- avg_reward_100
- steps_per_sec
