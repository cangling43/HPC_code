import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
import argparse
import matplotlib
import glob
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体支持
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
except:
    print("警告: 可能无法正确显示中文，请安装相应字体")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Pendulum-v1训练结果可视化")
    
    # 添加文件路径相关参数
    parser.add_argument("--csv_file", type=str, default=None,
                        help="具体CSV文件路径")
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="CSV文件所在的目录，程序会自动查找目录下的progress.csv文件")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="图表保存的目录，如果不指定，则保存在CSV文件同级目录下的figures子文件夹中")
    
    # 其他参数
    parser.add_argument("--smoothing", type=int, default=None,
                        help="平滑窗口大小，若不指定则自动选择")
    parser.add_argument("--auto_smoothing", action="store_true",
                        help="是否自动选择平滑窗口大小")
    parser.add_argument("--show", action="store_true",
                        help="是否显示图像")
    
    return parser.parse_args()

def auto_select_window_size(data_length):
    """自动选择合适的平滑窗口大小
    
    根据数据长度自动计算适当的平滑窗口大小
    对于短序列使用小窗口，长序列使用稍大窗口，但总体保持窗口大小适中以避免过度平滑
    
    参数:
        data_length (int): 数据序列的长度
        
    返回:
        int: 推荐的平滑窗口大小
    """
    if data_length < 50:
        # 数据点很少时，使用较小的窗口或不平滑
        return max(3, data_length // 10)
    elif data_length < 200:
        # 中等规模数据，窗口为数据长度的5%左右
        return max(5, data_length // 20)
    elif data_length < 1000:
        # 较大规模数据，窗口适中
        return max(10, data_length // 50)
    else:
        # 大规模数据，窗口相对较大但避免过度平滑
        return min(50, max(20, data_length // 100))

def smooth(data, window_size):
    """对数据进行平滑处理"""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='valid')

def format_scientific(x, pos):
    """格式化科学计数法"""
    if x == 0:
        return '0'
    exp = int(np.log10(abs(x)))
    if abs(exp) > 3:
        return f'{x/10**exp:.1f}e{exp}'
    else:
        return f'{x:.1f}'

def plot_rewards(df, save_dir, smoothing_window=None, show=False):
    """绘制奖励相关图表"""
    plt.figure(figsize=(16, 8))
    
    # 如果没有指定平滑窗口大小，则自动计算
    if smoothing_window is None:
        smoothing_window = auto_select_window_size(len(df))
        print(f"自动选择平滑窗口大小: {smoothing_window}")
    
    # 绘制原始奖励
    plt.subplot(1, 2, 1)
    x = df['Step']
    y = df['avg_reward']
    plt.plot(x, y, 'b-', alpha=0.2, label='单回合奖励')
    
    # 计算平滑后的奖励
    if len(y) > smoothing_window:
        y_smooth = smooth(y, smoothing_window)
        x_smooth = x[smoothing_window-1:][:len(y_smooth)]
        plt.plot(x_smooth, y_smooth, 'b-', label=f'平滑奖励 (窗口={smoothing_window})')
    
    # 平均奖励
    plt.plot(x, df['avg_reward'], 'r-', alpha=0.5, label='平均奖励')
    plt.plot(x, df['avg_reward_100'], 'g-', linewidth=2, label='最近100回合平均奖励')
    
    # 添加目标线和区域
    plt.axhline(y=-200, color='r', linestyle='--', alpha=0.5, label='优秀表现阈值 (-200)')
    plt.axhspan(-300, -150, alpha=0.1, color='g', label='目标区域')
    
    plt.title('奖励变化趋势')
    plt.xlabel('训练步数')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_scientific))
    
    # 绘制回合长度
    plt.subplot(1, 2, 2)
    
    # 检查episode_length列是否存在
    if 'episode_length' in df.columns:
        plt.plot(x, df['episode_length'], 'g-', alpha=0.3)
        
        if len(df['episode_length']) > smoothing_window:
            length_smooth = smooth(df['episode_length'], smoothing_window)
            x_smooth = x[smoothing_window-1:][:len(length_smooth)]
            plt.plot(x_smooth, length_smooth, 'g-', linewidth=2)
        
        plt.title('回合长度变化')
        plt.xlabel('训练步数')
        plt.ylabel('回合步数')
    else:
        # 如果没有episode_length列，则显示steps_per_sec
        if 'steps_per_sec' in df.columns:
            plt.plot(x, df['steps_per_sec'], 'g-', alpha=0.3)
            
            if len(df['steps_per_sec']) > smoothing_window:
                speed_smooth = smooth(df['steps_per_sec'], smoothing_window)
                x_smooth = x[smoothing_window-1:][:len(speed_smooth)]
                plt.plot(x_smooth, speed_smooth, 'g-', linewidth=2)
            
            plt.title('训练速度变化')
            plt.xlabel('训练步数')
            plt.ylabel('步数/秒')
        else:
            plt.text(0.5, 0.5, '无回合长度或速度数据', 
                     horizontalalignment='center', verticalalignment='center')
            plt.title('缺少数据')
    
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_scientific))
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'rewards.png'), dpi=300, bbox_inches='tight')
        print(f"已保存奖励图像到 {os.path.join(save_dir, 'rewards.png')}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_losses(df, save_dir, smoothing_window=None, show=False):
    """绘制损失函数和熵图表"""
    plt.figure(figsize=(16, 8))
    
    # 如果没有指定平滑窗口大小，则自动计算
    if smoothing_window is None:
        smoothing_window = auto_select_window_size(len(df))
    
    # 绘制策略损失
    plt.subplot(1, 3, 1)
    x = df['Step']
    y = df['policy_loss']
    plt.plot(x, y, 'b-', alpha=0.3)
    
    if len(y) > smoothing_window:
        y_smooth = smooth(y, smoothing_window)
        x_smooth = x[smoothing_window-1:][:len(y_smooth)]
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2)
    
    plt.title('策略损失 (Policy Loss)')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_scientific))
    
    # 绘制价值损失
    plt.subplot(1, 3, 2)
    y = df['value_loss']
    plt.plot(x, y, 'r-', alpha=0.3)
    
    if len(y) > smoothing_window:
        y_smooth = smooth(y, smoothing_window)
        x_smooth = x[smoothing_window-1:][:len(y_smooth)]
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2)
    
    plt.title('价值损失 (Value Loss)')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_scientific))
    
    # 绘制熵
    plt.subplot(1, 3, 3)
    y = df['entropy']
    plt.plot(x, y, 'g-', alpha=0.3)
    
    if len(y) > smoothing_window:
        y_smooth = smooth(y, smoothing_window)
        x_smooth = x[smoothing_window-1:][:len(y_smooth)]
        plt.plot(x_smooth, y_smooth, 'g-', linewidth=2)
    
    plt.title('策略熵 (Entropy)')
    plt.xlabel('训练步数')
    plt.ylabel('熵值')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_scientific))
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'losses.png'), dpi=300, bbox_inches='tight')
        print(f"已保存损失图像到 {os.path.join(save_dir, 'losses.png')}")
    
    if show:
        plt.show()
    else:
        plt.close()

def create_summary_dashboard(df, save_dir, smoothing_window=None, show=False):
    """创建综合仪表盘，汇总关键指标"""
    plt.figure(figsize=(16, 12))
    
    # 如果没有指定平滑窗口大小，则自动计算
    if smoothing_window is None:
        smoothing_window = auto_select_window_size(len(df))
    
    # 设置网格
    gs = GridSpec(3, 3, figure=plt.gcf())
    
    # 确保必要的列存在
    reward_col = 'avg_reward_100' if 'avg_reward_100' in df.columns else 'avg_reward'
    x = df['Step']
    
    # 1. 左上：最近100回合平均奖励曲线
    ax1 = plt.subplot(gs[0, 0:2])
    ax1.plot(x, df[reward_col], 'g-', linewidth=2)
    ax1.axhline(y=-200, color='r', linestyle='--', alpha=0.5)
    ax1.set_title('最近100回合平均奖励' if reward_col == 'avg_reward_100' else '平均奖励')
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('奖励')
    ax1.grid(True, alpha=0.3)
    
    # 2. 右上：最终性能指标
    ax2 = plt.subplot(gs[0, 2])
    # 计算关键指标
    final_reward = df[reward_col].iloc[-1]
    best_reward = df[reward_col].max()
    
    # 显示为文本
    ax2.axis('off')
    ax2.text(0.5, 0.9, "关键性能指标", ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.75, f"最终平均奖励: {final_reward:.2f}", ha='center', va='center')
    ax2.text(0.5, 0.65, f"最佳平均奖励: {best_reward:.2f}", ha='center', va='center')
    ax2.text(0.5, 0.55, f"训练总步数: {df['Step'].iloc[-1]:,}", ha='center', va='center')
    
    # 检查'steps_per_second'或'steps_per_sec'列是否存在
    if 'steps_per_second' in df.columns:
        ax2.text(0.5, 0.45, f"平均FPS: {df['steps_per_second'].mean():.2f}", ha='center', va='center')
    elif 'steps_per_sec' in df.columns:
        ax2.text(0.5, 0.45, f"平均FPS: {df['steps_per_sec'].mean():.2f}", ha='center', va='center')
    else:
        ax2.text(0.5, 0.45, f"平均FPS: 数据不可用", ha='center', va='center')
    
    # 添加评级
    if final_reward >= -200:
        rating = "优秀" 
        color = "green"
    elif final_reward >= -300:
        rating = "良好"
        color = "blue"
    elif final_reward >= -400:
        rating = "一般"
        color = "orange"
    else:
        rating = "需改进"
        color = "red"
        
    ax2.text(0.5, 0.3, f"整体评级: {rating}", ha='center', va='center', fontsize=16, 
             fontweight='bold', color=color)
    
    # 3. 中央：损失函数变化
    ax3 = plt.subplot(gs[1, 0:2])
    # 检查必要的列是否存在
    if 'policy_loss' in df.columns and 'value_loss' in df.columns:
        # 平滑处理
        window = smoothing_window
        if len(df) > window:
            policy_smooth = smooth(df['policy_loss'], window)
            x_smooth = x[window-1:][:len(policy_smooth)]
            ax3.plot(x_smooth, policy_smooth, 'b-', linewidth=2, label='策略损失')
            
            value_smooth = smooth(df['value_loss'], window)
            x_smooth = x[window-1:][:len(value_smooth)]
            ax3.plot(x_smooth, value_smooth, 'r-', linewidth=2, label='价值损失')
        else:
            ax3.plot(x, df['policy_loss'], 'b-', linewidth=2, label='策略损失')
            ax3.plot(x, df['value_loss'], 'r-', linewidth=2, label='价值损失')
            
        ax3.set_title('损失函数变化')
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('损失值')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, '损失函数数据不可用', ha='center', va='center')
        ax3.set_title('损失函数变化')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. 右中：热力图显示训练各阶段奖励分布
    ax4 = plt.subplot(gs[1, 2])
    # 将训练分为10个阶段
    stages = 10
    stage_size = len(df) // stages
    
    # 使用可用的奖励列
    reward_column = 'avg_reward' if 'avg_reward' in df.columns else reward_col
    
    # 准备热图数据
    heatmap_data = []
    for i in range(stages):
        start_idx = i * stage_size
        end_idx = (i+1) * stage_size if i < stages-1 else len(df)
        stage_rewards = df[reward_column].iloc[start_idx:end_idx]
        
        # 计算这个阶段的分位数
        q1 = np.percentile(stage_rewards, 25)
        q2 = np.percentile(stage_rewards, 50)
        q3 = np.percentile(stage_rewards, 75)
        
        heatmap_data.append([q1, q2, q3])
    
    # 绘制热图
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", 
                xticklabels=['25%', '50%', '75%'],
                yticklabels=list(range(1, stages+1)),
                ax=ax4)
    ax4.set_title('各训练阶段奖励分布')
    ax4.set_ylabel('训练阶段')
    
    # 5. 左下和中下：奖励分布变化
    ax5 = plt.subplot(gs[2, 0:2])
    # 将数据分成5个区间，查看分布变化
    segments = 5
    segment_size = len(df) // segments
    data = []
    labels = []
    
    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i+1) * segment_size if i < segments-1 else len(df)
        segment_data = df[reward_column].iloc[start_idx:end_idx]
        data.append(segment_data)
        labels.append(f"阶段{i+1}")
    
    # 使用violinplot显示奖励分布随时间的变化
    ax5.violinplot(data, showmedians=True)
    ax5.set_title('奖励分布随训练变化')
    ax5.set_xticks(range(1, segments+1))
    ax5.set_xticklabels(labels)
    ax5.set_ylabel('奖励')
    ax5.grid(True, alpha=0.3)
    
    # 6. 右下：熵变化
    ax6 = plt.subplot(gs[2, 2])
    if 'entropy' in df.columns:
        window = smoothing_window
        if len(df) > window:
            entropy_smooth = smooth(df['entropy'], window)
            x_smooth = x[window-1:][:len(entropy_smooth)]
            ax6.plot(x_smooth, entropy_smooth, 'r-', linewidth=2)
        else:
            ax6.plot(x, df['entropy'], 'r-', linewidth=2)
        
        ax6.set_title('策略熵变化')
        ax6.set_xlabel('训练步数')
        ax6.set_ylabel('熵值')
    else:
        ax6.text(0.5, 0.5, '熵数据不可用', ha='center', va='center')
        ax6.set_title('策略熵变化')
    
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Pendulum-v1 训练总结仪表盘', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为顶部标题留出空间
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'summary_dashboard.png'), dpi=300, bbox_inches='tight')
        print(f"已保存总结仪表盘到 {os.path.join(save_dir, 'summary_dashboard.png')}")
    
    if show:
        plt.show()
    else:
        plt.close()

def analyze_training(df, save_dir=None):
    """分析训练结果，给出评估，并可选择性地保存到training_log.txt文件"""
    # 确定使用哪个奖励列
    reward_col = 'avg_reward_100' if 'avg_reward_100' in df.columns else 'avg_reward'
    
    # 获取最终和最大的平均奖励
    final_avg_reward = df[reward_col].iloc[-1]
    max_avg_reward = df[reward_col].max()
    
    # 计算奖励改进
    initial_reward_idx = min(100, len(df)-1)  # 取前100个或更少
    initial_avg_reward = df[reward_col].iloc[initial_reward_idx]  # 取前100个或更少
    improvement = final_avg_reward - initial_avg_reward
    
    # 分析损失稳定性
    policy_loss_last_10pct = float('nan')
    value_loss_last_10pct = float('nan')
    
    if 'policy_loss' in df.columns:
        policy_loss_last_10pct = df['policy_loss'].iloc[int(0.9*len(df)):].std()
    
    if 'value_loss' in df.columns:
        value_loss_last_10pct = df['value_loss'].iloc[int(0.9*len(df)):].std()
    
    # 性能分析
    avg_steps_per_second = float('nan')
    if 'steps_per_second' in df.columns:
        avg_steps_per_second = df['steps_per_second'].mean()
    elif 'steps_per_sec' in df.columns:
        avg_steps_per_second = df['steps_per_sec'].mean()
    
    # 构建分析结果文本
    analysis_text = "\n========== 训练结果分析 ==========\n"
    analysis_text += f"训练总步数: {df['Step'].iloc[-1]:,}\n"
    analysis_text += f"最终平均奖励 (最近100回合): {final_avg_reward:.2f}\n"
    analysis_text += f"最高平均奖励 (最近100回合): {max_avg_reward:.2f}\n"
    analysis_text += f"奖励改进幅度: {improvement:.2f}\n"
    
    # 检查avg_steps_per_second是否为NaN
    if pd.isna(avg_steps_per_second):
        analysis_text += "平均每秒步数: 数据不可用\n"
    else:
        analysis_text += f"平均每秒步数: {avg_steps_per_second:.2f}\n"
    
    # 训练质量评估
    analysis_text += "\n训练质量评估:\n"
    
    # 奖励评估 (Pendulum-v1的好结果应该在-200左右)
    if final_avg_reward >= -200:
        analysis_text += "✓ 奖励达到优秀水平 (>= -200)\n"
    elif final_avg_reward >= -300:
        analysis_text += "✓ 奖励达到良好水平 (>= -300)\n"
    elif final_avg_reward >= -400:
        analysis_text += "△ 奖励达到中等水平 (>= -400)\n"
    else:
        analysis_text += "✗ 奖励未达到期望水平 (< -400)\n"
    
    # 损失稳定性评估
    if not pd.isna(policy_loss_last_10pct) and not pd.isna(value_loss_last_10pct):
        if policy_loss_last_10pct < 0.1 and value_loss_last_10pct < 0.1:
            analysis_text += "✓ 损失函数非常稳定\n"
        elif policy_loss_last_10pct < 0.5 and value_loss_last_10pct < 0.5:
            analysis_text += "✓ 损失函数较为稳定\n"
        else:
            analysis_text += "△ 损失函数波动较大\n"
    else:
        analysis_text += "- 损失函数稳定性分析：数据不可用\n"
    
    # 训练是否收敛
    if len(df) >= 100:
        reward_diff = abs(df[reward_col].iloc[-1] - df[reward_col].iloc[-100]).item()
        if reward_diff < 20:
            analysis_text += "✓ 训练已收敛\n"
        else:
            analysis_text += "△ 训练可能尚未完全收敛\n"
    else:
        analysis_text += "- 训练收敛性分析：数据不足\n"
    
    # 总体评估
    has_loss_data = not pd.isna(policy_loss_last_10pct)
    
    if final_avg_reward >= -200 and (not has_loss_data or policy_loss_last_10pct < 0.5):
        analysis_text += "\n总体评估: 非常成功 ⭐⭐⭐⭐⭐\n"
    elif final_avg_reward >= -300 and (not has_loss_data or policy_loss_last_10pct < 1.0):
        analysis_text += "\n总体评估: 成功 ⭐⭐⭐⭐\n"
    elif final_avg_reward >= -400:
        analysis_text += "\n总体评估: 良好 ⭐⭐⭐\n"
    elif improvement > 500:
        analysis_text += "\n总体评估: 尚可 ⭐⭐\n"
    else:
        analysis_text += "\n总体评估: 需要改进 ⭐\n"
    
    analysis_text += "================================\n"
    
    # 打印分析结果
    print(analysis_text)
    
    # 如果提供了保存目录，将分析结果保存到training_log.txt文件
    if save_dir:
        log_file_path = os.path.join(os.path.dirname(save_dir), 'training_log.txt')
        
        # 如果文件存在，追加内容；否则创建新文件
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                # 添加时间戳
                import datetime
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n\n可视化分析时间: {current_time}\n")
                f.write(analysis_text)
            print(f"训练分析结果已保存到 {log_file_path}")
        except Exception as e:
            print(f"保存分析结果到文件时出错: {e}")
    
    return analysis_text

def find_csv_file(csv_dir):
    """在指定目录中查找progress.csv文件"""
    if not os.path.exists(csv_dir):
        print(f"错误: 指定的目录 '{csv_dir}' 不存在!")
        return None
        
    # 首先查找目录中的progress.csv文件
    csv_path = os.path.join(csv_dir, "progress.csv")
    if os.path.exists(csv_path):
        return csv_path
        
    # 如果没有直接找到，尝试在子目录中查找
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file == "progress.csv":
                csv_path = os.path.join(root, file)
                print(f"在 {root} 中找到 progress.csv 文件")
                return csv_path
                
    # 最后尝试查找任何CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)
    if csv_files:
        print(f"未找到progress.csv文件，使用第一个找到的CSV文件: {csv_files[0]}")
        return csv_files[0]
    
    print(f"在 {csv_dir} 及其子目录中未找到任何CSV文件!")
    return None

def main():
    """主函数"""
    args = parse_args()
    
    # 确定CSV文件路径
    csv_file = None
    
    if args.csv_file:
        # 如果直接指定了CSV文件
        csv_file = args.csv_file
        if not os.path.exists(csv_file):
            print(f"错误: CSV文件 '{csv_file}' 不存在!")
            return
    elif args.csv_dir:
        # 如果指定了CSV文件所在目录
        csv_file = find_csv_file(args.csv_dir)
        if not csv_file:
            print("未能找到有效的CSV文件，请检查目录路径或直接指定CSV文件路径。")
            return
    else:
        # 如果既没有指定文件也没有指定目录
        print("错误: 请使用 --csv_file 指定CSV文件路径，或使用 --csv_dir 指定CSV文件所在目录。")
        return
    
    # 确定保存目录
    if args.output_dir:
        # 如果指定了输出目录
        save_dir = args.output_dir
    else:
        # 默认在CSV文件同级目录下创建figures子文件夹
        csv_dir = os.path.dirname(csv_file)
        save_dir = os.path.join(csv_dir, 'figures')
    
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建输出目录: {save_dir}")
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取数据: {len(df)}行 x {len(df.columns)}列")
        
        # 打印列名，帮助用户了解数据结构
        print("\n数据列名:")
        for col in df.columns:
            print(f"  - {col}")
        
        # 检查必要的列是否存在
        required_cols = ['Step']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"\n警告: 以下必要的列不存在: {', '.join(missing_cols)}")
            print("可能无法正确生成所有图表")
        
        # 检查奖励列
        reward_cols = [col for col in df.columns if 'reward' in col.lower()]
        if not reward_cols:
            print("\n警告: 未找到任何与奖励相关的列(包含'reward'的列名)，可能无法生成奖励相关图表")
        else:
            print(f"\n找到的奖励相关列: {', '.join(reward_cols)}")
            
        # 使用的主要列
        reward_col = 'avg_reward_100' if 'avg_reward_100' in df.columns else 'avg_reward'
        print(f"将使用 '{reward_col}' 作为主要奖励列")
        
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return
    
    # 确定平滑窗口大小
    smoothing_window = None
    if args.smoothing is not None:
        smoothing_window = args.smoothing
    elif args.auto_smoothing or args.smoothing is None:
        smoothing_window = auto_select_window_size(len(df))
        print(f"自动选择的平滑窗口大小: {smoothing_window}")
    
    # 绘制核心图表
    print("正在生成图表...")
    plot_rewards(df, save_dir, smoothing_window, args.show)
    plot_losses(df, save_dir, smoothing_window, args.show)
    create_summary_dashboard(df, save_dir, smoothing_window, args.show)
    
    # 分析训练结果并保存到training_log.txt
    analyze_training(df, save_dir)
    
    print(f"\n所有图表已保存到: {save_dir}")
    
    # 创建README文件，解释所有图表
    readme_path = os.path.join(save_dir, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# Pendulum-v1 训练结果可视化说明\n\n")
        f.write("## 图表说明\n\n")
        f.write("1. rewards.png - 奖励变化趋势和回合长度变化\n")
        f.write("   - 左图：显示平均奖励及其变化趋势\n")
        f.write("   - 右图：显示回合长度或训练速度变化\n\n")
        f.write("2. losses.png - 损失函数和熵变化\n")
        f.write("   - 左图：策略损失(Policy Loss)变化\n")
        f.write("   - 中图：价值损失(Value Loss)变化\n")
        f.write("   - 右图：策略熵(Entropy)变化\n\n")
        f.write("3. summary_dashboard.png - 训练总结仪表盘\n")
        f.write("   - 综合显示所有关键指标\n")
        f.write("   - 包含奖励曲线、损失变化、奖励分布和熵变化\n")
        f.write("   - 自动评级和训练结果评估\n\n")
        f.write(f"平滑窗口大小: {smoothing_window}（{'自动选择' if args.smoothing is None else '手动指定'}）\n\n")
        f.write("## 数据列\n\n")
        for col in df.columns:
            f.write(f"- {col}\n")
    
    print(f"已创建图表说明文件: {readme_path}")

if __name__ == "__main__":
    main() 