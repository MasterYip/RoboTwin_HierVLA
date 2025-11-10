import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_SR_metrics(json_file1_path, json_file2_path, max_iterations=None, 
                         save_path=None, dpi=300, markers=True):
    # 读取JSON文件
    def load_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['episodes']
    
    episodes1 = load_data(json_file1_path)
    episodes2 = load_data(json_file2_path)
    
    # 如果未指定最大迭代次数，则使用最小的数据长度
    min_length = min(len(episodes1), len(episodes2))
    if max_iterations is None or max_iterations > min_length:
        max_iterations = min_length
    
    # 准备数据
    def calculate_success_rates(episodes):
        success_counts = []
        total_success = 0
        for i, episode in enumerate(episodes[:max_iterations]):
            if episode['success']:
                total_success += 1
            success_rate = total_success / (i + 1)
            success_counts.append(success_rate)
        return success_counts
    
    def calculate_step_utilizations(episodes):
        return [episode['step_utilization'] for episode in episodes[:max_iterations]]
    
    # 成功率数据
    success_rates1 = calculate_success_rates(episodes1)
    success_rates2 = calculate_success_rates(episodes2)
    
    # step_utilization数据
    step_utils1 = calculate_step_utilizations(episodes1)
    step_utils2 = calculate_step_utilizations(episodes2)
    
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    markers = ['o', 's'] if markers else [None, None]  # 圆形和方形标记
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    line_styles = ['-', '--']
    
    # 创建画布
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6), dpi=dpi)
    
    # 绘制成功率曲线
    x = np.arange(1, max_iterations + 1)
    ax1.plot(x, success_rates1, label='SR_Hierarchical VLA', color=colors[0], 
             linestyle=line_styles[0], marker=markers[0], markevery=1, markersize=6)
    ax1.plot(x, success_rates2, label='SR_Flat VLA', color=colors[1], 
             linestyle=line_styles[1], marker=markers[1], markevery=1, markersize=6)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Success Rate', fontsize=12)
    ax1.set_title('Comparison of Success Rates', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=10)
    
    # # 绘制step_utilization曲线
    # ax2.plot(x, step_utils1, label='Experiment 1', color=colors[0], 
    #          linestyle=line_styles[0], marker=markers[0], markevery=1, markersize=6)
    # ax2.plot(x, step_utils2, label='Experiment 2', color=colors[1], 
    #          linestyle=line_styles[1], marker=markers[1], markevery=1, markersize=6)
    
    # ax2.set_xlabel('Iteration', fontsize=12)
    # ax2.set_ylabel('Step Utilization', fontsize=12)
    # ax2.set_title('Comparison of Step Utilizations', fontsize=14)
    # ax2.grid(True, alpha=0.3)
    # ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        if save_path.lower().endswith('.pdf'):
            plt.savefig(save_path, format='pdf', dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Combined charts saved to {save_path} (DPI={dpi})")
        plt.close()
    else:
        plt.show()



# 使用示例
# filename1 = "_benchmark_results_30000"
# pathname1 = "data/"+ filename1+".json"
# filename2 = "_benchmark_results_flat30000"
# pathname2 = "data/"+ filename2+".json"
filename1 = "_benchmark_results_multi10000"
pathname1 = "data/"+ filename1+".json"
filename2 = "_benchmark_results_multiflat10000"
pathname2 = "data/"+ filename2+".json"
plot_SR_metrics(
    pathname1, pathname2,
    save_path='pic/combined_metrics1.png',
    dpi=600  # 更高分辨率
)