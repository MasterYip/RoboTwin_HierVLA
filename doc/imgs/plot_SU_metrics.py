import json
import numpy as np
import matplotlib.pyplot as plt

# filename1 = "_benchmark_results_30000"
# pathname1 = "data/"+ filename1+".json"
# filename2 = "_benchmark_results_flat30000"
# pathname2 = "data/"+ filename2+".json"

# filename1 = "_benchmark_results_multi10000"
# pathname1 = "data/"+ filename1+".json"
# filename2 = "_benchmark_results_multiflat10000"
# pathname2 = "data/"+ filename2+".json"

filename1 = "multi20000_yl"
pathname1 = "data/"+ filename1+".json"
filename2 = "multi20000_flat"
pathname2 = "data/"+ filename2+".json"

def get_consistent_iterations(data1, data2, manual_iterations=None):
    """
    确定一致的迭代次数
    优先使用manual_iterations，否则使用两个JSON中较小的迭代次数
    """
    if manual_iterations is not None:
        return manual_iterations
    return min(len(data1['episodes']), len(data2['episodes']))

def extract_success_utilizations(data, max_iterations):
    """从数据中提取成功实验的step_utilization，最多取max_iterations次"""
    return [ep['step_utilization'] for ep in data['episodes'][:max_iterations] if ep['success']]

# 配置参数
FILE_PATHS = [pathname1, pathname2]  # 替换为你的两个JSON文件路径
COLORS = ['#1f77b4', '#ff7f0e']  
MARKERS = ['o', '^']  
LABELS = ['SU_Hierarchical VLA', 'SU_Flat VLA']  
OUTPUT_PATH = 'pic/step_utilization_comparison.png'  
DPI = 600  

# 加载数据
with open(FILE_PATHS[0], 'r') as f:
    data1 = json.load(f)
with open(FILE_PATHS[1], 'r') as f:
    data2 = json.load(f)

# 确定一致的迭代次数（可在此处设置manual_iterations）
# max_iterations = get_consistent_iterations(data1, data2)
max_iterations = 40

# 收集所有数据点用于统一x轴
all_utilizations = []
for data in [data1, data2]:
    utilizations = extract_success_utilizations(data, max_iterations)
    all_utilizations.extend(utilizations)

# 创建统一的x轴值（排序后的唯一值）
sorted_unique = sorted(list(set(all_utilizations)))
x_values = np.arange(len(sorted_unique))

plt.figure(figsize=(12, 7))

# 绘制两条曲线
for i, data in enumerate([data1, data2]):
    utilizations = extract_success_utilizations(data, max_iterations)
    
    # 统计每个utilization值的出现次数
    counts = {}
    for val in utilizations:
        counts[val] = counts.get(val, 0) + 1
    
    # 对应到统一的x轴
    y_values = [counts.get(val, 0) for val in sorted_unique]
    
    # 绘制曲线
    plt.plot(x_values, y_values, 
             color=COLORS[i], 
             marker=MARKERS[i], 
             linestyle='-', 
             linewidth=2,
             markersize=8,
             label=LABELS[i])
    
    # 标注每个数据点
    for x, y in zip(x_values, y_values):
        if y > 0:
            plt.text(x, y+0.2, str(y), 
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    color=COLORS[i])

# 图表美化
plt.xticks(x_values, [f"{val:.3f}" for val in sorted_unique], rotation=45)
plt.title('Step Utilization Distribution Comparison', fontsize=16, pad=20)
plt.xlabel('Step Utilization Value', fontsize=12)
plt.ylabel('Number of Successful Episodes', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
print(f"图片已保存至: {OUTPUT_PATH}")
plt.show()


# import json
# import matplotlib.pyplot as plt
# import numpy as np

# # filename1 = "_benchmark_results_30000"
# # pathname1 = "data/"+ filename1+".json"
# # filename2 = "_benchmark_results_flat30000"
# # pathname2 = "data/"+ filename2+".json"

# # filename1 = "_benchmark_results_multi10000"
# # pathname1 = "data/"+ filename1+".json"
# # filename2 = "_benchmark_results_multiflat10000"
# # pathname2 = "data/"+ filename2+".json"

# filename1 = "multi20000_yl"
# pathname1 = "data/"+ filename1+".json"
# filename2 = "multi20000_flat"
# pathname2 = "data/"+ filename2+".json"

# # 配置参数
# FILE_PATHS = [pathname1, pathname2]  # 替换为你的两个JSON文件路径
# COLORS = ['#1f77b4', '#ff7f0e']  # 两条曲线的颜色
# MARKERS = ['o', '^']  # 数据点标记形状 (圆形和三角形)
# LABELS = ['SU_Hierarchical VLA', 'SU_Flat VLA']  # 图例标签
# OUTPUT_PATH = 'pic/step_utilization_comparison.png'  # 输出图片路径
# DPI = 600  # 图片分辨率

# plt.figure(figsize=(12, 7))

# def extract_success_utilizations(file_path):
#     """从JSON文件中提取成功实验的step_utilization"""
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return [ep['step_utilization'] for ep in data['episodes'] if ep['success']]

# # 收集所有数据点用于统一x轴
# all_utilizations = []
# for file_path in FILE_PATHS:
#     utilizations = extract_success_utilizations(file_path)
#     all_utilizations.extend(utilizations)

# # 创建统一的x轴值（排序后的唯一值）
# sorted_unique = sorted(list(set(all_utilizations)))
# x_values = np.arange(len(sorted_unique))

# # 绘制两条曲线
# for i, file_path in enumerate(FILE_PATHS):
#     utilizations = extract_success_utilizations(file_path)
    
#     # 统计每个utilization值的出现次数
#     counts = {}
#     for val in utilizations:
#         counts[val] = counts.get(val, 0) + 1
    
#     # 对应到统一的x轴
#     y_values = [counts.get(val, 0) for val in sorted_unique]
    
#     # 绘制曲线（带标记点）
#     plt.plot(x_values, y_values, 
#              color=COLORS[i], 
#              marker=MARKERS[i], 
#              linestyle='-', 
#              linewidth=2,
#              markersize=8,
#              label=LABELS[i])
    
#     # 标注每个数据点
#     for x, y in zip(x_values, y_values):
#         if y > 0:  # 只标注有数据的位置
#             plt.text(x, y+0.2, str(y), 
#                     ha='center', 
#                     va='bottom',
#                     fontsize=9,
#                     color=COLORS[i])

# # 图表美化
# plt.xticks(x_values, [f"{val:.3f}" for val in sorted_unique], rotation=45)
# plt.title('Step Utilization Distribution Comparison', fontsize=16, pad=20)
# plt.xlabel('Step Utilization Value', fontsize=12)
# plt.ylabel('Number of Successful Episodes', fontsize=12)
# plt.grid(alpha=0.3)
# plt.legend(fontsize=12)
# plt.tight_layout()

# # 高清保存
# plt.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches='tight')
# print(f"图片已保存至: {OUTPUT_PATH}")

# plt.show()