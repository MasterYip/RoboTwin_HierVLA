### 7.4. 输出数据格式 (Output Data Format)

**完整JSON结构示例**:

```json
{
  "aggregate_metrics": {
    "policy_name": "HierarchicalQwenPI0",
    "task_config": "demo_randomized",
    "ckpt_setting": "flatpi0",
    "total_episodes": 100,
    "success_metrics": {
      "success_rate": 0.87,
      "success_count": 87,
      "failure_count": 13
    },
    "step_metrics": {
      "mean_steps": 142.5,
      "std_steps": 28.3,
      "min_steps": 95,
      "max_steps": 200,
      "mean_steps_successful": 138.2
    },
    "duration_metrics": {
      "mean_duration": 14.25,
      "std_duration": 2.83,
      "total_duration": 1425.0
    },
    "smoothness_metrics": {
      "mean_overall_smoothness": 0.782,
      "std_overall_smoothness": 0.053,
      "mean_action_smoothness": 0.815,
      "mean_joint_smoothness": 0.749,
      "mean_action_change": 0.0234
    },
    "robustness_metrics": {
      "mean_planning_failures": 0.15,
      "total_planning_failures": 15,
      "mean_collisions": 0.04,
      "total_collisions": 4
    }
  },
  "episodes": [
    {
      "episode_id": 0,
      "seed": 100000,
      "task_name": "stack_blocks_three",
      "instruction": "Stack the three colored blocks from largest to smallest",
      "success": true,
      "completion_steps": 127,
      "step_limit": 200,
      "step_utilization": 0.635,
      "duration_seconds": 12.73,
      "planning_failures": 0,
      "collision_count": 0,
      "smoothness_metrics": {
        "mean_action_change": 0.0218,
        "max_action_change": 0.156,
        "std_action_change": 0.0087,
        "action_smoothness_score": 0.823,
        "mean_joint_acceleration": 0.0124,
        "max_joint_acceleration": 0.089,
        "joint_smoothness_score": 0.758,
        "overall_smoothness": 0.791
      }
    },
    {
      "episode_id": 1,
      "seed": 100001,
      "task_name": "stack_blocks_three",
      "instruction": "Arrange blocks in a tower configuration",
      "success": true,
      "completion_steps": 135,
      "step_limit": 200,
      "step_utilization": 0.675,
      "duration_seconds": 13.51,
      "planning_failures": 0,
      "collision_count": 0,
      "smoothness_metrics": {
        "mean_action_change": 0.0201,
        "max_action_change": 0.142,
        "std_action_change": 0.0079,
        "action_smoothness_score": 0.841,
        "mean_joint_acceleration": 0.0118,
        "max_joint_acceleration": 0.081,
        "joint_smoothness_score": 0.772,
        "overall_smoothness": 0.807
      }
    }
  ]
}
```

### 7.5. 使用方法 (Usage Guide)

**自动化评估 (运行评估脚本时自动启用)**:

```bash
# 标准评估（自动包含benchmark）
bash eval.sh stack_blocks_three demo_randomized pi0_base_aloha_robotwin_full flatpi0 0 0

# 评估完成后，结果保存在：
# eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{timestamp}/_benchmark_results.json
```

**终端输出示例**:

```
Benchmark results saved to: eval_result/.../benchmark_results.json

============================================================
Benchmark Summary
============================================================
Success Rate: 87.0%
Mean Steps: 142.5 ± 28.3
Mean Duration: 14.25s
Overall Smoothness: 0.7820
Planning Failures: 15
============================================================
```

**数据分析示例 (Python脚本)**:

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载benchmark数据
with open('eval_result/.../benchmark_results.json', 'r') as f:
    data = json.load(f)

# 转换为DataFrame便于分析
episodes_df = pd.DataFrame(data['episodes'])

# 分析成功vs失败的平滑度差异
success_smoothness = episodes_df[episodes_df['success']==True]['smoothness_metrics'].apply(lambda x: x['overall_smoothness'])
failure_smoothness = episodes_df[episodes_df['success']==False]['smoothness_metrics'].apply(lambda x: x['overall_smoothness'])

print(f"Success smoothness: {success_smoothness.mean():.3f}")
print(f"Failure smoothness: {failure_smoothness.mean():.3f}")

# 绘制步数分布
episodes_df['completion_steps'].hist(bins=20)
plt.xlabel('Steps')
plt.ylabel('Frequency')
plt.title('Task Completion Steps Distribution')
plt.show()
```