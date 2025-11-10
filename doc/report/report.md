# Group003 项目报告：扁平化与分层VLA策略对比分析

**项目成员:**  叶雷 (镜像搭建以及任务实现)，李毅恒(分层策略以及任务实现), C角 (消融实验以及报告)
**项目周期:** 48小时

---

## 1. 项目概览 (Project Overview)

### 1.1. 项目背景与挑战

传统的扁平化VLA（Flat VLA）模型，如`PI0`，采用单一的端到端映射，即 $(\text{Vision}, \text{Language}, \text{State}) \rightarrow \text{Actions}$。这种结构在处理多步骤、长周期的复杂操作任务时，面临以下挑战：

* **可解释性差 (Limited interpretability)**：决策过程是一个“黑盒”，难以调试。
* **学习效率低 (Inefficient learning)**：模型必须从头学习所有行为，难以泛化。
* **泛化能力弱 (Poor generalization)**：难以将学到的技能迁移到新的任务变体中。

### 1.2. 核心任务 (Core Objective)

本项目旨在复现一个扁平化VLA（`PI0`）作为基线（Baseline），并在此基础上，设计、实现并对比两种分层VLA（Hierarchical VLA）策略。

我们将重点评估分层结构在**任务成功率、动作合理性（效率与平滑度）、泛化鲁棒性**方面的提升，并分析其对模型可解释性、样本效率和推理开销的影响。

---

## 2. 镜像环境配置、代码管理及数采微调管线搭建

### 2.1. 服务器镜像配置及一键部署 (Environment Setup)

基于原始镜像`25fall-robotwin-h200:vulkan-cuda12.8`，配置新的RoboTwin环境镜像`25fall-masteryip-hier-vla:v1.x_gpu`系列，集成以下组件：

* RoboTwin仿真平台
* Pi0模型代码及依赖
* Hierarchical VLA相关代码及依赖
* 数据采集与处理脚本
此镜像可一键部署，方便团队成员快速搭建环境。

<p align="center">
  <img src="../imgs/docker_images.png" height="300">
  <br>
  <text>Docker系列镜像</text>
</p>

### 2.2. 代码管理与协作 (Code Management)

基于`RoboTwin`开源代码仓库，创建`RoboTwin_HierVLA`新仓库，进行代码版本控制与协作开发。

<p align="center">
  <img src="../imgs/git_vcs.png" height="300">
  <br>
  <text>Docker系列镜像</text>
</p>

### 2.3. Xmind思维导图工作流 (Xmind Workflow)

基于Xmind思维导图，规划项目工作流与任务分配，确保各成员明确职责与时间节点。

<p align="center">
  <img src="../imgs/task_assign.png" height="250">
  <img src="../imgs/task_decompose.png" height="250">
  <br>
  <text>Xmind思维导图工作流</text>
</p>

### 2.4. 数据采集与微调管线搭建 (Data Collection & Fine-tuning Pipeline)

> [!NOTE]
> Checkout Pi0 train data gen & training command see <https://robotwin-platform.github.io/doc/usage/Pi0.html#1-environment-setup>

**Data Collection**

```bash
# Under RoboTwin_HierVLA root directory
bash collect_data.sh stack_blocks_three demo_randomized 0
bash collect_data.sh blocks_ranking_rgb demo_randomized 1
```

**Convert Data to pi0 training data**

```bash
# Under RoboTwin_HierVLA/policy/pi0 directory
mkdir processed_data && mkdir training_data
# bash process_data_pi0.sh ${task_name} ${task_config} ${expert_data_num}
bash process_data_pi0.sh stack_blocks_three demo_randomized 50
bash process_data_pi0.sh place_burger_fries demo_randomized 50

# hdf5_path: The path to the generated HDF5 data (e.g., ./training_data/${model_name}/)
# bash generate.sh ${hdf5_path} ${repo_id}
bash generate.sh ./training_data/flatpi0/ flatpi0
```

**Finetune Model**

> In `RoboTwin_HierVLA/policy/pi0/src/openpi/training/config.py`, you only need to write repo_id on your datasets.(e.g., repo_id=demo_clean_repo)

> [!WARNING]
> Change UV source for uv update:
>
> ```bash
> export UV_INDEX_URL=http://nexus.sii.shaipower.online/repository/pypi/simple/
> ```
>
> Update openpi cache path by
>
> ```bash
> export OPENPI_DATA_HOME=/inspire/ssd/project/25jinqiu07/public/hiervla_003/RoboTwin_HierVLA/.cache/openpi
> ```
>
> AND you should put `paligemma_tokenizer` and `pi0_base` into the cache folder.

```bash
# compute norm_stat for dataset
# uv run scripts/compute_norm_stats.py --config-name ${train_config_name}
uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full

# bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
bash finetune.sh pi0_base_aloha_robotwin_full flatpi0 0,1,2,3
```

**Eval Trained Pi0 Model Commands**

```bash
# Under RoboTwin_HierVLA/policy/pi0 directory
bash eval.sh ${task_name} ${task_config} ${train_config_name} ${model_name} ${seed} ${gpu_id}
bash eval.sh place_burger_fries demo_randomized pi0_base_aloha_robotwin_full flatpi0 0 0
```

---

## 3. 基线 VLA 策略 (Baseline: Flat VLA)

我们选用`pi0_base`预训练模型，并在`RoboTwin`平台的`demo_randomized`环境下进行微调，作为扁平化VLA策略的基线实现。该模型直接将视觉、语言和状态输入映射为动作输出，适用于多种操作任务。

---

## 4. 分层 VLA 策略实现 (Hierarchical VLA Strategies)

### 4.1. 整体架构设计 (Architecture Design)

分层VLA策略采用**两阶段规划执行框架**，将传统扁平化VLA的单一映射过程解耦为"高层规划"与"低层执行"两个独立模块。该架构的核心思想是：利用大型视觉-语言模型（VLM）的强大理解与推理能力进行任务分解，同时保留底层VLA模型在精细运动控制上的优势。

**架构层次划分：**

* **高层规划器（High-Level Planner）**：基于Qwen3-VL-8B-Instruct视觉-语言模型，负责理解复杂任务指令并进行阶段性分解。
* **低层执行器（Low-Level Executor）**：复用PI0基线模型，负责将高层规划生成的运动级指令转化为精确的关节动作序列。

该设计参考了Hi Robot等工作中的分层提示策略（Hierarchical Prompting），但在实现上进行了针对性改进，以解决传统分层方法中存在的计划一致性问题。

<p align="center">
  <img src="../imgs/HierVLA_sch.svg" height="600">
  <br>
  <text>工作流程图示 (Workflow Diagram)</text>
</p>

### 4.3. 两阶段规划机制 (Two-Phase Planning Mechanism)

本实现采用创新的两阶段规划机制，结合**基于感知的进度评估（Perception-Based Progress Evaluation）**，有效解决了传统分层方法中存在的计划漂移（Plan Drift）和进度判断不准确的问题：

**阶段一：初始高层规划（Initial High-Level Planning）**

在任务开始时，Qwen3-VL接收主任务指令和初始视觉观测，生成一个固定的高层计划（3-6个里程碑式步骤）。例如，对于"整理餐桌"任务，可能生成：

1. 识别并定位餐具位置
2. 左臂抓取盘子，右臂抓取杯子
3. 将餐具移动至收纳区
4. 释放并归位双臂

这一初始计划在整个任务执行过程中保持不变，作为后续所有决策的上下文基准。

**阶段二：进度感知的运动指令生成与完成度评估（Progress-Aware Motion Command Generation with Completion Evaluation）**

在执行过程中，系统每隔N步（默认10步，约1秒）重新调用Qwen3-VL，执行**双重任务**：

1. **生成运动级指令（Motion Command Generation）**
   * 输入包含**初始计划全文**，并标注当前进度（✓已完成、→当前执行、○待执行）
   * 输入包含**当前视觉观测**，用于判断实际执行状态
   * 输出**单一运动级指令**，描述未来约10秒内双臂的具体动作（如"左臂：抓取红色方块。右臂：保持当前姿态"）

2. **评估当前子任务完成度（Subtask Completion Evaluation）**
   * **关键创新**：摒弃传统的步数计数器（Step Counter）方法，改用**视觉感知判断**
   * VLM基于当前图像观测，明确回答当前子任务是否已完成（YES/NO）
   * 输出完成度判断的**视觉依据**（如"红色方块已被抓取并抬起，离开桌面"）
   * 仅当VLM明确判断当前子任务完成时，系统才自动推进至下一子任务

**输出格式示例（Structured Output）：**

```
MOTION_COMMAND: Left arm: approach and grasp the red block. Right arm: maintain current position.
SUBTASK_COMPLETE: NO
COMPLETION_REASONING: The red block is still on the table surface, not yet grasped by the gripper.
PROGRESS_SUMMARY: Approaching target object, grasp action in progress.
```

这种设计确保了：

1. **一致性（Consistency）**：所有运动指令都参考同一份初始计划，避免了重复规划导致的目标漂移。
2. **适应性（Adaptability）**：通过视觉反馈动态调整运动细节，应对执行偏差。
3. **准确性（Accuracy）**：基于视觉感知的完成度判断，比固定步数更可靠，能适应不同执行速度和意外情况。
4. **可解释性（Interpretability）**：显式的进度标注和完成度推理使得调试和干预成为可能。

**为何摒弃步数计数器（Why Not Step Counting）：**

传统方法使用固定步数（如50步）来判断子任务完成，存在以下问题：

* **不准确**：不同子任务耗时差异大（抓取可能需要20步，移动可能需要80步）
* **不鲁棒**：执行速度受环境干扰影响，固定步数无法适应
* **不灵活**：无法处理提前完成或执行失败的情况

改用视觉感知判断后，系统能够：

* **动态适应**：根据实际执行状态决定是否推进，而非盲目计数
* **及时响应**：子任务提前完成时立即推进，提高效率
* **异常处理**：长时间未完成时可检测到（始终返回NO），便于干预

### 4.4. 代码实现细节 (Implementation Details)

**核心模块组成：**

1. **`qwen3vl_model.py`** - Qwen3VL高层规划器封装

   该模块实现了Qwen3-VL-8B-Instruct的推理接口，核心类`Qwen3VLPlanner`提供以下关键方法：

   * `generate_initial_plan(images, state)`: 接收初始观测，调用VLM生成3-6步高层计划，返回子任务列表。内部使用专门设计的规划提示词，要求模型输出结构化的编号列表。

   * `generate_motion_command_with_evaluation(images, state)`: **核心创新方法**，同时完成两项任务：
     * 基于当前观测和执行进度，生成单条运动级指令
     * **通过视觉感知评估当前子任务是否完成**，返回YES/NO判断及推理依据
     * 返回结构化字典，包含：`motion_command`（PI0指令）、`current_subtask_complete`（完成标志）、`completion_reasoning`（视觉依据）、`progress_summary`（进度摘要）

   * `process_completion_evaluation(evaluation_result)`: 处理完成度评估结果，若VLM判断子任务已完成（YES），则自动调用`mark_subtask_completed()`推进至下一子任务。

   * `mark_subtask_completed()`: 标记当前子任务完成，更新进度索引，触发下一子任务的运动指令生成。

   * `get_progress_info()`: 返回当前规划状态的完整信息，包括完成度评估历史，用于日志记录和调试。

   **提示词工程（Prompt Engineering）**：

   该模块的关键在于精心设计的双任务提示词（`_construct_motion_command_with_evaluation_prompt`）。提示词要求VLM严格按照以下格式输出：

   ```
   MOTION_COMMAND: [具体动作指令]
   SUBTASK_COMPLETE: [YES/NO - 必须基于当前图像的视觉证据]
   COMPLETION_REASONING: [完成度判断的视觉依据]
   PROGRESS_SUMMARY: [整体进度描述]
   ```

   提示词中明确要求：
   * **视觉依据优先**："仅当你能在当前图像中看到明确的视觉证据证明子任务目标已达成时，才回答YES"
   * **具体示例引导**：提供正负样本（如"抓取方块" → 方块已被抓起并离开桌面则YES，否则NO）
   * **禁止假设**："基于当前图像判断，而非假设或推测"

   模型加载采用Hugging Face Transformers库，支持自动设备映射（`device_map="auto"`）和混合精度推理（`dtype="auto"`），在单张GPU上显存占用约8GB。

2. **`hier_qwen_pi.py`** - 分层策略协调器

   该模块定义了`HierarchicalQwenPI0`类，整合高层规划器与低层执行器，实现完整的分层决策流程：

   * **初始化阶段**: 同时加载Qwen3VL规划器和PI0执行器，设置重规划频率（`replan_frequency`，默认10步）。**移除**了原`steps_per_subtask`参数，因为不再依赖步数计数。

   * **观测更新逻辑** (`update_observation_window`):
     * 首次调用时触发`generate_initial_plan()`，生成固定的高层计划。
     * 后续调用时，根据步数计数器决定是否调用`generate_motion_command_with_evaluation()`。
     * **新增完成度评估处理**：每次调用VLM后，立即处理返回的`SUBTASK_COMPLETE`字段。若为YES，自动推进至下一子任务并重新生成运动指令。
     * 将当前运动指令传递给PI0执行器，更新其语言输入和视觉观测窗口。

   * **动作生成** (`get_action`): 直接调用PI0执行器的`get_action()`方法，返回动作块（通常为10×14的关节速度序列）。

   * **状态管理**: 维护`step_count`（总步数）、`motion_command_step_count`（当前运动指令已执行步数）、`initial_plan_generated`（初始规划标志）等状态变量。**移除**了`subtask_step_count`（子任务步数计数器），改为依赖VLM的视觉判断。

3. **`deploy_policy.py`** - 策略工厂接口

   在原有的`get_model()`函数中增加了条件分支，通过配置参数`hierarchical=True`切换至分层策略模式：

   ```python
   if usr_args.get("hierarchical", False):
       return HierarchicalQwenPI0(...)
   else:
       return PI0(...)
   ```

   这一设计保持了与现有评估脚本（`eval.sh`）的完全兼容性，无需修改环境交互代码。

**接口兼容性保证：**

分层策略类实现了与扁平化PI0模型完全一致的公共接口：

* `observation_window` 属性（暴露PI0的观测窗口）
* `set_language(instruction)` 方法（设置主任务指令）
* `update_observation_window(img_arr, state)` 方法（更新观测）
* `get_action()` 方法（获取动作输出）
* `reset_obsrvationwindows()` 方法（重置状态）

**完成度评估的鲁棒性设计：**

为处理VLM输出不稳定的情况，`_parse_motion_command_with_evaluation()`方法采用了多重解析策略：

1. **正则表达式提取**：优先使用正则匹配结构化字段
2. **关键词匹配**：若结构化解析失败，尝试匹配关键词（YES/NO）
3. **默认值兜底**：解析完全失败时，默认为NO（保守策略，避免错误推进）

这确保了即使VLM偶尔输出格式不规范，系统仍能稳定运行。

---

## 5. 性能基准测试系统 (Performance Benchmarking System)

### 5.1. 基准测试框架概述 (Benchmark Framework Overview)

为了系统性地评估不同VLA策略的性能，我们开发了一套自动化的基准测试框架。该框架不仅记录传统的成功率指标，还引入了多维度的定量评估体系，包括动作平滑度、执行效率、系统鲁棒性等关键指标。

**设计目标 (Design Objectives):**

1. **全面性 (Comprehensiveness)**: 覆盖任务成功率、执行效率、动作质量、系统鲁棒性四大维度
2. **自动化 (Automation)**: 无需人工干预，自动记录所有关键指标
3. **可追溯性 (Traceability)**: 保存每个episode的详细数据，支持事后分析
4. **可视化友好 (Visualization-Ready)**: 输出JSON格式，便于生成图表和报告

### 5.2. 核心模块设计 (Core Module Design)

基准测试系统由三个核心类组成，位于 `envs/utils/benchmark.py`：

#### 1. EpisodeBenchmark 类 (Episode-Level Tracker)

**功能**: 追踪单个任务执行过程中的所有细粒度指标。

**关键属性**:

```python
class EpisodeBenchmark:
    # 基本信息
    episode_id: int          # 试验编号
    seed: int                # 随机种子
    task_name: str           # 任务名称
    instruction: str         # 语言指令
    
    # 执行结果
    success: bool            # 是否成功
    completion_steps: int    # 完成步数
    step_limit: int          # 步数上限
    duration_seconds: float  # 执行时长
    
    # 动作轨迹数据
    actions: List[ndarray]         # 每步的动作向量
    joint_states: List[ndarray]    # 每步的关节状态
    
    # 平滑度指标
    action_velocities: List[ndarray]     # 动作变化率
    joint_accelerations: List[ndarray]   # 关节加速度
    
    # 异常统计
    planning_failures: int   # 规划失败次数
    collision_count: int     # 碰撞次数
```

**核心方法**:

* `record_step(action, joint_state)`: 在每个仿真步骤调用，自动记录动作和关节状态

  ```python
  def record_step(self, action: np.ndarray, joint_state: np.ndarray):
      self.actions.append(action.copy())
      self.joint_states.append(joint_state.copy())
      self.completion_steps += 1
      
      # 计算动作速度（相邻步骤的动作差）
      if len(self.actions) > 1:
          action_diff = np.abs(self.actions[-1] - self.actions[-2])
          self.action_velocities.append(action_diff)
      
      # 计算关节加速度（二阶导数）
      if len(self.joint_states) > 2:
          vel_curr = self.joint_states[-1] - self.joint_states[-2]
          vel_prev = self.joint_states[-2] - self.joint_states[-3]
          accel = np.abs(vel_curr - vel_prev)
          self.joint_accelerations.append(accel)
  ```

* `compute_smoothness_metrics()`: 计算多种平滑度评分

  ```python
  def compute_smoothness_metrics(self) -> Dict[str, float]:
      metrics = {}
      
      # 动作平滑度：基于动作变化的方差
      action_vels = np.array(self.action_velocities)
      variance = np.var(action_vels)
      metrics['action_smoothness_score'] = 1.0 / (1.0 + variance)
      
      # 关节平滑度：基于关节加速度的方差
      joint_accels = np.array(self.joint_accelerations)
      jerk_variance = np.var(joint_accels)
      metrics['joint_smoothness_score'] = 1.0 / (1.0 + jerk_variance)
      
      # 综合平滑度：两者的平均值
      metrics['overall_smoothness'] = (
          metrics['action_smoothness_score'] + 
          metrics['joint_smoothness_score']
      ) / 2.0
      
      return metrics
  ```

#### 2. PolicyBenchmark 类 (Policy-Level Aggregator)

**功能**: 聚合多个episode的数据，计算整体统计指标。

**核心方法**:

* `start_episode()`: 开始新的episode追踪
* `record_step()`: 转发给当前episode的tracker
* `mark_episode_success()`: 标记episode完成并保存结果
* `compute_aggregate_metrics()`: 计算聚合统计量

**聚合指标示例**:

```python
{
    "success_metrics": {
        "success_rate": 0.85,           # 成功率 85%
        "success_count": 85,
        "failure_count": 15
    },
    "step_metrics": {
        "mean_steps": 127.3,            # 平均步数
        "std_steps": 23.5,              # 步数标准差
        "min_steps": 89,                # 最少步数
        "max_steps": 200,               # 最多步数
        "mean_steps_successful": 115.2  # 成功case平均步数
    },
    "smoothness_metrics": {
        "mean_overall_smoothness": 0.78,    # 平均平滑度
        "mean_action_smoothness": 0.82,     # 动作平滑度
        "mean_joint_smoothness": 0.74,      # 关节平滑度
        "mean_action_change": 0.023         # 平均动作变化量
    },
    "robustness_metrics": {
        "mean_planning_failures": 0.12,     # 平均规划失败次数
        "total_planning_failures": 12,
        "mean_collisions": 0.03,            # 平均碰撞次数
        "total_collisions": 3
    }
}
```

#### 3. 与环境集成 (Environment Integration)

**修改点 1**: `envs/_base_task.py` - 在 `__init__` 中初始化benchmark追踪

```python
def _init_task_env_(self, ...):
    # ...existing code...
    
    # Benchmark tracking
    self.benchmark_tracker = None
    self.benchmark_enabled = kwags.get("benchmark_enabled", False)
```

**修改点 2**: `envs/_base_task.py` - 在 `take_action` 中记录每步数据

```python
def take_action(self, action, action_type='qpos'):
    # ...existing code...
    
    # Get current joint states
    current_jointstate = np.array(left_jointstate + right_jointstate)
    
    # Benchmark tracking: record action and joint state
    if self.benchmark_enabled and self.benchmark_tracker:
        self.benchmark_tracker.record_step(action, current_jointstate)
    
    # ...existing execution code...
```

**修改点 3**: `envs/_base_task.py` - 在任务成功时自动标记

**修改点 4**: `script/eval_policy.py` - 评估流程集成

### 5.3. 平滑度评估算法 (Smoothness Evaluation Algorithm)

平滑度是评估机器人动作质量的关键指标。我们采用多层次的平滑度计算方法：

**层次 1: 动作层平滑度 (Action-Level Smoothness)**

基于相邻时间步的动作变化量：

$$
\text{ActionVelocity}_t = |a_t - a_{t-1}|
$$

$$
\text{ActionSmoothnessScore} = \frac{1}{1 + \text{Var}(\text{ActionVelocity})}
$$

* 低方差 → 动作变化平稳 → 高平滑度分数
* 高方差 → 动作变化剧烈 → 低平滑度分数

**层次 2: 关节层平滑度 (Joint-Level Smoothness)**

基于关节加速度（二阶导数）：

$$
\text{JointAcceleration}_t = |(q_t - q_{t-1}) - (q_{t-1} - q_{t-2})|
$$

$$
\text{JointSmoothnessScore} = \frac{1}{1 + \text{Var}(\text{JointAcceleration})}
$$

* 低加速度方差 → 运动轨迹平滑 → 高平滑度分数
* 高加速度方差 → 运动抖动明显 → 低平滑度分数

**层次 3: 综合平滑度 (Overall Smoothness)**

$$
\text{OverallSmoothness} = \frac{\text{ActionSmoothnessScore} + \text{JointSmoothnessScore}}{2}
$$

**实际意义**:

* 分数 > 0.8: 运动非常平滑，接近人类操作水平
* 分数 0.6-0.8: 运动较平滑，可接受
* 分数 < 0.6: 运动抖动明显，需要优化

### 5.4. 关键优势 (Key Advantages)

1. **零侵入性 (Non-Invasive)**: 通过环境接口注入，无需修改策略代码
2. **细粒度追踪 (Fine-Grained)**: 记录每一步的动作和状态，支持深度分析
3. **实时计算 (Real-Time)**: 边执行边计算指标，无需事后处理
4. **标准化输出 (Standardized)**: JSON格式，兼容各种可视化工具
5. **可扩展性 (Extensible)**: 易于添加新的评估指标（如能量消耗、安全性等）

### 5.5. 未来扩展方向 (Future Extensions)

* **能量效率 (Energy Efficiency)**: 基于关节速度和负载计算能量消耗
* **安全性指标 (Safety Metrics)**: 跟踪与环境的最小距离、碰撞力度
* **可解释性可视化 (Interpretability Visualization)**: 生成轨迹热图、注意力可视化
* **在线对比分析 (Online Comparison)**: 实时对比多个策略的性能曲线

---

## 6. 性能对比 (Performance Comparison)

### 6.1. 实验1

Base Model: finetuned pi0 (10000 episode on several tasks)
VLA Framework:

* Flat model
* First plan steps and input all at once
* Replan steps every 10 sim step, pass current step instruction

[实验1 VLM-VLA实验数据](data/VLA_compare.csv)
![alt text](../imgs/exp1_vlacmp.png)

### 6.2. 评估维度 (Evaluation Metrics)

我们将根据 `05-evaluation-pipeline.md` 文档，从以下三个维度进行评估：

#### 1. 任务成功率 (Task Success Rate)

* **指标**: `success_rate`。
* **测试用例 (Test Cases)**:
  * Blocks Ranking Size: `___________`
  * Stack Blocks Three: `___________`
  * Complex Task 1 (Assemble tools): `___________`
  * Complex Task 2 (Organize utensils): `___________`

#### 2. 动作合理性 (Action Rationality / Quality)

* **指标 1 (效率)**: `average_steps` (平均步数) 和 `completion_time` (平均完成时间)。
* **指标 2 (平滑度)**: `action_smoothness`。
* **结果**:
  * 平均步数: `___________`
  * 动作平滑度: `___________`

#### 3. 策略泛化能力 (Generalization Capability)

* **方法**: 跨域评估（Cross-Domain Evaluation）。
* **测试**: 使用在 `demo_clean` (干净) 配置下训练的模型，在 `demo_randomized` (视觉/物理随机化) 或 `hard_randomized` (困难随机化) 配置下进行评估。
* **指标**: 成功率下降幅度。
* **结果**:
  * *Clean -> Randomized 成功率*: `___________`

### 6.3. 结果汇总表 (Results Summary)

| 策略 (Strategy) | 成功率 (SR) (简单任务) | 成功率 (SR) (复杂任务) | 动作质量 (平滑度/效率) | 泛化能力 (SR in Randomized) |
|:--- |:--- |:--- |:--- |:--- |
| **Flat VLA (基线)** | `___________` | `___________` | `___________` | `___________` |
| **Strategy 1 (外部)** | `___________` | `___________` | `___________` | `___________` |
| **Strategy 2 (内部)** | `___________` | `___________` | `___________` | `___________` |

---

## 7. 消融与机制分析 (Ablation Studies & Analysis)

### 7.1. 消融实验 (Ablation Experiments)

* **实验 1: 规划器 vs 执行器 (Planner vs. Executor)**
  * *目的*: 验证 `TaskDecompositionModule` (高层规划器) 的必要性。
  * *设置*: 仅使用 `Strategy 1`，将其中的高层规划器替换为一个“扁平”的指令（即直接将原始复杂指令"整理餐桌"喂给`GraspController`）。
  * *预期*: 任务失败，证明高层规划器对于理解复杂指令至关重要。
  * *结果*: `___________`

* **实验 2: 专用技能 vs 通用技能 (Specialized vs. General Skills)**
  * *目的*: 验证 `SkillController` 模块化的优势。
  * *设置*: 在 `Strategy 2` 中，将所有专用的 `SkillController` (Reach, Grasp, Place) 替换为同一个 `Flat VLA` (PI0基线) 来执行所有子任务。
  * *预期*: 成功率下降，或样本效率降低。证明专用技能控制器在学习效率和鲁棒性上的优势。
  * *结果*: `___________`

### 7.2. 洞察与改进 (Insights & Improvements)

* **优势与权衡 (Strengths & Weaknesses)**:
  * **Flat VLA**:
    * *优势*: 结构简单，端到端。
    * *劣势*: 可解释性差，难以调试，泛化能力弱，样本效率低。
  * **Hierarchical (分层策略)**:
    * *优势*: **可解释性强** (显式的子目标)；**样本效率高** (模块化学习，预期2-3倍提升)；**泛化性强** (技能可组合、可复用)；**易于调试**。
    * *权衡 (Trade-off)*: 增加了**推理开销** (约 20ms) 和 **GPU显存占用** (约 1GB)。

* **未来改进路径 (Optimization Paths)**:
  * **Phase 2**: 扩展技能库（`SkillController`），支持工具使用、双臂协同；实现动态重规划（Dynamic Re-planning）。
  * **Phase 3**: 实现技能的在线学习（Online Adaptation）和从演示中学习（Learning from Demonstrations）。

### 7.3

---

## 8. 时间线与里程碑 (Timeline & Milestones)

* [x] 环境搭建与数据收集脚本分析 (`T=0-2h`)
* [ ] 基线 Flat VLA 训练与评估 (`T=2-12h`)
* [ ] 分层策略 1 (外部) 实现与调试 (`T=6-24h`)
* [ ] 分层策略 2 (内部) 数据准备与实现 (`T=6-30h`)
* [ ] 性能评估与数据汇总 (`T=30-40h`)
* [ ] 消融实验 (`T=40-44h`)
* [ ] 最终分析与报告撰写 (`T=44-48h`)

---

## 9. 备注与问题记录 (Notes & Issues)

### 9.1. 技术问题 (Technical Issues)

1. **GPU 内存占用 (OOM)**: `Flat PI0` 基线模型推理需要约 4GB 显存。`HierarchicalPI0` (策略2) 因为需要加载多个技能控制器，预计需要 5GB 显存。
2. **推理延迟 (Inference Latency)**: `Flat PI0` 推理时间约 100ms (`PI0-Base`)。`Strategy 1` (外部) 增加了LLM规划开销，`Strategy 2` (内部) 增加了子任务选择开销，总延迟预计为 120ms。

### 9.2. 解决方案 (Solutions & Workarounds)

1. **内存管理**: 严格遵守 `eval.sh` 中的 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` 设置，限制JAX/TensorFlow仅使用 40% 的显存，防止OOM。
2. **延迟优化**: 如果延迟成为瓶颈，可考虑将模型从 `PI0-Base` (100ms) 切换为 `PI0-FAST` (50ms)，以抵消分层带来的开销。

### 9.3. 参考资料 (References)

* [1] RoboTwin2 Official Doc: <https://robotwin-platform.github.io/doc/usage/index.html>
* [2] DeepWiki: <https://deepwiki.com/RoboTwin-Platform/RoboTwin>
* [3] 项目文件1-6
