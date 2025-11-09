# Group003 Task Record

## I. Project Overview (项目概览)

### Core Objective (核心任务)

- Compare performance between Flat VLA and Hierarchical VLA strategies
- Validate the impact of hierarchical structure on model capabilities

---

## II. Baseline VLA Strategy (基础VLA策略构建)

### Model Selection (模型选择)

- **Selected Model**:
  - [ ] Wall-OSS
  - [x] π₀

### Deployment Environment (任务部署)

#### Simulation Platform (仿真平台)

- **Platform**: **RoboTwin** / RoboCasa / Custom Benchmark
- **Docker Configuration**:
  - Image: `25fall-masteryip-hier-vla:v1.0_gpu`
    ![alt text](imgs/docker_image.png)

#### Task Types (任务类型)

- **Simple Tasks (初始任务)**:
  - [ ] Blocks Ranking Size - Data Collection
  - [ ] Stack Blocks Three - Data Collection

- **Complex Tasks (复杂任务)**:
  - [ ] "按指令组装工具" (Assemble tools by instruction)
  - [ ] "整理混杂餐具并归位" (Organize and return utensils)

---

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
# repo_id: The name of the dataset (e.g., my_repo)
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
> export OPENPI_DATA_HOME=../../.cache/openpi
> # Use abs dir
> export OPENPI_DATA_HOME=/inspire/ssd/project/25jinqiu07/public/hiervla_003/RoboTwin_HierVLA/.cache/openpi
> ```
>
> AND you should put `paligemma_tokenizer` and `pi0_base` into the cache folder.

```bash
# compute norm_stat for dataset
# uv run scripts/compute_norm_stats.py --config-name ${train_config_name}
uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full

# train_config_name: The name corresponding to the config in _CONFIGS, such as pi0_base_aloha_robotwin_full
# model_name: You can choose any name for your model
# gpu_use: if not using multi gpu,set to gpu_id like 0;else set like 0,1,2,3
# bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
#bash finetune.sh pi0_base_aloha_robotwin_full demo_clean 0,1,2,3
bash finetune.sh pi0_base_aloha_robotwin_full flatpi0 0,1,2,3
```

**Eval Trained Pi0 Model Commands**

```bash
# Under RoboTwin_HierVLA/policy/pi0 directory
# ckpt_path like: policy/pi0/checkpoints/pi0_base_aloha_robotwin_full/demo_clean/30000
bash eval.sh ${task_name} ${task_config} ${train_config_name} ${model_name} ${seed} ${gpu_id}
bash eval.sh place_burger_fries demo_randomized pi0_base_aloha_robotwin_full flatpi0 0 0
# bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
# This command trains the policy using the `demo_clean` setting ($model_name)
# and evaluates it using the same `demo_clean` setting ($task_config).

# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh blocks_ranking_rgb demo_randomized pi0_base_aloha_robotwin_full demo_clean 0 0
```

---

### Implementation (实现方案)

#### Flat VLA Fine-tuning (扁平化策略微调)

- **Input**: Instruction + Image
- **Output**: Direct action sequence generation
- **Training Progress**:
  - Dataset preparation: ___________
  - Training status: ___________
  - Checkpoint: ___________

---

## III. Hierarchical VLA Strategies (分层VLA策略实现)

### Strategy 1: Hierarchical Prompting (分层提示策略)

- **Method**: VLM as high-level planner → VLA executor
- **Reference**: Hi Robot, etc.
- **Implementation Notes**:
  - High-level planner: ___________
  - Low-level executor: ___________
  - Integration status: ___________

#### Qwen Deployment

Qwen: <https://github.com/QwenLM/Qwen3-VL/tree/main>

dependency:

```bash
conda deactivate
source .venv/bin/activate
pip install "transformers>=4.57.0"
pip install accelerate
```

### Strategy 2: Internal Hierarchical Modeling (内化分层策略)

- **Method**: Multi-stage VLA output (subtask planning + action sequence)
- **Reference**: Wall-OSS, π₀.₅, etc.
- **Implementation Notes**:
  - Architecture design: ___________
  - Training approach: ___________
  - Status: ___________

---

## IV. Performance Comparison (性能对比)

### Evaluation Metrics (评估维度)

#### Task Success Rate (任务成功率)

- **Test Cases**:
  - Blocks Ranking Size: ___________
  - Stack Blocks Three: ___________
  - Complex Task 1: ___________
  - Complex Task 2: ___________

#### Action Rationality (动作合理性)

- **Expert Scoring**: ___________
- **Automated Metrics**:
  - Action smoothness: ___________
  - Path efficiency: ___________

#### Generalization Capability (策略泛化能力)

- **Unseen Environments**: ___________
- **Novel Tasks**: ___________
- **Metrics**:
  - Adaptability: ___________
  - Robustness: ___________

### Results Summary (结果汇总)

| Strategy | Success Rate | Action Quality | Generalization |
|----------|--------------|----------------|----------------|
| Flat VLA | ___________  | ___________    | ___________    |
| Hierarchical Prompting | ___________ | ___________ | ___________ |
| Internal Hierarchical | ___________ | ___________ | ___________ |

---

## V. Ablation Studies & Analysis (消融与机制分析)

### Ablation Experiments (消融实验)

- **Module 1**: ___________
  - Impact: ___________
- **Module 2**: ___________
  - Impact: ___________

### Insights & Improvements (改进方向)

- **Strengths & Weaknesses**:
  - Flat VLA: ___________
  - Hierarchical strategies: ___________
  
- **Optimization Paths**:
  - ___________
  - ___________

---

## VI. Timeline & Milestones (时间线与里程碑)

- [ ] Environment setup & data collection
- [ ] Baseline Flat VLA training
- [ ] Hierarchical strategy 1 implementation
- [ ] Hierarchical strategy 2 implementation
- [ ] Performance evaluation
- [ ] Ablation studies
- [ ] Final analysis & report

---

## VII. Notes & Issues (备注与问题记录)

### Technical Issues (技术问题)

- ___________

### Solutions & Workarounds (解决方案)

- ___________

### References (参考资料)

- ___________
