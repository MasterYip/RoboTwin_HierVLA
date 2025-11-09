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

Data Collection Commands:

```bash
bash collect_data.sh stack_blocks_three demo_randomized 0
bash collect_data.sh blocks_ranking_rgb demo_randomized 0
```

- **Complex Tasks (复杂任务)**:
  - [ ] "按指令组装工具" (Assemble tools by instruction)
  - [ ] "整理混杂餐具并归位" (Organize and return utensils)

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
