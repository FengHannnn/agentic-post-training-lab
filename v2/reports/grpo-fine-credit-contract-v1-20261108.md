# Fine GRPO Credit Comparison — Six-Hour Experiment

本轮比较旧 fine 信用分配与新的动作局部信用，学习率统一为 `1e-6`。两支从最佳加权 SFT 模型 W 独立初始化，每支最多 48 组，整个作业累计最多 6 小时，包含 W、两支最终模型评测、共同更新节点诊断与打包。

## 对照与信用定义

| 名称 | 初始化 | 信用分配 | 学习率 |
|---|---|---|---|
| W | 冻结的最佳加权 SFT | 评测基线 | — |
| legacy | W，独立新建优化器 | `potential_rtg_depth_v1`，沿用 fine RL17 的规则 | `1e-6` |
| local | W，独立新建优化器 | `outcome_local_verified_v2`，终局组内优势加动作局部进展 | `1e-6` |

`legacy` 重新训练，使用的信用规则与 fine RL17 一致。本轮训练课程、任务和评测集已更新，因此 legacy 的本轮成绩由本轮评测产生。

旧方法先计算每步 `outcome_reward + potential_after - potential_before`，累加从该动作到结束的奖励，再按相对动作位置跨轨迹标准化。终止时 potential 回到零，回报中包含 `-potential_before`。相对位置相同的动作可能处于不同业务状态，从而出现低进展轨迹中的无效动作获得正优势。历史第 19 组提供了这一具体例子。

新方法对第 i 条轨迹、第 t 个继续执行动作使用：

`A_local[i,t] = A_outcome[i] + 0.2 × 本动作新取得的已验证指标数`

其中 `A_outcome` 沿用 K=4 的终局回报组内标准化。局部部分只使用环境验证产生的 `new_verified_metrics`，与提示前缀及先前动作去重；重复取证、SQL 错误、无效工具调用的局部加成为零。终止时 potential 清零产生的负差分不计入局部加成。环境逐步验证 potential 和 process_reward 的一致性，审计再次重放轨迹。

局部加成是额外的动作信用设计，明确记录为“outcome + local verification signal”混合方法；它采用即时局部加成，属于自定义动作信用目标。该加成具有正向偏置，和保持原目标的 potential 等价塑形采用不同定义。全部轨迹终局回报相同时，只要存在新验证进展，local 也可以更新；全部动作综合优势均为零时跳过。额外 SFT、DPO 更新均为零。

## 固定条件与预算

两支只允许 `run_id` 与 `credit_method` 不同。固定 W 权重、reference、基础模型版本、任务顺序、采样种子、K=4、每组一次优化、LoRA 配置、TIS、KL、裁剪、训练和评测解码参数。采样仍使用上一轮已验证的同次前向概率探针与原生 LoRA 张量检查。

- W adapter SHA256：`f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f`。
- 训练：沿用扩展后的 24 个训练任务，最多每支 48 组；每个任务配一个完整任务组和一个恢复组。两支合计最多 384 条训练轨迹，最后另采 8 条轨迹检查 checkpoint 刷新。
- 评测：4 个全新 development 任务族，ID 为 `fine-development-000` 至 `003`；与历史 C3、C4、上一轮 curriculum development 分离。
- 每个模型标准评测 132 条：完整 greedy 4、完整 sample 16、恢复 greedy 28、恢复 sample 84；另有 24 条新增错误专项评测、14 次质量检查动作。W 与两支合计 468 条评测轨迹和 42 次质量检查动作。
- 共同节点：两支均达到第 12 次实际更新时，各取首次达到该更新数的 checkpoint，执行 4 条完整 sample 和 12 条专项 sample，共 32 条诊断轨迹。若任一支少于 12 次更新，记录未达到并成对跳过该诊断。诊断结果用于解释信用信号，不参与模型选择。
- 训练组数相同；实际优化更新数按 eligibility 和 KL/clip guard 单独记录。最终比较同时报告有效组比例、实际更新数、非零动作信用数及 common-update 诊断。

累计墙钟预算为 21,600 秒，续跑沿用已消耗预算。作业先测 W 评测耗时，再以实测评测、采样和更新速度估算剩余工作。训练仅在两支完成相同组数、完整/恢复配对周期结束时决定是否继续下一周期。预计剩余时间不足下一周期加最终工作预留时，两支在相同偶数组数停止，并锁定 `training_stop.json`。最多 48 组是上限，实际完成组数以该记录为准。

最终工作预留包括两支完整评测、两支刷新采样、两个小诊断、审计与打包；其中完整评测每支预留 `max(2400 秒, 1.2 × W 实测评测秒数)`，其余合计 1,200 秒。各子进程使用剩余累计预算作为超时；保留 180 秒供打包。上一轮同课程 outcome 作业耗时约 5 小时 20 分钟，本轮 fine 更新较多并增加共同节点诊断，运行时由预留机制决定最终组数。

## 判断规则

主指标为采样完整任务成功数；W 完整任务采样成功率达到 100% 时，主指标转为采样恢复成功数。候选须在采样完整、采样恢复、greedy 完整、greedy 恢复四项成功数均不低于 W，且主指标严格增加。满足条件后按主指标增益、恢复增益排序，同分保留 legacy。报告同时给出 local 对 legacy、两支各自对 W 的配对改善和退步数。

新增错误专项评测和共同更新节点诊断用于定位机制。该轮使用 development 集探索信用分配；候选进入下一轮全新任务族独立确认后，再评定稳定提升。

## 运行与续跑

将 `grpo-fine-credit-v1-20261108-run.zip` 上传到 `/root/rivermind-data/AgenticPostTrainingLab/`，在该目录解压，ZIP 内顶层为 `v2/`。原实例的模型缓存保留。新作业可使用克隆后的另一张 L40S，作业内及断点续跑时固定 GPU UUID、软件和驱动身份。

```bash
cd /root/rivermind-data/AgenticPostTrainingLab
/root/rivermind-data/envs/apt-vllm/bin/python -m zipfile -e grpo-fine-credit-v1-20261108-run.zip .
cd v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_fine.py --config configs/grpo-fine-credit-v1.json
```

发生中断时，在相同实例、同一目录、同一发布代码上续跑：

```bash
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_fine.py --config configs/grpo-fine-credit-v1.json --resume
```

终端显示总阶段进度条，以及采样、打分、反向传播和评测的细分进度条。完成或异常退出后输出 `Download:` 路径和状态。首轮返回包为 `v2/grpo-fine-credit-v1-20261108-review.zip`；续跑返回包带 `attempt-002` 等后缀。优化器状态保留在云端用于续跑。

报告在 v2 项目内的路径为 `reports/grpo-fine-credit-contract-v1-20261108.md`，验证记录为 `reports/grpo-fine-credit-validation-v1-20261108.json`。

## 代码阅读顺序

| 文件相对路径 | 大致作用 |
|---|---|
| `configs/grpo-fine-credit-v1.json` | 总预算、两支配置和停止规则 |
| `src/apt_v2/grpo_fine_study.py` | 新开发任务、信用方法记录、诊断计划和选择规则 |
| `src/apt_v2/grpo_credit.py` | legacy 与 local 的动作优势计算 |
| `src/apt_v2/grpo_curriculum.py` | 训练课程、错误上下文、任务隔离和专家可解性检查 |
| `scripts/run_v2_grpo_fine.py` | 两支调度、成对停止、续跑、进度条和打包 |
| `scripts/grpo_fine_common.py` | 固定配置检查与时间预留计算 |
| `scripts/grpo_worker.py` | 准备模型、采样、缓存打分与执行更新 |
| `src/apt_v2/grpo_optimizer.py` | token 概率、GRPO 损失、LoRA 反向传播和保护门槛 |
| `scripts/probe_v2_grpo_fine.py` | 第 12 次实际更新的配对诊断 |
| `scripts/evaluate_v2_grpo.py` | W 和最终两支的完整、恢复、专项评测 |
| `scripts/audit_v2_grpo_fine.py` | 重放轨迹、重算目标、核对 checkpoint 与汇总对比 |
| `scripts/build_v2_grpo_fine_release.py` | 生成并逐文件校验完整运行 ZIP |

## 本地验证结果

运行包干净解压后，全部 220 项 GRPO 测试通过，耗时约 3 分 31 秒；258 个绑定文件的 SHA256 校验通过，四个新增脚本的命令行入口检查通过。验证覆盖真实小模型 LoRA 反向传播与手工目标对照、旧 fine 轨迹信用重算、新信用采样接入、原生轨迹重放、checkpoint/tensor 身份、故障续跑、累计预算和成对提前停止。

308 个训练/开发任务与错误起点组合均存在 12 步预算内的专家解法，最长 11 步。真实 Qwen tokenizer 检查得到最长起始提示 1,591 tokens、最长专家上下文 1,968 tokens、最长专家动作 142 tokens，满足 8,192/256 的上下文/动作上限。

以上为 CPU 本地验证与已记录原生轨迹验证结果。L40S 训练和本轮模型效果由云端作业产生，返回包将用于逐轨迹重放与对照分析。
