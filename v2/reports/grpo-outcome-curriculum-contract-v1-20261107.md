# GRPO Outcome Curriculum Round 1 — Frozen Experiment Contract

本轮进入纯 outcome GRPO 优化：从最佳加权 SFT（W）分别训练两支模型，使用相同的扩充任务、恢复场景、组数和随机种子，比较学习率 1e-6 与 1e-5。目标是提高完整任务和出错后的恢复成功率。

用户已确认本轮总预算为 6 小时（21600 秒）。两支各 48 组，完整作业包含 W 基线、训练、最终评测和审计。三份配置已冻结，云端作业由用户在 L40S 实例启动。报告只提供链接。

项目内报告相对路径：reports/grpo-outcome-curriculum-contract-v1-20261107.md

云端报告绝对路径：/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-outcome-curriculum-contract-v1-20261107.md

**本轮针对的问题**

C4 中，W、outcome RL16、fine RL17 的采样完整任务成功数分别为 28/48、30/48、29/48，采样恢复成功数为 171/252、175/252、171/252，greedy 恢复成功数为 61/84、58/84、58/84。两支 RL 均未通过冻结验收。

本轮重点训练三类恢复：SQL 得到数值但遗漏指标 alias；连续使用错误表名或列名；重查月度数据后仍提交错误差额或错误证据。旧 C4 任务用于问题分析，新训练与开发任务使用独立身份和随机种子。

**双分支与训练边界**

| 项目 | control | lr10x |
|---|---|---|
| 初始化 | W，独立 AdamW | W，独立 AdamW |
| 学习率 | 1e-6 | 1e-5 |
| 训练目标 | outcome GRPO | outcome GRPO |
| 参考策略 | 冻结 W | 冻结 W |
| 组内采样数 | K=4 | K=4 |
| 正常/恢复组比例 | 1:1 | 1:1 |
| 每组最多更新次数 | 1 | 1 |
| LoRA | q_proj/v_proj，r=8，alpha=16，dropout=0 | 相同 |

两支对照的配置差异限于运行标识与学习率。奖励沿用任务成功奖励减去完整轨迹动作成本，每步成本 0.01；恢复前缀计入任务总动作成本，梯度仅作用于模型继续生成的动作。零方差组直接跳过，保留所有组和跳过原因。

beta=0.02，PPO clip=0.2，TIS cap=2，梯度范数上限 1，sampled K3 上限 0.05，clip fraction 上限 0.5。训练采样 temperature=1、top_p=1、top_k=-1；每次响应最多 256 tokens，上下文最多 8192 tokens；任务动作上限为 12。

本轮新增 SFT 更新数为 0，新增 DPO 更新数为 0。W 的历史来源包含 SFT 513 次和 DPO 11 次更新。新增恢复前缀由环境脚本构造，专家动作用于验证起点可恢复性；模型继续生成的动作由自身策略采样。数据课程、学习率、组数以及实际更新次数分别记录。

与 C2/C3 相比，本轮增加训练任务族和恢复起点，并按预算扩展组数；本轮两支之间的对照变量为学习率。

W adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f

基座：Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。

**已确认预算**

| 作业总上限 | 每支训练组数 | 活跃训练任务 | 两支训练轨迹 | 最终刷新轨迹 | 三模型评测轨迹 | 质量检查动作 |
|---|---:|---:|---:|---:|---:|---:|
| 6 小时（已确认） | 48 | 12 个旧任务 + 12 个新任务 | 384 | 8 | 468 | 42 |

总上限包括 W 基线、两支训练、最终评测、刷新检查和审计，预留 180 秒打包。共 384 条训练轨迹、8 条刷新轨迹、468 条开发评测轨迹，以及 42 次单步质量检查。总阶段进度条为 200 项；断点续跑沿用原 6 小时总预算。

C3 实测总耗时 11910 秒，约 3 小时 18 分；其中两支 24 组训练的收集与更新合计 5850 秒，三模型评测合计 5832 秒。本轮 48 组双分支增加训练组数与针对性恢复评测，预计约 5–6 小时。本轮使用用户新确认的 6 小时总上限。

**任务和评测**

完整档使用原训练集 12 个任务，加上 12 个新训练任务，交替排列；每个任务依次进行正常起点组和恢复起点组。新任务覆盖全部月份、跨年月份、更多地区，以及带/不带瞬时查询故障的情况。恢复组保留原 7 类场景，并加入上述 3 类问题；轮换场景安排，使新增问题同时出现在旧、新任务族中。

4 个开发任务使用新年份、地区组合、任务族、数据库种子和 segment。训练集、开发集与历史开发/测试任务、C4 的 12 个确认任务，在 task ID、family ID、seed 和 segment 上互不重叠。

每个模型的固定评测如下；比较 W、control 最终策略、lr10x 最终策略共三个模型。

| 评测部分 | 每模型数量 | 用途 |
|---|---:|---|
| 采样完整任务 | 16 | 主要结果 |
| greedy 完整任务 | 4 | 稳定性 |
| 采样恢复，原 7 类 | 84 | 主要结果 |
| greedy 恢复，原 7 类 | 28 | 稳定性 |
| 新增 3 类恢复，greedy/sample 各一次 | 24 | 针对性诊断 |
| 7 个固定起点，各重复 greedy 一次 | 14 个动作 | 重复性检查 |

标准评测为 132 条完整轨迹，针对性诊断为 24 条完整轨迹，共 156 条轨迹；加上 14 次单步质量检查，每模型进度条总数为 170。

开发集选优规则在运行前冻结：W 的采样完整任务若未饱和，以完整任务成功数为主要指标；若饱和，则以恢复成功数为主要指标。候选策略需主要指标高于 W，采样完整/恢复与 greedy 完整/恢复成功数均不低于 W。多个候选同时满足时依次比较主要指标增量、采样恢复、采样完整，完全相同则选择 control。新增 3 类恢复结果单列展示。下一次正式独立确认另行冻结新任务。

**运行与审计**

本轮在同一张 L40S 上交替训练两支。启动时绑定新实例 GPU UUID、驱动和软件版本；整个作业及续跑沿用该身份。每个收集/更新进程独立启动并释放显存。终端同时显示总阶段进度、组内 rollout 进度、打分/反向传播动作进度，以及 170 项模型评测进度。

保留逐组完整轨迹、原生 tokens 与行为概率、old/reference 分数、LoRA 权重、Adam/RNG 状态、真实更新与跳过原因、KL/clip/TIS 指标、环境回放、质量检查和配对成功/失败变化。Adam/RNG 的 state.pt 保留在云端，回包带哈希和状态往返验证结果。

单个分支失败时保留两支已经提交的组和检查点；续跑读取原配置与累计用时。成功或失败均生成 review ZIP 和日志。最终审计重新计算训练目标、权重变化、标准评测、针对性恢复和开发集选优结果。

**本地验证**

从运行 ZIP 解压得到的干净目录执行完整 GRPO 检查，148 项全部通过，包括历史回归 124 项和新增课程与调度检查 24 项。检查包括真实 SQLite 恢复、原生提示/响应回放、篡改拒绝、新旧任务隔离、两支交替执行、失败续跑和累计预算，以及旧 C2/C3/C4 默认行为。

完整档 24 个训练任务与 4 个开发任务，各检查正常起点及 10 类恢复，共 308 个专家路径，全部成功，最长 11 步。使用已经核对哈希的 Qwen 原生分词器测得：最长起始提示 1599 tokens，恢复过程最长提示 1981 tokens，最长单步专家响应 144 tokens。184 个 Python 文件语法检查通过，5 个相关 CLI 入口可用。运行包内 234 个绑定文件通过哈希核验，W 的 112 个 LoRA 张量完成 CPU 投影检查。

本地完成环境与 CPU 检查；L40S 上的真实策略采样、LoRA 更新和最终效果将在运行回包中审计。

**启动和回传**

运行包文件名：grpo-outcome-curriculum-v1-20261107-run.zip。包内以 v2/ 为顶层目录，解压到 /root/rivermind-data/AgenticPostTrainingLab，更新 v2/ 下同名源文件。W adapter、已审计的 GPU profile 和运行源码均随包提供；基座权重和已有 apt-vllm 环境沿用当前实例。

启动命令：

```bash
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_curriculum.py --config configs/grpo-outcome-curriculum-v1.json
```

中断后续跑，在同一实例、同一 v2 目录使用原命令追加 --resume。两支已提交的组和检查点会继续保留，累计预算保持原上限。

完成时终端打印 Download 路径，首轮回包为 /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-curriculum-v1-20261107-review.zip；续跑回包使用 review-attempt-002.zip 等后缀。回传该 review ZIP 后进行结果审计。

源码与输入逐文件绑定在 grpo_curriculum_release_manifest.json。发布 ZIP 的 SHA256 单列在发布收据，解压后的源文件以 manifest 核验。

**建议代码阅读顺序**

| 文件相对路径 | 大致作用 |
|---|---|
| src/apt_v2/tasks.py | 任务与订单退款数据、标准 SQL 和真值 |
| src/apt_v2/environment.py | 工具执行、证据验证和任务奖励 |
| src/apt_v2/grpo_curriculum.py | 扩充任务、恢复起点、训练顺序和专家校验 |
| src/apt_v2/grpo.py | 模型继续执行、轨迹回放和组内优势 |
| src/apt_v2/grpo_optimizer.py | token 概率、GRPO 损失和 LoRA 更新 |
| scripts/grpo_worker.py | 隔离的收集与更新工作进程 |
| scripts/grpo_curriculum_common.py | 预算档位、配对配置和开发集选优 |
| src/apt_v2/grpo_evaluation.py | 标准开发评测计划和配对汇总 |
| scripts/evaluate_v2_grpo.py | 三模型评测与原生轨迹核验 |
| scripts/run_v2_grpo_curriculum.py | 总入口、进度条、预算和断点续跑 |
| scripts/audit_v2_grpo_curriculum.py | 双分支训练与评测审计 |
| scripts/build_v2_grpo_curriculum_release.py | 冻结源码、打包与哈希检查 |

本轮运行标识为 grpo-outcome-curriculum-v1-20261107。冻结配置为 configs/grpo-outcome-curriculum-v1.json、configs/grpo-outcome-curriculum-control-v1.json、configs/grpo-outcome-curriculum-lr10x-v1.json。
