# AgenticPostTrainingLab — V1 History and Archive
更新：2026-10-30。本文供追溯历史实验；当前项目入口为 [V2 Handoff](PROJECT_HANDOFF_V2_2026-10-30.md)。

## 1. V1 的定位与最终状态

V1 是基于 HuggingFaceTB/SmolLM2-135M-Instruct 的后训练学习与实验阶段，完成手写损失、真实 LoRA SFT/DPO/GRPO、采样概率契约、检查点恢复和诊断对照。任务以算术、严格 JSON 工具调用、括号/运算语义为主，尚未形成当前 V2 的多轮业务 Agent 闭环。

SFT 带来了明显的结构化输出提升；后续 DPO/GRPO 在固定开发评测上的增益有限。最后的平衡示范 + GRPO 得到 1 个开发样本的改善，仍未升级冻结基线。用户随后决定封存 V1、转向 1.5B 和 Agentic RL，V1 不再默认继续训练。

这里的旧命名 sft-v2、expanded_v2、dpo_v2 都是 **135M 阶段的版本号**，与项目目录 v2 下的 1.5B 新架构不同。模型容量是用户提出的可能瓶颈之一，旧实验没有通过仅更换参数量的控制对照确认这一因果关系。

## 2. 归档、路径与恢复依据

项目根：
C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab。

归档：
[AgenticPostTrainingLab-v1-20261029.zip](C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029.zip)

凭据：
[Receipt](C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029-receipt.json)；
[Manifest](C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029-manifest.json)。

- ZIP SHA-256：6348f933a916f976eb16670caff132a54cbf09710e99a3a7d86e2a73dfbec500。
- ZIP 327,327,114 bytes；2015 个原始文件，未压缩 514,282,926 bytes。
- 包含源码、配置、数据、报告、实验产物、adapter、optimizer、旧模型缓存及 Git 元信息记录。
- 排除 .venv、.git 对象目录、pycache、环境秘密文件、临时缓存及整个 v2。
- 创建时 ZIP CRC 与成员 SHA-256 验证通过，原文件保留；本交接时重新核对 ZIP 哈希及全部 2015 个原文件，均一致。
- 归档时 Git HEAD：eda30590b021798edf8c6209f732917aabeeb38e。大量后期文件尚未提交，但已收入归档；恢复不能只依赖 git checkout。

原始文档保留于 [2026-10-27 Handoff](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/PROJECT_HANDOFF_2026-10-27.md)。其当时的下一步现多已完成，查状态应优先用本文及具体运行 summary。原 Desktop 路径已不存在。

## 3. 旧 Phase 与已完成工作

| 旧 Phase | 内容 | 最终情况 |
|---|---|---|
| Phase 0–1 | 概念、数据与手写 SFT/DPO/GRPO 损失 | 已完成基础实现与测试 |
| Phase 2 | 真实模型 LoRA SFT、数据扩充、工具格式/括号评测 | 完成，有结构化输出收益 |
| Phase 3 | DPO、偏好诊断、chosen-only/真实 rejected 控制实验 | 完成，最终统一开发集未见额外增益 |
| Phase 4 | vLLM rollout、概率对齐、校正、重放与性能实验 | 工程契约完成；严格等价未过的 microbatch 未替换串行路径 |
| Phase 5 | 真 GRPO 更新、更新后刷新采样、分级奖励、示范热启动 | 多轮有限实验完成；最终未晋升新基线 |
| 旧 Phase 6–7 方向 | 完整 Agent、扩展场景与规模 | 由 V2 新架构承接，不能记作 V1 已实现 |

模型固定 revision：
12fd25f77366fa6b3b4b768ec3050bf629380bac。

LoRA：q_proj/v_proj，r=4、alpha=8、dropout=0，共 230,400 参数、120 个训练张量。

冻结基线位于 artifacts/baselines/sft-v2-phase3；原 SFT 运行 artifacts/checkpoints/sft-lora-20261008-202445-a3d10354。完整策略参数摘要：
9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb。

后续 DPO/GRPO 都是候选分支，没有覆盖此冻结基线。

## 4. 结果脉络

### 4.1 SFT 与 DPO

SFT 数据 expanded_v1 为 train 120/dev 24，expanded_v2 为 train 144/dev 32。严格工具调用评测由 base 0/20 到 SFT 19/20；v2 括号评测 33/48，空格扰动 12/12。0/20 反映严格协议通过率，不是底座模型算术能力为零。

SFT 数据与训练量曾同时变化，因此版本差异应按整个实验配置描述。

DPO 从 28 对/28 步开始，后续扩大数据、加入 chosen-only SFT、SFT anchor 和真实 rejected 控制。出现过偏好 loss 改善而生成质量下降。

最终匹配实验使用相同 38 个 prompt、38 步、lr=1e-5，对比 chosen-only SFT / synthetic rejected DPO / real rejected DPO；共同 56 个开发题 strict 均为 36、semantic 均为 44，未获得额外开发收益，并观察到个别格式退化，未晋升基线。

### 4.2 采样、概率契约与真 GRPO

vLLM 负责生成，PyTorch/HF/PEFT 负责训练；没有使用 TRL 或 SGLang 管理训练。

工程上区分：

- q：真实采样分布。
- p_old：更新前策略在训练侧重算概率。
- p_new：更新中的策略。
- p_ref：冻结 SFT reference。

跨引擎严格 logprob 等价未通过后，采用训练侧同路径 FP32 old/new 契约，并显式处理 q 与 p_old 差异。使用 token TIS cap=2、PPO clip=0.2、sampled k3 beta=0.02。这些是采样/截断估计，不能把 q 偷换成 p_old。

microbatch=8 在固定 32 回答更新工作负载上测得 6.46×，但 logprob 最大差约 5.72e-5，高于 1e-5 门槛，因此正式更新保持串行。该数字只属于这项受限 benchmark。

| 实验 | 训练信号与更新 | 主要结果 |
|---|---|---|
| 首次 GRPO | 144×8=1152 回答；62 mixed groups，496 回答；16 logical steps | dev strict 36/56 不变 |
| 更新后重新采样 | 从首轮候选刷新 144×8；59 mixed，472 回答；再 15 logical steps | train strict 28→30/48；dev 36/56 不变 |
| 采样探索 | 24 题×3 温度×32=2304 回答 | 增大采样/温度未解决自然表达难题，高温格式更差 |
| 分级奖励离线重评分 | 4920 条已有记录，0/.1/.3/1 | 揭示严格匹配与语义正确的差异；此步无训练 |
| 同源奖励对照 | 同一 31-step 父模型、同一 768 回答；两分支各 6 logical steps | equivalent binary 无提升；graded train strict 30→29/48；dev strict 36、verified 44 不变 |

首次 16 logical steps 中实际 optimizer 调用 17 次，包含恢复一致性探针中随后丢弃的一次；不要把它计为 17 个保留训练步。刷新轮按用户选择延续首轮 Adam，16→31；reference 仍为冻结 SFT，并非新的 old policy。

31-step 父候选参数摘要：
70804ab1c9344f7578396e4ccaa90dfe1ea374c69f96cb69ee3970d036eedf7d。

分级奖励接受经过验证的加法/乘法子树交换与正确数值，不是任意代数等价。旧 strict、等价 binary、graded 必须分开报告。相同 dev 输出由 strict 36 到 verified 44 是评分口径变化，不是训练提升。对全错组赋予部分分会制造方差，也可能强化更完整的错误，因此随后引入正确示范。

### 4.3 24 个实际训练题的补充评测——已经完成

旧 reward comparison 报告里“待补实际训练题前后评测”已经落地。CPU 上对 parent、equivalent_binary、graded 各做 24 条 greedy，共 72 条。

三者均 strict 6/24、verified 13/24、graded mean 0.6，逐题输出没有变化。21 个与云端重叠的 anchor 全部匹配，权重前后不变、0 次 backward、0 次 optimizer update。

teacher-forced 概率出现小变化，但没有使所检查错误样本的关键正确 token 成为首选。该补评是统一 CPU 条件下的比较，anchor 匹配只覆盖那 21 项。

证据：
[Supplementary Summary](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/grpo-training-case-evaluation-20261028-02/summary.json)。

### 4.4 正确示范热启动与最终平衡分支

第一版：24 条 chosen、18 SFT steps。原 24 题 verified 13→14，但 train48 38→35、dev56 44→39，出现 8 个必要括号回归。行为门槛正确中止，未启动后续采样/GRPO。用户看到的 warm start behavior safety gate failed 是该实验结论，不是单纯环境故障。

调整为 36 条示范，增加 12 条 left_grouped；两遍共 72 次样本曝光、18 SFT steps，lr=5e-5。定义 A=原 31-step 候选，B=平衡 SFT，C=B 后再 5-step GRPO：

| 指标 | A | B | C |
|---|---:|---:|---:|
| 原 24 题 verified | 13/24 | 15/24 | 15/24 |
| 原 24 题 strict | 6/24 | 7/24 | 7/24 |
| train48 verified | 38/48 | 37/48 | 37/48 |
| train48 strict | 30/48 | 29/48 | 29/48 |
| dev56 verified | 44/56 | 44/56 | 45/56 |
| dev56 strict | 36/56 | 36/56 | 37/56 |
| 新增 12 题 verified | 11/12 | 11/12 | 11/12 |

旧 8 个括号回归全部修复；新增单个 train 错误：14*55+26 被写为 14*(55+26)。

从 B 刷新 24×32=768 回答，其中 verified 381，17 mixed groups、7 全错组；执行 5 logical GRPO steps。SFT 使用独立优化过程；GRPO 继承原父模型的 31-step Adam，最终为 36，不是把 18 个 SFT step 加进 Adam GRPO 计数。

自然表达难题 8 个 greedy 仍为 0/8，B 对应采样仅 1/256 正确。C 尚未再刷新采样，随后用户改选 V2。最终 baseline_promoted=false。

B 参数摘要：b9ba137848deb1fd653d140a764bc31f2393637f20dcc0776f3b403abc2b5223。
C 参数摘要：86e10d229e457ed8d6aa17151e2f1828702e804a86723aff721657780ebffa41。

## 5. 关键经验如何进入 V2

1. **先掌握任务协议再做 RL。** V1 SFT 对格式有效；V2 baseline 也暴露协议和证据链问题，因此下一步先 SFT。
2. **评分器与学习收益分开。** 更合理的等价判断能改善测量，却不会自动改变策略；必须在统一规则下做前后评测。
3. **正确轨迹覆盖很关键。** 全错组的相对分级可能只偏好局部完整错误；示范需覆盖容易退化的结构，并按真实任务分布平衡。
4. **偏好/概率变化不是终点。** 同时跟踪实际 greedy 成功、随机成功率、格式、恢复、成本和逐题变化。
5. **保持身份和可重放。** 输入摘要、模型/adapter/optimizer 哈希、冻结 reference、原生 token、概率语义、逻辑步数与真实调用数要明确。
6. **扩大场景和能力。** V2 已换 1.5B，并以实际多轮 SQLite 对账为起点；逐轮可验证进度比仅对最终算术字符串打分更贴合 Agentic RL。

V1 可作为简历中的工程闭环、控制实验与诊断经历。效果数字应标明任务、口径和对照；V2 的 SFT/DPO/GRPO 效果将在新任务上重新验证。

## 6. 按需追溯索引

以下相对路径均从项目根目录出发，详细工作流无需开局全部读取。

| 内容 | 主要证据 |
|---|---|
| 原始规划和早期 SFT/DPO | PROJECT_HANDOFF.md；PROJECT_HANDOFF_2026-10-18.md；PROJECT_HANDOFF_2026-10-27.md |
| 更新后刷新与 Adam 延续 | reports/phase5-policy-iteration-results.md；artifacts/checkpoints/grpo-iteration-20261028-01 |
| 分级奖励定义/离线结果 | reports/phase5-graded-reward-rescore.md |
| 同源奖励分支比较 | reports/phase5-reward-comparison-results.md；artifacts/checkpoints/reward-comparison-20261028-01 |
| 实际 24 题补充评测 | artifacts/evaluations/grpo-training-case-evaluation-20261028-02 |
| 第一次示范回归与中止 | artifacts/evaluations/grpo-demonstration-warmstart-review-20261028 |
| 最终平衡示范及 GRPO | reports/phase5-balanced-warmstart-results.md；artifacts/evaluations/grpo-balanced-warmstart-review-20261029 |
| 最初 V2 提议 | reports/project-v2-architecture-proposal.md；具体选择以新 V2 Handoff 为准 |

demonstration/balanced review 目录中含 input.zip、extracted、local_validation、CPU 权重/优化器校验和逐题变化等证据。不要假定所有回传 checkpoint 都直接放在根 artifacts/checkpoints 下，先按对应 review 的 extracted 目录寻找。

可直接阅读：[Policy Iteration](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase5-policy-iteration-results.md)、[Reward Comparison](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase5-reward-comparison-results.md)、[Balanced Warmstart](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase5-balanced-warmstart-results.md)。

## 7. 接续边界

V1 归档是历史依据，V2 是继续开发位置。不要把 V1 adapter/Adam/rollout 缓存接到 1.5B，也不要因旧文档遗留待办而重新启动 V1。用户当前要求整理交接，没有要求新增训练或发布简历结论。

V2 的完整架构、阶段安排、最新 baseline、下一步 SFT、模型缓存和云端命令约定均见 [V2 Handoff](PROJECT_HANDOFF_V2_2026-10-30.md)。
