# Phase 5：L20 微批并行对照结果

日期：2026-10-27。结论：本次固定工作负载出现显著加速，但并行 4、8 均未通过预先声明的全部数值门；按原规则保留串行 1。本轮没有发布新的训练候选，没有证明行为指标提升。

## 输入与复核范围

收到 `grpo-microbatch-review-20261027.zip`，SHA256：`38246d77d02756238658821369582c4e05c4ed10500fbabd0c112d0b007d7ff1`。

本地已安全解包至 `artifacts/evaluations/grpo-microbatch-20261027-01`，复核 ZIP 完整性、45 项输出哈希、30 项源码哈希及其与当前工作区的一致性，并重新核验首次 GRPO 更新的 46 项输入产物。配置与预先提交的计划一致。

同一个有效 batch：4 组 × 8 回答，共 32 条、915 个 response token。起点是首次更新第 16 步候选，参数摘要 `853c4027c91b968faadba67d46e3e9f2f239f107ab7bf8e58f276b0a643fe55b`，reference 固定原始 SFT。所有模式复用相同回答、old/q/reference、优势与 Adam 起始状态。实验设备为 NVIDIA L20。

本地重算了各项门判定、计时中位数、加速比、token/s、损失分项关系、推荐结果和成本估算，并核对逐探针文件与汇总一致。记录：`artifacts/evaluations/grpo-microbatch-review/local_audit.json`。

本地未加载预训练模型或重跑 GPU 前向/反向。原始梯度与临时更新权重没有导出，所以梯度误差、参数误差与运行结束后的显存中状态恢复，是云端脚本的运行记录，不能表述为本地独立重演已证明。

## 吞吐与显存

| 一次处理回答数 | 前向/反向次数 | 一次有效 batch 耗时中位数 | 相对串行加速 | 峰值已分配显存 |
| --- | --- | --- | --- | --- |
| 1 | 32 | 3.4549 秒 | 1.00 倍 | 1.223 GiB |
| 4 | 8 | 0.9473 秒 | 3.65 倍 | 1.647 GiB |
| 8 | 4 | 0.5352 秒 | 6.46 倍 | 2.223 GiB |

每种方式计时三次，轮换执行顺序；每次都恢复起点。计时包含损失、反向、停止检查、梯度裁剪、Adam 更新，排除 checkpoint 恢复与 CPU 快照。显存包含常驻模型与 reference，并非纯激活显存，也不是整张卡的 nvidia-smi 占用。

这些结果支持“此前逐条处理限制了这批训练计算的吞吐”。6.46 倍仅对应本次 32 条回答的更新计算，不代表整个 GRPO 作业或其他模型、序列长度都会获得同样加速；采样、reference 评分、评估与存取时间不在这张吞吐表里。

## 数值门

所有误差均相对原串行实现，且从相同候选与 Adam 状态开始。

| 检查 | 门限 | 微批 4 | 微批 8 |
| --- | --- | --- | --- |
| token logp 最大绝对误差 | 1e-5 | 4.1485e-5，失败 | 5.7220e-5，失败 |
| loss 绝对误差 | 1e-6 | 7.7649e-8，通过 | 5.9084e-8，通过 |
| 梯度最大绝对误差 | 1e-6 | 2.2026e-7，通过 | 2.5332e-7，通过 |
| 梯度整体相对 L2 误差 | 1e-4 | 2.0867e-5，通过 | 2.5133e-5，通过 |
| 更新后 adapter 最大误差 | 1e-6 | 7.4506e-9，通过 | 7.4506e-9，通过 |
| Adam 状态最大误差 | 1e-7 | 2.1886e-8，通过 | 2.5146e-8，通过 |

临时 Adam step 精确为 17，所有探针均通过原停止检查。三次计时内，同一模式的损失与各项评分汇总精确重复。串行 loss 为 -0.0155393303；并行差异很小，但 logp 门仍必须按事前规则判为失败，没有修改门限或自动启用并行。

仅凭这些汇总不能确定误差来自哪一个 GPU 算子、批形状变化或 padding 执行分支，也不能直接断言实现存在语义错误。CPU 的不等长回答、mask、梯度测试通过，且本次其他五项门通过，支持它在这个 batch 上的更新结果非常接近串行；这仍不足以替代更广泛验证或证明长程训练等价。

本次是 HF 串行与 HF 并行之间的检查，与此前 vLLM sampler 对 HF trainer 的跨引擎偏差是不同对照；旧跨引擎失败状态保持不变。

## 更新、状态与费用

云端记录共 12 次临时 optimizer 更新，全部丢弃；新发布候选更新为 0，生成新回答为 0。运行记录表明最终恢复原候选参数摘要，冻结 base 与 reference 未变、reference 无梯度、输入文件哈希未变。没有输出新的 adapter，没有晋升 baseline。

完整作业 37.317 秒，按 2.18 元/时估计约 0.0226 元，无作业错误。该估计不含开机、空闲、存储、取整费用；脚本没有自动关机。

## 当前决定与下一步

维持原契约推荐 `microbatch_size=1`，不因看到加速而追溯放宽门限。本次吞吐探索已产出可用证据，不需要把它当作继续模型实验的无限阻塞项：下一轮真正的 Phase 5 策略实验可以继续采用现有串行路径，从当前候选生成新 rollout，绑定新的 old-policy 版本，reference 仍固定 SFT，并与同一开发集比较。

若之后单独推进并行训练，应预先定义统一的批处理评分路径，让同一版本的 old/new 使用相同批形状与 mask，再检查无更新 ratio、多个不同长度 batch、梯度与恢复一致性。那是新增的执行契约实验，不能据本轮单个 batch 的接近结果直接声称已获准或已完成。

本轮未修改训练源码，因此未重复此前通过的 327 项测试；完成的是新回传产物的哈希、数值摘要与记账一致性复核。

## 代码阅读顺序

1. `configs/grpo_microbatch_benchmark.json`：先看六项数值门与“全部通过且更快才推荐”的约束。
2. `scripts/benchmark_grpo_microbatch.py` → `load_inputs`、内部 `reset`：为什么同一批回答还要恢复同一候选和完整 Adam 状态。
3. `src/agentic_post_training/rollouts/batched_scores.py` → `score_responses`：右 padding、长度 attention mask、因果评分的 response 切片。
4. `src/agentic_post_training/trainers/grpo_microbatch.py` → `accumulate_microbatches`：一个微批一次反向，分母仍是完整 batch 的 915 个 response token。
5. `scripts/benchmark_grpo_microbatch.py` → `numerical_gate`、`recommend`：为什么并行已经更快且更新误差很小，最终推荐仍为 1。
6. `tests/integration/test_grpo_microbatch.py`、`tests/unit/test_grpo_microbatch_benchmark.py`：不等长评分/梯度、探针恢复与失败回退的覆盖范围。
7. `artifacts/evaluations/grpo-microbatch-20261027-01/correctness.json`、`timing.json`、`summary.json`，再读 `grpo-microbatch-review/local_audit.json`：区分云端运行证据和本地可重新计算的检查。
