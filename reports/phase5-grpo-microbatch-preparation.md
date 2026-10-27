# Phase 5：GRPO 微批并行与云端对照准备

日期：2026-10-27。当前状态：实现与 CPU 验证完成，真实模型的 L20 GPU 对照待运行。仍属于 Phase 5；本轮承接“每次只处理一条回答是否充分利用 GPU”的讨论。

## 本轮改变

新增可一次前向多条回答的完整序列评分，以及每个微批一次反向的梯度累积。保留原串行实现作为比较基准，没有修改此前云端运行绑定的源码。旧 reward、TIS、reference 惩罚、更新停止阈值均沿用。

有效 batch 仍然是 4 组 × 每组 8 条回答 = 32 条；只改变同时处理的回答数量。原串行方式需要 32 次前向/反向，微批 4 需要 8 次，微批 8 需要 4 次，然后都只执行一次 optimizer.step。

每条回答的平均损失乘以其 response token 数 / 整个有效 batch 的 response token 总数，再在微批内求和、反向。这保留全 batch 的 token 平均目标，而不是把各回答或各微批等权平均。本次固定输入有 915 个 response token，包含 EOS。

序列使用右侧 padding；attention mask 根据真实长度生成，因此 EOS 与 PAD 同 ID 时，真实 EOS 仍参与注意力与评分。每行只提取 response 对应的因果预测位置；prompt 与 padding 不进入 response 损失分母。多条回答共享前向图，所以每个微批只反向一次。

## 云端实验约束

输入绑定第一次真实 GRPO 更新的最终 checkpoint，第 16 步，summary SHA256 为 `1816932fd3df9a5aa6bb3e18c402cb4775378d50c78c2311c1910950194213ad`。选择原计划首个 batch 的同 4 组、同 32 条存储回答，不引入开发集。候选权重、Adam 一阶/二阶矩与 step 计数在每次探针前恢复为相同起点，reference 仍是原始冻结 SFT。

这批旧 rollout 仅用于固定工作负载的执行方式比较，不是刷新后的第二轮策略训练。所有探针更新均被丢弃，不导出新候选 adapter，不晋升 baseline，也没有生成新回答。

比较串行 1 与并行 4、8：

- 原始 response token logp 最大绝对误差 ≤ 1e-5；有效 batch loss 绝对误差 ≤ 1e-6。
- 裁剪前 LoRA 梯度最大绝对误差 ≤ 1e-6，整体相对 L2 误差 ≤ 1e-4。
- 一次临时更新后的 adapter 最大误差 ≤ 1e-6，Adam 状态最大误差 ≤ 1e-7，step 必须精确为 17。
- 停止检查、有限非零梯度、梯度裁剪仍执行。原先跨引擎概率门的失败状态没有改变，本次不通过放宽门限来制造通过。

三次正确性探针同时预热三种方式。随后每种方式计时三次，顺序轮换为 1/4/8、4/8/1、8/1/4；GPU 同步后计时，排除 checkpoint 恢复和 CPU 参数快照，包含真实损失、反向、停止检查、梯度裁剪与 Adam 更新。记录耗时、response token/s、峰值已分配显存；显存指标包含常驻模型与 reference，不是纯激活显存。

共执行并丢弃 12 次临时 optimizer 更新，最后恢复候选权重并检查完整参数摘要、冻结 base、reference、输入文件哈希。只有数值全部通过且计时中位数快于串行的微批才进入推荐；否则保留串行。该结果只适用于这一个固定 batch 和当次 GPU，三次计时不是跨任务统计结论。

## 本地验证

`python -m pytest tests -q`：327 项通过，1 条既有 tiny PEFT 保存警告。新增 22 项测试覆盖不等长 prompt/response、EOS=PAD、原始评分对齐、各微批梯度一致、最后不足一批、停止时不更新，以及完整微批基准流程的 12 次探针和恢复。完整流程使用 CPU tiny 模型及模拟 CUDA 计时接口，不构成真实 GPU 速度证据。

最初未限定目录运行 pytest，错误收集了 baseline 内的历史测试快照，发生模块重名收集失败；改用项目既定 `tests` 目录后完整通过，未改动历史快照。

CPU 输入预检通过：46 项首次训练产物哈希、26 项绑定源码哈希、原始 74 个 baseline 快照与 8 个 base cache 文件，以及 144×8 rollout/评分/掩码关联均核对。预检在本地用 baseline 哈希检查替代 Linux cloud bundle 检查，并模拟 L20 设备身份，仅验证输入契约；没有加载预训练模型、执行 CUDA 或验证云端依赖。记录保存在 `artifacts/evaluations/grpo-microbatch-preparation/cpu_input_preflight.json`。

## 云端执行

将 `grpo-microbatch-update-20261027.zip` 上传到 `/root/rivermind-data`，在当前 L20 实例终端执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e grpo-microbatch-update-20261027.zip .
cd AgenticPostTrainingLab
python scripts/benchmark_grpo_microbatch.py --hourly-price 2.18
```

ZIP 只包含本轮新增文件，不覆盖原先被哈希绑定的源码。沿用现有环境，不重新安装依赖。真实执行会恢复原始 cloud bundle/依赖/GPU 身份检查；需保留此前 first-update 和 Phase 5 rollout 目录。

完成后在项目根目录下载 `grpo-microbatch-review-20261027.zip`。详细结果目录为 `artifacts/evaluations/grpo-microbatch-20261027-01`。脚本拒绝覆盖同名输出，不自动重试；失败时保留已有逐探针文件并打包。最长期限取 30 分钟与按填写时价估计 10 元预算所允许时间的较小值；只约束作业，不包含开机、空闲、存储、计费取整。脚本不会自动关机。

## 代码阅读顺序

1. `configs/grpo_microbatch_benchmark.json`：先区分有效 batch 32、微批 1/4/8，以及数值门和零发布更新约束。
2. `src/agentic_post_training/rollouts/batched_scores.py` → `score_responses`：右 padding、按长度 attention mask、`len(prompt)-1` 的因果评分切片。
3. `src/agentic_post_training/trainers/grpo_microbatch.py` → `accumulate_microbatches`：response 长度 / 全 batch token 数的权重；共享前向图只反向一次。
4. 同文件 → `guarded_microbatch_step`：一次有效 batch 对应一次更新，沿用原停止与梯度裁剪。
5. `scripts/benchmark_grpo_microbatch.py` → `load_inputs`、`benchmark` 内 `reset`：绑定候选、原始数据与完整 Adam 状态；每个探针独立从第 16 步开始。
6. 同文件 → `gradient_difference`、`numerical_gate`、`recommend`：绝对/相对梯度误差、参数和 Adam 状态检查，以及有条件速度推荐。
7. `tests/integration/test_grpo_microbatch.py`、`tests/unit/test_grpo_microbatch_benchmark.py`：先读评分与梯度等价测试，再读完整探针恢复流程及失败回退测试。

GPU 结果回来后先审核数值与吞吐，再决定后续训练使用的微批。下一次真正策略迭代仍需要从当前候选重新采样、绑定新的 old-policy 版本，并保留原 SFT reference 与开发集对照。
