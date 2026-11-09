# C2 outcome GRPO 首轮正式结果与 L20 迁移评估

日期：2026-11-06。运行：grpo-outcome-formal-v1-20261105，attempt 2。

## 结论

首轮 outcome GRPO 已完成并通过云端与本地完整审计。24 个训练组产生 16 次真实更新，选定第 24 组节点对应的 RL16 为 C2 开发候选。完整成功率保持 16/16；主要观察指标 sample recovery 从 W 的 61/84（72.62%）提高到 63/84（75.00%），增加 2.38 个百分点。配对结果为 2 条改善、0 条退化，另外 61 条共同成功、21 条共同失败。

本轮训练和三次评测均在 L40S 上完成。用户确认实例克隆到 L20 发生在本轮结束之后，114 条逐阶段资源快照全部指向同一张 L40S。因此，本轮保留为完整的 L40S 实验记录；后续 L20 建立新的运行与资源记录。

## 开发结果

每个模型独立评测 132 条完整 episode，另有 7 个上下文×2 次 greedy 单动作质量检查。第 8 组节点实际更新数为 6，第 24 组节点为 16；所有分支从相同的冻结开发任务、场景与 seed 计划评测。

|指标|W / RL0|第 8 组 / RL6|第 24 组 / RL16|
|---|---:|---:|---:|
|sample complete|16/16（100.00%）|16/16（100.00%）|16/16（100.00%）|
|sample recovery|61/84（72.62%）|60/84（71.43%）|63/84（75.00%）|
|greedy complete|4/4（100.00%）|4/4（100.00%）|4/4（100.00%）|
|greedy recovery|19/28（67.86%）|20/28（71.43%）|20/28（71.43%）|

完整成功率饱和，按既定边界以恢复成功率作为主要观察指标。RL16 的 greedy recovery 同时由 19/28 提高到 20/28，增加 3.57 个百分点。开发候选排序为 RL16、RL6；数值验收阈值继续沿用暂不确定的约定，C4 按计划执行新任务最终确认。

|随机恢复场景|W|RL16|净成功数变化|
|---|---:|---:|---:|
|format_repair|12/12|12/12|0|
|unknown_tool_repair|12/12|12/12|0|
|sql_unverified_requery|6/12|6/12|0|
|repeat_evidence_error|7/12|7/12|0|
|stale_evidence_repair|12/12|12/12|0|
|repeat_sql_error|6/12|7/12|1|
|failed_check_repair|6/12|7/12|1|

三个成功状态变化均为失败→成功：

|模式|任务|场景|rollout|
|---|---|---|---:|
|sample|development-000|failed_check_repair|2|
|sample|development-002|repeat_sql_error|1|
|greedy|development-002|sql_unverified_requery|0|

## 训练与复核证据

返回包 SHA256：a1b657b7989f0bbb2384efcbc11c59644064eb8f420e46b55d422b597e0506dd。629 个 ZIP 条目的 CRC、路径、重复项与符号链接检查通过；188 个原始源文件、1 次 EOS 校验补丁及 22 项输入绑定来源一致。补丁安装记录证实云端最新 Adam/RNG 文件 SHA 校验通过，原始源码快照与修订快照均保留。

训练及刷新共 25 组、100 条 episode、675 个生成动作、56655 个响应 token，另有 25 个探针 token；最大 prompt 为 2855 token。24 个训练组中 16 组更新、8 组零方差跳过。所有 16 次更新分别改变 112 个 LoRA 张量，累计验证 1792 项变化；评分缓存、q/token 对齐、实际权重 delta、冻结 reference、Adam step 与 RNG roundtrip 均通过。

训练 25 次加载与三个评测分支均按实际 adapter 独立重建 BF16 packed LoRA 张量，与各 worker 的 112 个实际张量 SHA 匹配。独立复算 16 次 loss/TIS/k3 指标，最大差 5.315e−8；最大 sampled k3=3.4884e−6，clip fraction 始终为 0，最大 TIS cap fraction 为 0.000168691（0.0169%）。

三次评测共 396 条 episode、1922 个生成动作，以及 42 次质量动作。逐条严格环境重放、calendar 最终证据、固定评测覆盖、重复 greedy token 一致性及 worker 加载证明通过。本地重新执行完整正式审计，candidate ranking 与全部比较汇总和云端完全一致。

EOS 补丁修复后，group-0017 按原 checkpoint/seed 成功采样并提交；后续采样、更新、第 24 组评测、最终刷新及审计均正常结束。最终保存权重位于 checkpoints/group-0021，随后两个训练组为零方差，故第 24 组节点仍对应这份 RL16 权重。

候选 adapter SHA256：6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6。谱系为 SFT513 / DPO11 / RL16；起始 W SHA256 为 f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。本地已将候选 adapter 与模型身份、原配置、选择记录冻结保存，供后续比较使用。

## 代码阅读顺序

以下路径均相对 v2 项目根目录，按表中顺序阅读。

|文件相对路径|大致作用|
|---|---|
|`configs/grpo-outcome-formal-v1.json`|本轮训练、评测和资源预算配置。|
|`scripts/run_v2_grpo.py`|训练、评测、断点续跑和结果打包的总入口。|
|`scripts/grpo_common.py`|输入校验、运行环境绑定和 checkpoint 管理。|
|`src/apt_v2/tasks.py`|业务任务与数据划分。|
|`src/apt_v2/environment.py`|SQLite 工具环境、动作执行与成功判定。|
|`src/apt_v2/dpo_recovery.py`|恢复场景及起始上下文。|
|`src/apt_v2/grpo.py`|轨迹采集、重放校验、回报和组内优势计算。|
|`scripts/grpo_vllm.py`|vLLM 采样、行为概率和实际加载权重校验。|
|`src/apt_v2/grpo_optimizer.py`|GRPO 损失、评分缓存和梯度更新。|
|`scripts/grpo_worker.py`|执行单组采样、训练更新与状态保存。|
|`src/apt_v2/grpo_evaluation.py`|评测计划、指标汇总和配对比较。|
|`scripts/evaluate_v2_grpo.py`|运行 W 与 GRPO 的开发评测。|
|`scripts/audit_v2_grpo.py`|复核训练轨迹、更新记录和权重变化。|
|`scripts/audit_v2_grpo_formal.py`|汇总正式审计、候选排序与结果曲线。|
|`scripts/build_v2_grpo_release.py`|生成冻结的云端运行包。|
|`hotfixes/grpo-native-eos-v1/apply.py`|安装本轮 EOS 校验修复并保留已有运行状态。|

## 资源与预算

两次尝试累计 8528.2785 秒，即 2 小时 22 分 8 秒，首次正式作业在 4 小时上限内完成。当前正式作业使用 1/3，总记录耗时约 2.369 GPU 小时。

全部资源快照来自 NVIDIA L40S，UUID=92e14e85-628d-e5a4-af16-0db4defb2afb。Python 3.11.17、torch 2.8.0+cu128、CUDA 12.8、vLLM 0.11.0、transformers 4.57.6、peft 0.17.1。更新峰值 allocated 为 8.11 GiB，reserved 为 8.57 GiB；显式 activation checkpoint 覆盖 28 个 decoder 层。

## 改用 L20 的实验边界

L20 可以作为后续实验平台。克隆实例保留的是模型权重、数据、配置和运行文件；GPU 更换会改变计算与资源环境。具体处理如下：

|方面|影响与后续记录方式|
|---|---|
|本轮 L40S 结果|保留本次权重、成功率、审计与资源记录；迁移发生在本轮结束后|
|checkpoint 与数据|在 L20 校验 base revision、adapter、输入数据与配置 SHA256，沿用已冻结内容|
|训练与采样|浮点计算、kernel 与采样边界的差异可能改变后续轨迹；记录新 GPU、UUID、驱动及软件环境|
|W 与 GRPO 比较|在同一 L20 环境中，用相同协议重新评测 W 和 GRPO 候选，建立 L20 上的配对比较|
|效率与预算|L20 单独测量实际耗时、token 吞吐与显存；保留原作业 wall-time 上限和进度条|
|运行身份|后续使用新的 L20 配置/运行标识，保留已完成 L40S 运行的冻结记录|

vLLM 0.11.0 将可复现性的硬件范围限定为相同硬件与相同 vLLM 版本；因此，跨 L40S/L20 使用相同 seed 时，逐 token 轨迹也应重新记录和核验。[vLLM 0.11.0 reproducibility](https://docs.vllm.ai/en/v0.11.0/usage/reproducibility.html)

现有正式 v1 配置绑定 L40S 与本轮 GPU 身份。后续运行包将显式采用 L20 配置并绑定新环境。C2 outcome 候选已经保存；进入后续阶段时，先在 L20 建立 W/候选的共同评测基线，再衔接 C3 细粒度奖励分支与 C4 新任务确认。
