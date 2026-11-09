# Phase 5：候选策略刷新与第二轮有限更新准备

日期：2026-10-28。新增入口、版本契约、CPU 测试和真实输入预检已完成；新一轮 vLLM 采样与 GPU 更新尚未执行。本报告不预判学习收益。

## 本轮变化与固定身份

首轮 GRPO 的开发严格成功为 36/56 → 36/56，保留该结果。本轮从其最终候选生成新回答，随后建立新的 old 分数；reference 继续固定原始 SFT。用户已明确选择延续首轮 Adam 动量。

| 对象 | 固定来源 |
| --- | --- |
| base | SmolLM2-135M-Instruct，revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` |
| sampler / 本轮 old / 初始 new | `artifacts/checkpoints/grpo-first-20261027-01/final-checkpoint/adapter` |
| 候选完整参数摘要 | `853c4027c91b968faadba67d46e3e9f2f239f107ab7bf8e58f276b0a643fe55b` |
| 候选 adapter 文件哈希 | `fd2ae12c4c1b1a5fd79d5ff7ed64a134967e83e226aeaf36631dc9fda1756913` |
| 固定 reference / tokenizer | `artifacts/baselines/sft-v2-phase3/checkpoint` |
| reference 完整参数摘要 | `9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb` |
| 父 Adam 文件哈希 | `3ea65d8f0beaba4836e550a901be7bfabfa9bb069eca3d9cc75742121022a17b` |

`verify_parent` 核对父 summary、其 46 个绑定文件以及候选/optimizer 独立身份。`setup` 校验冻结 baseline 的 74 个快照和 8 个 base 缓存文件、revision、数据与 train/dev 隔离。正式入口固定 NVIDIA L20 及既有依赖版本；CPU 预检不冒充 GPU 运行检查。

增量包只添加新文件。包内清单还记录运行依赖的既有文件哈希，统一入口先验证这些文件，云端旧文件缺失或漂移时停止。旧云端清单、源码、数据与实验门限不改写。

## 预先固定的采样与更新范围

新配置为 `configs/grpo_policy_iteration.json`。144 个训练数学输入及 48 个训练诊断、56 个开发输入 ID 沿用先前配置；每个训练数学输入新采 8 条，共 1152 条，全部保留。开发题不用于采样或选择组。

采样 seed 为 `202610280 + group_index * 1000 + sample_index`，1152 个 seed 唯一且与旧 144×8 批次无交集。最初接手方案中的 `20261028` 会与旧基数 `20261021` 部分重叠，已经在执行配置中修正。优化阶段 seed 单独为 20261028。

采样沿用 T=1、top_p=1、top_k=-1、无惩罚、最多 64 个回答 token、context512；vLLM 原生 LoRA、FP16、raw_logprobs、并发4，配置与运行时实际值核对。训练继续 HF 原始 FP32/eager、eval/dropout0、关闭 TF32，使用串行评分与反向。

新采样按原生 token 重新评分和严格 reward 重审。初始 new 仅与本轮 old 要求 ≤1e-5；不再要求 reference 与 old 相等。q、old、new、reference 的定义和 token TIS/sample k3 近似目标保持不变。

混合奖励组由新样本决定。事先固定按 case ID 顺序处理所有混合组，每组只用一次、每批最多4组，最多36个逻辑步；不复用旧62组列表，不按结果挑回答。零混合组则记录 `no_signal`，不自动增加采样。停止门在本轮第一步即触发时记录 `stopped_before_update`。这两种情况仍保存最终状态和行为比较，不声称发生了更新。

沿用 lr=1e-5、beta=0.02、PPO clip=0.2、TIS cap=2、梯度 clip=1。sampled k3>0.05 或 ratio 超区间比例>0.5 时清空当前梯度并停止后续更新；非有限值/零梯度报错。检查在更新前执行，不是更新后的回滚。

## Adam 来源与轮内恢复

`import_parent_adam` 使用 `weights_only=True` 读取父状态，核对文件哈希、旧 input digest、父逻辑步16、可训练参数名/顺序、Adam 超参数、所有动量 shape/dtype/有限性和每项 Adam step=16。导入后比较动量，要求误差为0。父文件和旧绑定保持原样。

新 checkpoint digest 联合绑定候选/reference、配置、源码/数据、实际新 rollout summary/manifest 和按规则确定的组批次。`optimizer_lineage.json` 显式记录旧绑定到新绑定的关系。跨轮导入不恢复父 RNG；本轮采用独立 seed。轮内 checkpoint 则保存并恢复本轮 CPU/CUDA RNG。

本轮逻辑第1步完成后 Adam 累计 step=17。若至少完成两步，保留一次不中断第2步探针，然后从本轮第1步恢复并重放第2步：adapter 误差≤1e-6、Adam误差≤1e-7、loss差≤1e-6。探针分支丢弃，实际 optimizer/backward 调用单独计数。

最终 checkpoint 的 `step` 表示本轮逻辑步，Adam buffer 的 `step` 表示从首轮累计的步数，summary 同时报告二者。少于两次成功更新时，恢复探针标为不适用及原因，不虚报通过。最终在新 base 上重载候选，核对完整参数摘要、8条 logp 和8题 greedy token。

## 行为评测与验收

原 SFT、上一候选、新候选在同一 GPU、相同模板和 FP32/eager greedy 条件下各生成 48+56 条，共312条预测。恢复/重载探针另计。reference 分数另存 `reference_scores.pt`，支持检查 old/reference 的不同身份。

报告三组逐题比较：SFT→上一候选、上一候选→新候选、SFT→新候选。严格成功要求 EOS；列出新增成功、退步、JSON/schema/route 退化和 length 次数。工程通过、行为安全和严格净收益分别记录。行为安全要求相对 SFT 和上一候选均无新增结构退化、开发严格总分不降；安全通过仍不能替代收益证据。`baseline_promoted` 始终 false。

开发集已暴露，训练诊断属于既有训练输入；本轮不产生最终盲测、多 seed 稳定性或长期在线循环结论。token TIS 与 sampled k3 的原有近似限制继续保留。

## 本地验证范围

新增17项测试覆盖：新旧 seed 隔离、混合组顺序/零信号、输入绑定变化、不安全/漂移文件拒绝；原生记录与 reward 损坏拒绝；三方逐题比较；采样失败阻断训练、失败仍打包及拒绝重复运行。

真实 tiny Llama+PEFT 测试构造首轮16步非零 Adam 状态和固定 reference，验证新 old/new 一致而 reference 不同、继承动量后真实更新、轮内恢复下一步参数/动量完全一致、累计 step=18，以及错误哈希/绑定/参数顺序/步数/shape/NaN/超参数在导入前拒绝。完整新采样审计入口使用1152条合成原生记录测试 reward/分数/mask/summary 重算，属于接口测试，不是实际新 rollout。

全套测试当前为344项；最终执行结果以本次交付的验证记录为准。只有一条已有 tiny PEFT 提示。

真实135M输入预检核对144题、48+56评测、长度上限；CPU加载父候选并独立重建完整参数摘要，确认120个可训练张量/230400参数，Adam导入误差0。没有模型前向、采样、反向或optimizer更新，也没有GPU依赖验证。归档在 `artifacts/evaluations/grpo-iteration-preflight/`。

## 云端执行与结果取回

前提：已有 L20 实例保存冻结 baseline/base 缓存、首轮 GRPO 完整产物，以及此前增量源码；新包不包含模型和旧产物。不在云环境执行 `pip install -e .`。

上传 `grpo-policy-iteration-20261028.zip` 到 `/root/rivermind-data` 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e grpo-policy-iteration-20261028.zip .
cd AgenticPostTrainingLab
python scripts/run_grpo_iteration.py --hourly-price 2.18
```

时价变化则填写当时实际值。统一入口顺序运行新采样/评分/审计，再运行有限更新/评测；采样失败不会启动训练。每个阶段组分别最多1800秒/估算10元，预留清理时间；两阶段合计边界约60分钟/20元，额外的CPU预检、打包、开机/空闲/存储/计费取整不计入该估算。无自动重试、扩轮或自动关机。

新目录为 `artifacts/trajectories/grpo-refresh-20261028-01` 和 `artifacts/checkpoints/grpo-iteration-20261028-01`。任一目录或本轮归档已存在则拒绝重新运行。失败时先取回证据，不直接重复命令或删除旧目录。

统一下载包放在项目根目录：

```text
/root/rivermind-data/AgenticPostTrainingLab/grpo-policy-iteration-review-20261028.zip
```

包内保留 `trajectories/`、`checkpoints/`、`evaluations/` 层次；若失败，仅包含实际产生的目录和错误记录。运行成功时也保留两个阶段独立ZIP。下载统一包并回传后复核原始记录、源码/输入身份、状态与行为；空闲GPU由用户在平台处理。

## 代码阅读顺序

1. `configs/grpo_policy_iteration.json`：先读父候选/reference身份、seed、输入ID、Adam模式、最大36步和停止门；混合组列表待新样本按既定规则产生。
2. `src/agentic_post_training/trainers/policy_iteration.py`：`verify_parent` 绑定首轮候选；`select_mixed_batches` 固定选择规则；`import_parent_adam` 区分跨轮动量导入与轮内恢复，重点看两个 input digest。
3. `scripts/iteration_common.py`：`setup` 检查模型/数据/训练开发隔离；`identity` 记录共同来源；`validate_records` 验证新 seed、原生 token、解码、EOS及reward。
4. `scripts/collect_grpo_iteration.py`：`generate` 从候选而非SFT采样；`score` 从相同候选计算old/new；`audit` 重算reward、mask和summary。`write_summary=False` 为训练前只读重审路径。
5. `scripts/train_grpo_iteration.py`：`context` 接纳通过重审的新rollout；`train` 分别加载候选policy和SFT reference，建立新绑定、继承Adam、执行恢复探针和最终重载；`compare_three` 逐题列出改善和退步。
6. `scripts/run_grpo_iteration.py`：`preflight` 的CPU证据边界；`dispatch` 的先采样后训练、失败阻断与统一归档。最后对照三个新测试文件，先看真实Adam继承/恢复，再看新rollout审计与拒绝边界。

现有 `trainers/grpo.py` 的目标、串行累积和停止逻辑直接复用；本轮没有启用微批路径、修改旧门限或接入完整多轮工具Agent。
