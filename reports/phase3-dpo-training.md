# Phase 3 第 3 步：DPO 训练、保存与重载

日期：2026-10-09。

## 本步的训练目标

把已读过的 chosen/rejected 数据接入真实模型。policy 和 reference 都从固定 SmolLM2 base＋冻结的 v2 SFT adapter 开始；policy 继续更新 LoRA，reference 是另一份独立模型，所有参数 requires_grad=False、eval 模式，并在 no_grad 下计算回答分数。

使用 Phase 1 的 sigmoid DPO：

```text
policy_margin = log π(chosen | prompt) − log π(rejected | prompt)
reference_margin = log π_ref(chosen | prompt) − log π_ref(rejected | prompt)
relative_margin = beta × (policy_margin − reference_margin)
loss = −log sigmoid(relative_margin)
```

这里每条回答的 log probability 是有效回答 token 的 log probability 之和，不是 token 平均值。包括回答模板的结束区域，不包括 prompt 和 padding。模型前向不传 labels，而是将 input_ids 和 loss_mask 交给手写 sequence_log_probs，内部只做一次 causal shift，避免把 -100 用作 gather 索引。

reference 固定，数据固定，所以一次性缓存 train/development 中两条回答的分数。缓存 detached，不连接 reference 的梯度图。这是离线偏好优化，没有 rollout、在线采样、reward model 或 GRPO。本阶段先复用自己的损失与 PyTorch 循环，未接入 TRL。

## 首轮配置

`configs/dpo_cpu.json`：28 对训练偏好、10 对开发偏好，max_steps=28，batch 固定为每步一个偏好对（chosen/rejected 合并为两条序列前向），学习率 1e-4，beta=0.1，max_grad_norm=1，AdamW weight_decay=0，seed=42。

每轮使用固定 seed 的 randperm 打乱偏好对，28 步遍历训练集一次；不做早停或开发集选优。CPU float32、eager attention、2 个线程。LoRA 结构继承 v2，不重新初始化；没有 scheduler、混合精度、梯度累积或断点续训。

## 日志与指标

每步保存 pair_id、pair_index、DPO loss、relative_margin、preference_accuracy、chosen/rejected 的序列 log probability、裁剪前 grad_norm 和学习率。train/development 评测对每个偏好对计算指标后取平均，不按回答 token 数加权。

preference_accuracy 使用 relative_margin > 0，即相对 reference 的 chosen/rejected 优势是否提高；不是贪心生成正确率，也不是单纯 chosen_log_prob > rejected_log_prob。初始 policy 与 reference 完全相同，relative_margin=0、loss≈log(2)=0.693147；严格 >0 统计使初始 accuracy=0。这是平局的定义，不表示初始模型所有回答都错。

sequence log probability 会受回答长度影响，chosen 与 rejected 长度并不完全一样。训练 loss 降低或偏好准确率提高，不能直接证明实际工具请求变好；必须在第 4 步进行同条件逐题行为比较。

## 安全与一致性检查

训练前校验冻结快照及 base 缓存、偏好数据文件哈希与基线身份，重新运行语义、split 和 token 审计。完整参数 SHA256 确认 policy/reference 初始一致。

训练后要求：reference 全部参数哈希不变且无梯度；policy 冻结参数哈希不变，policy 参数整体发生变化。只有 LoRA 参数进入 optimizer。

保存到新的 `artifacts/checkpoints/dpo-lora-时间-随机编号/`，含 adapter、tokenizer、train_log.jsonl、summary.json。保存的是在 v2 上继续更新后的完整 LoRA adapter，重载时加载固定裸 base＋此 DPO adapter 即可，不需要再叠加一份 v2 adapter。

释放两份训练模型后加载新的 base＋保存的 DPO adapter，在第一条开发偏好对（chosen/rejected 两序列）上比较全部 logits，容差 atol=rtol=1e-5。此探针证明该输入的保存重载一致性，不是所有输入或跨硬件一致性的保证。

optimizer、scheduler 状态未保存，不能把 adapter 保存称为支持断点续训。基础权重和 adapter 继续由 Git 忽略；实验配置、代码、中文报告入 Git。

## 本机首轮结果

运行目录：`artifacts/checkpoints/dpo-lora-20261009-205045-21dd3596`。完成 28 步，训练参数 230,400 个，保持 v2 的 LoRA 结构。

| 指标 | train 前 → 后 | development 前 → 后 |
| --- | --- | --- |
| 平均 DPO loss | 0.693147 → 0.465978 | 0.693147 → 0.461117 |
| 相对偏好 accuracy | 0/28 → 25/28 | 0/10 → 9/10 |
| 平均 relative_margin | 0 → 0.607413 | 0 → 0.604880 |
| chosen 平均序列 log probability | -2.617167 → -3.187616 | -2.480493 → -2.659235 |
| rejected 平均序列 log probability | -5.827785 → -12.472366 | -6.886070 → -13.113614 |

参数检查全部通过：初始 policy/reference 一致，reference 无梯度且参数不变，policy 的冻结参数不变，policy 已更新。第一条开发偏好对的保存重载 logits 匹配，最大绝对差 0.0。

值得注意的是，chosen 的平均 log probability 也下降了，只是 rejected 下降更多，使相对 margin 提高。DPO 优化的是相对偏好，不保证 chosen 的绝对概率总是增加。此结果尤其需要结合第 4 步的实际生成检查；尚不能声称工具行为已经改善。

首轮保存时 PEFT 的默认 embedding 自动检查尝试查询远程 config，受限网络使查询失败，但回退保存与重载验证成功。随后检查本机 PEFT 实现，将当前脚本改为明确 `save_embedding_layers=False`（词表和 embedding 未改变、未训练），并在模型加载前设置 HF 离线模式。更新后的保存调用由 tiny 模型集成测试验证；没有因此重新执行本次 28 步训练。历史日志里的网络回退情况保留。

## 阅读顺序与运行方式

1. 本文与 `configs/dpo_cpu.json`。
2. `trainers/dpo.py`：encode_pair → pair_log_probs → cache_reference → objective → train_dpo → evaluate_preferences。
3. `scripts/train_dpo.py`：数据审计、加载两份模型、参数校验、保存重载。
4. `tests/integration/test_dpo_training.py`：tiny Llama 的训练方向与冻结不变量验证。

在项目根目录使用项目解释器；默认只读取本地模型缓存：

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -q
& ".\.venv\Scripts\python.exe" scripts\train_dpo.py --config configs\dpo_cpu.json
```

再次运行会创建新的结果目录，不覆盖 v2 或已有 DPO 结果。两份模型同时存在，内存需求高于此前单模型 SFT。

## 下一步

固定本次 DPO 产物，比较 v2 SFT-only 与 SFT＋DPO 的逐题生成。保持 tokenizer、system prompt、base revision、贪心生成与评分协议一致；分别统计改进、退步、不变，以及路由、严格 AST、数值和成对表现。开发偏好指标与行为评测分开解释。
