# Phase 2：SFT 数据流水线与 LoRA 训练

## 当前进度

已实现单轮对话数据读取、回答边界识别、回答专用标签、右侧 padding 和离线检查，并增加真实 tokenizer 接入脚本。下一份文档是 [接入真实 tokenizer](phase2-real-tokenizer.md)。后续再接入 LoRA、训练循环、adapter 保存与训练前后评测。当前代码没有运行模型训练，也不需要租 GPU。

## 阅读顺序

1. `data/examples/sft_train.jsonl`：理解训练样本。
2. `src/agentic_post_training/data/sft.py`：依次阅读 `SFTExample`、`load_examples`、`encode_sft`、`collate_sft`。
3. `src/agentic_post_training/data/demo_tokenizer.py`：理解教学模板；它按字符编码，不能用于预训练模型。
4. `scripts/inspect_sft_data.py`：观察回答边界、batch 长度和训练 token 数。
5. `tests/unit/test_sft_data.py`：理解错误边界与损失一致性测试。

## 样本格式

每行是一个独立 JSON 对象，包含 `id`、`system`、`prompt`、`response` 四个非空字符串。回答中的工具调用 JSON 在这里是文本标签，尚未执行工具。当前只支持单轮 system/user/assistant，不支持多轮工具轨迹。

训练样本与评测样本分别保存。现有 3 条训练样本和 2 条评测样本只用于调试流水线，不能作为模型能力提升的证据；正式实验需要扩充数据，并按任务与题目去重划分。

## 必须理解的四个字段

| 字段 | 用途 | prompt | response | padding |
| --- | --- | --- | --- | --- |
| `input_ids` | 模型输入的 token 编号 | 保留 | 保留 | 填充编号 |
| `attention_mask` | 标记有效输入位置 | 1 | 1 | 0 |
| `labels` | 标准交叉熵目标 | -100 | 对应 token 编号 | -100 |
| `loss_mask` | Phase 1 手写损失的选择掩码 | 0 | 1 | 0 |

prompt 虽然不计算 loss，但仍是回答生成所依赖的上下文，因此 attention mask 为 1。`-100` 是标准交叉熵的忽略标记，不是词表中的 token。

注意：Phase 1 的 `masked_cross_entropy` 内部使用 `gather`，不接受标签中的 `-100`。应调用 `masked_cross_entropy(logits, batch["input_ids"], batch["loss_mask"])`。真实模型的标准交叉熵接口则使用 `labels`。测试验证了两种方式计算结果一致。

## 回答边界如何确定

先对 system/user 使用模板，并开启 assistant 生成提示；再对包含回答的完整对话使用同一模板。只有前者的 token 序列确实是后者的前缀，才用其长度作为回答起点。assistant 开头标记属于 prompt；回答内容与模板自动添加的结束标记参与训练。

不同模型模板可能不满足前缀条件。遇到这种情况主动报错，后续接入真实 tokenizer 时需要检查该模型模板，不能猜测边界。当前也不会截断超长样本，以免切断工具调用 JSON 或结束标记。

padding 根据原始长度生成，不能通过 `input_ids == pad_token_id` 判定；一些模型会复用 EOS 作为 PAD，按编号判定会错误屏蔽真实结束 token。

所有字段保持原始位置对齐，预处理不进行 causal shift；由 Phase 1 损失或真实模型内部执行一次 shift。

## 本地运行

在项目根目录执行：

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py --data data\examples\sft_eval.jsonl
```

如果包尚未安装，先执行 `& ".\.venv\Scripts\python.exe" -m pip install -e .`。本步骤复用当前 PyTorch 与 pytest，不需要安装新的训练依赖或下载权重。

## 完成这一小步的标准

能够解释 attention mask 与 loss mask 的区别、为什么 labels 使用 -100、为什么不能重复 shift、为什么真实结束 token 仍然需要训练。读完并运行检查后，下一步接入真实 tokenizer，验证模板与回答边界，再进入 LoRA 训练。
