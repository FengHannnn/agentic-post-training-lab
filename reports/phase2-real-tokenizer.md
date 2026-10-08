# Phase 2：接入真实 tokenizer

## 这一步做什么

用真实 tokenizer 验证现有 `encode_sft` 和 `collate_sft`，观察真实 chat template、回答起点、结束 token 与 padding。只加载 tokenizer，不加载模型权重、不运行训练、不需要 GPU。

默认使用 `HuggingFaceTB/SmolLM2-135M-Instruct`，换一个模型家族，并为后续 CPU 小规模训练做准备。模型主要面向英语，现有中文数据这里只用于编码检查，不代表它的中文回答能力。正式训练时会调整任务数据与语言。脚本可通过 `--model` 指定另一个模型或本地 tokenizer 目录。

## 安装与运行

在本项目根目录、独立虚拟环境内执行：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e ".[tokenizer]"
& ".\.venv\Scripts\python.exe" scripts\inspect_hf_tokenizer.py
& ".\.venv\Scripts\python.exe" scripts\inspect_hf_tokenizer.py --data data\examples\sft_eval.jsonl
& ".\.venv\Scripts\python.exe" -m pytest tests -v
```

首次运行需要联网，缓存放在项目内 `.cache/huggingface`，已从 Git 排除。依赖暂时限制在 Transformers 4.x，避免教学过程中跨大版本接口变化；后续训练框架版本会另行验证。

已经缓存后可以离线运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\inspect_hf_tokenizer.py --local-files-only
```

脚本支持 `--revision`，正式实验应使用固定模型仓库提交版本；当前默认 `main` 便于首次学习。网络下载失败时可重新运行，或用 `--model` 指向已下载的 tokenizer 目录。

## 阅读顺序

先阅读 `src/agentic_post_training/data/hf_tokenizer.py`，再阅读 `scripts/inspect_hf_tokenizer.py`。前者只负责加载与接口适配，后者复用已读过的数据预处理函数进行实际检查。

`AutoTokenizer.from_pretrained` 加载分词配置与词表，不等于 `AutoModelForCausalLM.from_pretrained`。只有后者才会加载模型权重。本步骤没有使用后者。

适配器显式指定 `return_dict=False`，使 `apply_chat_template` 返回 token ID 列表，与 `ChatTokenizer` 约定一致。解码时保留特殊 token，关闭额外空格清理，便于看清边界。没有 PAD 时复用 EOS；两者都没有时主动报错。SFT 使用右侧补齐，这不代表将来批量生成也必须使用右侧补齐。

## 输出需要看懂什么

1. PAD 与 EOS 的字符串和编号是否相同；不能仅凭编号识别 padding。
2. prompt 的尾部包含 assistant 开头标记，这部分不训练。
3. 回答区域包含标准回答及模板的终止标记，终止标记需要训练。
4. token 数与字符数不同，中文字符可能被拆为多个 token。
5. batch 的 padding 对应 `labels=-100`、`loss_mask=0`。

脚本检查当前模型的 tokenizer EOS 是否出现在回答区域。其他模型可能使用不同的回合终止方式；若检查失败，应阅读该模型模板再适配，不应直接删除检查。

本步骤的单元测试使用替身对象，不联网；实际脚本运行才检验真实下载文件。通过 tokenizer 检查并不代表模型已完成推理或微调。

## 后续

理解实际输出后，接入真实模型与 PEFT，进行极小规模 LoRA 训练，确认 loss、梯度、adapter 保存和加载。正式效果评测需要更多数据与独立评测集。

## 参考资料

- [Hugging Face 对话模板文档](https://huggingface.co/docs/transformers/en/chat_templating)
- [SmolLM2-135M-Instruct 模型说明](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
