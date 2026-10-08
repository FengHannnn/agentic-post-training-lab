# Phase 2：乘法与括号组合数据 v2

## 新增内容

保留 v1 全部样本，新增 24 条训练题、8 条开发题，专门覆盖：

| 表达式结构 | 新训练样本 | 新开发样本 |
| --- | --- | --- |
| a*(b+c) | 6 | 2 |
| a*(b-c) | 6 | 2 |
| (b+c)*a | 6 | 2 |
| (b-c)*a | 6 | 2 |

问法包括显式算式、Multiply、product，以及先加减再乘等叙述。模型需要保留外部乘数和括号内部两个操作数，不应退化成只生成括号中的部分表达式。

未使用失败题 `2*(6+5)` 本身作为训练或开发样本。训练与开发使用不同操作数，原 20 条诊断题仍然保留。新开发题共用部分训练模板，因此不构成模板外泛化测试。

## 文件与配置

- `data/training/sft_train_expanded_v2.jsonl`：144 条，104 条工具调用、40 条 final。
- `data/training/sft_dev_expanded_v2.jsonl`：32 条，24 条工具调用、8 条 final。
- `data/training/expanded_v2_targets.jsonl`：176 条对应校验目标。
- `data/training/expanded_v2_manifest.json`：来源、数量和范围说明。
- `configs/sft_lora_expanded_v2_cpu.json`：288 步，完整遍历新训练集两轮。

每 5 条 v1 训练样本插入 1 条新增样本，避免将新增算术题全部集中在训练末尾。保留现有格式与路由样本。v1 文件与配置没有被覆盖，方便对照。

## 检查与运行

项目根目录执行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\validate_sft_dataset.py --config configs\sft_lora_expanded_v2_cpu.json
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\train_sft_lora.py --config configs\sft_lora_expanded_v2_cpu.json --local-files-only
```

审计脚本根据配置的 validation_targets 选择校验文件，仍支持原 v1 配置。校验也拒绝同一集合中的重复输入，避免相同问题出现不同标签。所有目标经过 JSON、schema、路由、表达式、计算结果与跨集重叠检查，另用真实 tokenizer 检查长度和回答边界。

保持模型版本、LoRA 参数和学习率不变，重新从基础模型与新 LoRA 开始训练，不在旧 adapter 上续训。训练完成后用新的产物目录运行 evaluate_tool_calls.py，比较原诊断题的格式与路由是否保持、括号题是否改善。

由于已根据原评测失败设计数据，即使原 20 题全部通过，也不能作为未查看测试集上的成绩。下一阶段应新增未用于调参的复合表达式测试，覆盖不同问法和操作数。
