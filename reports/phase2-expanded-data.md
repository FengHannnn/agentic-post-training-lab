# Phase 2：扩充格式与路由数据

## 本次改动

新增 120 条训练样本与 24 条开发样本，保留原来的 4 条训练数据、2 条旧评测数据、20 条工具评测题和原配置。所有新回答统一为严格 JSON，不附自然语言、Markdown 或计算后的最终答案。

| 数据 | calculator 请求 | final 回答 | 合计 |
| --- | --- | --- | --- |
| 扩充训练集 | 80 | 40 | 120 |
| 扩充开发集 | 16 | 8 | 24 |

80 条训练计算题覆盖乘法、加法、减法、除法、括号后除法、括号后乘法、负数乘法、小数乘法，每类 10 条。每类包含多种问法，例如 multiply、product、times，以及“从某数减去某数”这类参数顺序容易出错的表达。

40 条 final 样本覆盖问候、感谢、告别，以及包含 calculator 或数字但并未请求计算的礼貌话语，避免仅凭关键词或数字调用工具。

## 数据组织

- `data/training/sft_train_expanded_v1.jsonl`：120 条训练数据。
- `data/training/sft_dev_expanded_v1.jsonl`：24 条开发数据，用于训练前后 loss。
- `data/training/expanded_v1_targets.jsonl`：每条样本的路由、参考表达式、结果和类别，供审计使用，不作为额外模型输入。
- `data/training/expanded_v1_manifest.json`：数量、来源与范围说明。

样本按两个工具调用、一个 final 交错排列，配合当前顺序训练循环，避免先训练完所有计算题再集中训练问候。当前并未引入随机采样。

数据按确定性模板构造，回答按统一结构生成，未调用另一个大模型生成标签。训练集与开发集共用问法模板，但操作数和问候文本不同；不能把这种划分称为严格的模板外泛化测试。

## 标签与隔离检查

每条标签经过现有 score_output 检查，要求结构、路由、表达式与结果一致。参考结果由受限计算器验证。审计同时检查训练、开发、原评测之间的 ID、输入和精确算术 AST 是否重叠。

检查不会识别所有数学等价表达式，也不能自动证明自然语言问题一定对应标注表达式；模板编写和人工阅读仍然必要。不要把原 20 条题目答案放进训练集。它们已用于观察并指导本次数据设计，因此后续应作为开发诊断集，最终能力结论仍需新的未查看测试集。

## 本地命令

在项目根目录执行，不需要安装新依赖：

```powershell
& ".\.venv\Scripts\python.exe" scripts\validate_sft_dataset.py
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\train_sft_lora.py --config configs\sft_lora_expanded_cpu.json --local-files-only
```

审计默认只使用本地 tokenizer 缓存。如果仅检查数据而不加载 tokenizer，可加 `--skip-tokenizer`。

新配置固定为此前相同的模型提交版本，保持 rank=4、alpha=8、学习率=0.001、batch_size=1、CPU float32。max_steps=240，让 120 条训练数据各参与两次更新；max_new_tokens=64，减少较长 JSON 被输出长度上限截断的情况。

训练从基础模型和新初始化的 LoRA 开始，不接着旧 adapter 训练。产物在新目录中，旧结果不覆盖。240 步与旧 20 步实验同时改变了数据量和训练量，不能把差异全部归因于某一个因素。

## 训练后比较

使用新的训练目录运行评测：

```powershell
& ".\.venv\Scripts\python.exe" scripts\evaluate_tool_calls.py --run artifacts\checkpoints\新训练目录 --local-files-only
```

上面“新训练目录”需替换为训练脚本实际输出的目录名。先对照旧结果 JSON 合法率 70%、结构/路由正确率 45%、计算题执行通过 7/16，查看格式错误和错误路由是否减少。数据扩充不保证改善，需根据实际结果判断。

## 阅读顺序

先看训练集前 6 行，理解交错排列，再看各运算类别和 final 路由反例；然后阅读 data/validation.py 与 validate_sft_dataset.py，最后查看新的训练配置。
