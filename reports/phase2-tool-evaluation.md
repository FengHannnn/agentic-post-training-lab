# Phase 2：独立工具调用评测

## 目标

把“loss 降低、单题输出 JSON”推进到多个未训练题目的行为评测。使用已保存的 adapter，比较相同基础版本的 base 与 LoRA。不会重新训练或修改 adapter。

当前有 20 条英语题：16 条计算题和 4 条问候题，覆盖加减乘除、括号、负数、小数和问题表达变化。它们与当前 4 条训练样本、2 条旧评测样本检查精确输入重叠；不保证语义题型独立。这仍是小规模诊断。

## 运行

已有训练依赖即可运行，无需新增安装。项目根目录执行，将 `--run` 指向含 summary.json、adapter/、tokenizer/ 的完整训练结果目录，而不是只指向 adapter/。

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\evaluate_tool_calls.py --run artifacts\checkpoints\sft-lora-20261008-005426-9d8f2036 --local-files-only
```

模型已缓存时可离线运行。采用 CPU float32，按顺序评测 base 后再挂载 LoRA，避免同时保留两份基础模型。每题最多生成 64 token，固定贪心生成；base 与 LoRA 使用相同题目和配置。

## 阅读顺序

1. `data/eval/tool_calls_v1.jsonl`：输入、期望路由、参考表达式和结果。
2. `src/agentic_post_training/environments/calculator.py`：只解释算术 AST，不使用 eval。
3. `src/agentic_post_training/evaluation/tool_calls.py`：解析、校验、执行、评分与统计。
4. `scripts/evaluate_tool_calls.py`：基础模型和 adapter 的实际生成。
5. `tests/unit/test_tool_call_evaluation.py`：评测器本身的边界验证。

## 指标定义

| 指标 | 判断方式 | 分母 |
| --- | --- | --- |
| JSON 合法 | 整段文本是一个 JSON 值，不接受重复键和 NaN | 所有题 |
| 结构合法 | 精确符合 tool_call/calculator/expression 或 final/answer 结构 | 所有题 |
| 路由正确 | 结构合法，且选择期望的 tool_call 或 final | 所有题 |
| 参数可执行 | 返回合法 calculator 请求，表达式符合算术范围且可执行 | 计算题 |
| 参数结构正确 | 表达式 AST 与参考结构一致 | 计算题 |
| 工具执行成功 | 本地计算器完成执行 | 计算题 |
| 结果正确 | 执行结果与参考值在容差内相等 | 计算题 |
| 请求执行闭环通过 | 计算题要求路由、参数结构和结果均正确；问候要求合法非空 final | 所有题 |

参数结构比较会忽略空白和多余括号，但不接受全部数学等价表达式。例如 `27*13` 与 `13*27` 数值相等、结构不同，结果正确但参数结构不通过。这是保守指标，需要阅读逐题结果区分合理改写和真实错误。

问候仅判断 final 路由与非空文本，不判断语义质量。闭环止于解析请求与计算器执行，不包含将工具结果传回模型再生成最终答复，因此不是完整多步 Agent 评测。

## 输出与失败原因

新的结果目录在 `artifacts/evaluations/tool-calls-时间-随机编号/`，包含 base_predictions.jsonl、lora_predictions.jsonl、summary.json 和中文 report.md。保存原始含特殊 token 输出，也保存用于评分的文本、工具结果和失败阶段。

评分文本只通过 tokenizer 去掉特殊 token，不抽取 JSON 片段、不去除 Markdown 围栏、不修复坏 JSON。`hit_token_limit` 用于识别生成长度上限，不能直接认定所有达到上限的输出都不完整。

计算器只支持有限数字、+、-、*、/、括号与一元正负号，限制字符数、节点数及结果范围，拒绝函数调用、名称、幂、除零等。独立检查 JSON 与 schema，才能区分模型格式失败、路由失败和参数失败。

## 如何使用结果

先找失败类别，不急着增加步数。若多数问题结构正确但表达式错，应增加运算解析数据；若输出自然语言或 Markdown，应加强格式数据；若问候误用计算器，应增加路由反例。将此评测集用于反复调参后，它应视为开发集，最终结论需要另建未查看测试集。

同时对比 3 步和 20 步 adapter 可观察训练步数影响，每次运行会产生独立目录。不要从 20 条题外推全面工具能力。
