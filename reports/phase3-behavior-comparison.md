# Phase 3 第 4 步：SFT-only 与 SFT＋DPO 逐项行为比较

日期：2026-10-10。

## 比较的问题

第 3 步看到偏好开发 loss 从 0.693147 降到 0.461117，9/10 对的相对 margin 提高。本步检查这是否转化为生成正确性的变化，并保留可能的退步。相对概率变化和实际贪心生成结果是不同指标。

SFT-only 是冻结的 v2 adapter；SFT＋DPO 是 `artifacts/checkpoints/dpo-lora-20261009-205045-21dd3596` 的完整 LoRA adapter。两者共享固定 SmolLM2 base 提交，不叠加两份 adapter，不做继续训练或输出修复。

## 数据和条件

| 数据组 | 题数 | 用途 |
| --- | --- | --- |
| tool_calls | 20 | 原协议、路由与算术工具开发诊断 |
| brackets | 48 | 数字、问法和运算结构的成对诊断 |
| spacing_control | 12 | Calculate 紧凑算式控制 |
| preference_development | 10 | DPO 开发偏好题目上的实际生成 |

每个模型重新生成 90 次，共 180 次。主诊断含已见 SFT 表达式探针，记录 seen_sft_training_prompt/expression；同时记录是否见过 DPO 训练 prompt/expression。偏好开发数据没有参与梯度更新，但已用于审计和训练前后指标查看，不能视为最终未查看测试集。数据组之间含重叠题意，因此不把 90 条总计当独立测试集准确率。

所有输入保持各自相同的 system/prompt。固定 CPU float32、eager attention、2 个线程、seed=42、贪心生成、num_beams=1、最多 64 新 token。对比 tokenizer vocab、chat template、PAD/EOS 和特殊 token 配置，并逐题确认编码的 prompt token 完全一致。

原三组题目从冻结快照读取；偏好开发数据文件哈希必须与训练时记录一致。保存 base 提交、数据和 adapter 文件哈希、版本、生成条件、逐题 raw_prediction 与 prediction。

## 评分与变化

保持原评分器：严格 JSON、精确 schema、路由、受限计算器执行、严格 AST 和当前数字结果。原始输出只去 tokenizer 特殊 token，不补 JSON 或表达式。

- 严格 end_to_end_pass：工具请求结构、路由、表达式 AST、结果都正确。
- semantic_pass：工具题结构、路由、数值正确；问候题检查 final 协议、路由和非空文本。
- pair_semantic_pass：括号与无括号两题都数值正确。

逐题按 ID 对齐，禁止重复 ID 或标准目标不一致。每个评分指标分别记录：

| 状态 | SFT | DPO |
| --- | --- | --- |
| improved | 错 | 对 |
| regressed | 对 | 错 |
| both_pass | 对 | 对 |
| both_fail | 错 | 错 |

另外记录 prediction 是否变化。两个模型都错不等于输出没变；严格指标与数值指标的变化也可能不同。为主诊断和控制题保留问法、数字、结构分组与成对统计。

## 代码阅读顺序

1. `evaluation/comparison.py`：transition → compare_records；理解 ID 对齐与不同指标的改善/退步。
2. `scripts/compare_sft_dpo.py`：核对基线、加载 adapter、逐题生成、保存与汇总。
3. 结果目录的 `report.md`、各组 `*_changes.json` 和 `summary.json`。

在项目根目录运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\compare_sft_dpo.py --dpo-run artifacts\checkpoints\dpo-lora-20261009-205045-21dd3596
```

只做评测，不更新权重。新建时间和随机编号的输出目录，不覆盖历史产物。程序默认离线使用模型缓存。

本次新增 4 项对比测试，验证 ID 顺序不同仍可对齐、重复 ID 拒绝、目标不同拒绝，以及严格/数值变化分别统计。完整测试 75 项通过，保留已有 tiny 模型 PEFT 保存提示。

## 本次实测结果

结果目录：`artifacts/evaluations/sft-dpo-20261010-150134-fc7e5e9a`。180 次生成完成；原三组共 80 条 SFT prediction 与冻结历史文本全部一致，无差异 ID。

| 数据组 | SFT 严格通过 | DPO 严格通过 | 改善 | 退步 | 两者都对 | 两者都错 |
| --- | --- | --- | --- | --- | --- | --- |
| 原工具题 | 19/20 | 20/20 | 1 | 0 | 19 | 0 |
| 括号主诊断 | 33/48 | 33/48 | 3 | 3 | 30 | 12 |
| 空格控制 | 12/12 | 11/12 | 0 | 1 | 11 | 0 |
| 偏好开发题 | 7/10 | 8/10 | 1 | 0 | 7 | 2 |

本轮数值/路由通过数与严格通过数一致，因此语义变化的改善/退步数量也一致。两模型各组 JSON、schema 与路由均通过，没有触及 64-token 生成上限；变化主要落在算术表达式，不能归因于 JSON 修复或生成截断。

括号主诊断的成对数值通过从 11/24 降到 10/24，空格控制从 6/6 对降到 5/6 对。总分相同也可以伴随不同错误分布和成对一致性的退步。

| 主诊断问法 | SFT 数值正确 | DPO 数值正确 |
| --- | --- | --- |
| spaced | 11/12 | 11/12 |
| compact | 12/12 | 11/12 |
| sequence | 4/12 | 5/12 |
| natural | 6/12 | 6/12 |

## 具体改善与退步

下面只展示 expression 字段，完整 JSON 与 raw_prediction 保存在逐题文件中。

| 输入 | 标准表达式 | SFT 输出 | DPO 输出 | 变化 |
| --- | --- | --- | --- | --- |
| Calculate 2 * (6 + 5). | 2*(6+5) | 2*6+5 | 2*(6+5) | 改善 |
| Multiply 2 by 6 first, then add 5. | 2*6+5 | 2*6 | 2*6+5 | 改善 |
| Multiply 3 by 7 first, then add 4. | 3*7+4 | 3*(7+4) | 3*7+4 | 改善 |
| Calculate 3 * 41 + 12. | 3*41+12 | 3*41+12 | 3*(41+12) | 退步 |
| Compute 3*41+12. | 3*41+12 | 3*41+12 | 3*(41+12) | 退步 |
| Take 3 and multiply it by the result of adding 41 and 12. | 3*(41+12) | 3*(41+12) | (3+41)*12 | 退步 |
| Calculate 10 * (37 + 16). | 10*(37+16) | 10*37+16 | 10*(37+16) | 偏好开发题改善 |

原工具题中的唯一改善与主诊断的第一行是相同输入，不能算作两条独立证据。空格控制的退步也是数字组 (3,41,12) 的右侧无括号表达式，不能把它与主诊断同族变体当成完全独立的失败现象。

## 结果如何解读

本轮 DPO 影响了真实生成，并修正了部分错误；但它没有带来一致的运算顺序鲁棒性提升。原 20 题变成 100% 只说明这份已查看开发诊断通过，不能证明全面泛化。natural 问法仍为 6/12，偏好开发的左侧无括号题仍有两条错误。

相对偏好 accuracy=9/10 与实际开发生成正确率=8/10 不矛盾：前者比较指定 chosen/rejected 的概率比相对 reference 是否提高；后者检查模型贪心生成的完整回答。两者定义不同，不应把 9/10 直接写成工具成功率。

这是固定 v2 起点、一个 seed、28 步、混合真实/合成负例的小实验，没有同预算继续 SFT 的对照。因此可报告此次 SFT→DPO 前后行为变化，但不能据此证明 DPO 优于继续 SFT，也不能确定数据比例、长度偏差、学习率或模型容量是哪一个造成了退步。

Phase 3 的四步最小闭环已完成。后续可先总结日志、数据来源与失败题，再按明确问题设计单变量实验；本步没有据开发结果追加训练或改参数。
