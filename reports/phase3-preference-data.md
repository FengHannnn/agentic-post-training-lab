# Phase 3 第 2 步：小规模偏好数据

日期：2026-10-08。

## 从监督标签到偏好对

SFT 的每条样本只有一个 response。DPO 的每条样本是在相同 system/prompt 下给出 chosen 和 rejected：希望策略相对冻结 reference，提高 chosen 相对于 rejected 的序列概率。两条回答不是两个不同问题，也不应只按格式或长度判断好坏。

本步不训练模型。用第 1 步冻结的 v2 SFT adapter 生成新的输入候选，再建立有来源记录的偏好数据。

## 数据设计

数据目录：`data/preferences/dpo_v1/`。

| split | 算术 | 问候 | 总偏好对 | family |
| --- | --- | --- | --- | --- |
| train | 24 | 4 | 28 | 10 |
| development | 8 | 2 | 10 | 4 |

训练数字组是 (4,13,9)、(5,17,8)、(6,19,7)、(7,23,11)、(8,29,10)、(9,31,14)；开发数字组是 (10,37,16)、(11,43,18)。每组包含四个结构：

| prompt 模板 | chosen 表达式 | 合成错误对照 |
| --- | --- | --- |
| Calculate a * (b + c). | a*(b+c) | a*b+c |
| Multiply a by b first, then add c. | a*b+c | a*(b+c) |
| Multiply the sum of b and c by a. | (b+c)*a | b+c*a |
| Add b to the product of c and a. | b+c*a | (b+c)*a |

逐组人工复核了模板与表达式关系：先乘后加对应无括号，先求和再乘对应括号；数字均非退化值，错误对照与正确结果不同。所有 chosen 使用严格 calculator JSON。问候 chosen 使用非空 final JSON，合成 rejected 为 calculator 请求；当前只学习路由和协议，不训练问候语的细粒度偏好。

同一数字组合的四条题属于同一 family，整组只进入一个 split。训练和开发共用上述措辞模板，因此开发只检查新数字组合上的表现，不证明模板外或开放域泛化。本轮没有把空格、前缀等所有变体一起扩充，也没有创建最终测试集。

## chosen 与 rejected 的来源

chosen 是人工复核过的确定性模板标签，不声称由模型自己探索得到。

逐题使用冻结 v2、固定 base 提交、CPU float32、eager attention、贪心生成、最多 64 新 token。若模型输出非空且结构、路由或数值有错误，则原样使用评分输出作为 rejected；不补 JSON、不改表达式。原始含特殊 token 的 raw_prediction 同时保留。

若模型已经在结构、路由与数值上答对，使用明确标注的合成错误对照。特别是交换乘法或加法顺序、严格 AST 不同但数值正确的输出，不作为语义错误 rejected。实际偏好对混合真实模型错误与合成负例，应分别统计来源；不能称为全部来自模型的自然偏好数据。

每条样本包含 id、family_id、system、prompt、chosen、rejected、category、expected_type、expected_expression、expected_result 和 provenance。后续 DPO 只把 system/prompt 和对应回答送入 tokenizer；参考表达式、结果与 provenance 是审计侧字段，不是额外模型输入。

provenance 保存基线清单哈希、两条回答来源、失败类别、模型原始/评分输出、是否达到生成上限及逐项评分。`manifest.json` 保存 split 汇总、真实 token 长度、模型版本和数据文件 SHA256。完整生成日志另存 `artifacts/evaluations/dpo-preferences-v1/v2_predictions.jsonl`，该目录由 Git 忽略；版本化的偏好行中仍含生成证据。

## 审计规则

1. chosen 必须通过严格 JSON、schema、路由、表达式 AST 与数值评分。
2. rejected 必须非空、与 chosen 不同，并且不通过结构/路由/数值语义校验；仅 AST 不同不足以判错。
3. train/development 的 ID、输入、prompt、精确表达式 AST 无重叠；family 不跨 split。
4. 对现有全部 eval JSONL、v1/v2 SFT train/dev 与英语 examples 做 ID、prompt 和精确表达式隔离检查。受保护集合自身含重复记录，但不会作为新样本训练。
5. 使用冻结 tokenizer，chosen/rejected 分别复用 encode_sft 的模板前缀、response mask 与长度检查，max_length=192，不截断长回答。
6. 重新审计时核对数据文件与基线清单哈希；真实模型 rejected 必须等于保存的 generation.prediction。

自动算术校验不能证明自然语言标签含义；本轮通过明确模板及人工复核补足这一点。精确 AST 隔离也不是所有数学等价表达式的隔离。语义评分只验证本组数值，仍不是符号等价证明。

## 本次审计结果

实际生成 38 次：train 中 13 个 rejected 来自 v2 真实错误，15 个为合成负例；development 中分别为 3 个和 7 个。所有算术 rejected 都是可执行但参数/数值错误，所有问候 rejected 都是合成错误路由，本版没有格式损坏负例。

chosen 完整 token 长度最大 train=88、development=90；rejected 最大分别为 88、89。回答区域最少 chosen=12、rejected=24 token，均通过 max_length=192 的真实模板检查。训练中 chosen/rejected 平均回答长度约为 25.50/26.79，开发约为 25.80/27.50。长度不完全相同，后续需要观察 DPO 的序列 log probability 和长度偏差。

所有 chosen、rejected、family 隔离与受保护数据隔离检查通过。逐条复核了 38 个输入及表达式：例如训练 `Multiply 4 by 13 first, then add 9.` 的 chosen 为 `4*13+9`，v2 的实际错误会按生成日志保留。完整测试 70 项通过，含新增的 6 项偏好审计边界测试；仍有此前 tiny 模型保存的 PEFT 提示。

## 阅读与运行顺序

先读本文，再看 `data/preferences/dpo_v1/train.jsonl` 的前四条；随后阅读 `data/preferences.py` 的 PreferenceExample 和 validate_preferences，最后阅读 `scripts/build_dpo_preferences.py` 的候选设计与来源选择。

在项目根目录重新审计，无需重新生成：

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_dpo_preferences.py --audit-only
& ".\.venv\Scripts\python.exe" -m pytest tests -q
```

只有数据目录和对应日志目录都不存在时才能生成，已有输出不会覆盖。复现实验可用 `--output data/preferences/dpo_v1_repeat` 指定新目录，对应日志会另存为 `artifacts/evaluations/dpo-preferences-dpo_v1_repeat`。重新生成不保证不同依赖或硬件下每条输出完全一致，需重新审计。

本数据用于最小 DPO 训练闭环，不承诺它优于继续 SFT。下一步从冻结的 v2 同时初始化 policy/reference，训练新的 policy adapter，验证 reference 不变、policy 更新、保存重载，再复用逐项行为评测。
