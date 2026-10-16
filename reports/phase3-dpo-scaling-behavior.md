# Phase 3：v2 DPO规模对照的实际生成评测

日期：2026-10-16。项目根目录：`C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab`。

## 本步范围

比较冻结v2 SFT、A小集28步、B小集168步、C大集168步的真实生成。模型均使用同一固定基础模型版本，切换四个命名LoRA adapter，各次只启用一个，参数冻结、no_grad，不训练。设置均为CPU float32、eager注意力、greedy、单beam、最多64个新token、相同tokenizer/模板/EOS/PAD。

| 数据组 | 记录数 | 含义 |
| --- | --- | --- |
| preference_v2_development | 56 | 新开发集，48算术+8问候 |
| tool_calls | 20 | 原工具开发诊断 |
| brackets | 48 | 原括号/问法诊断，含已见SFT表达式探针 |
| spacing_control | 12 | 原空格/措辞控制 |
| preference_v1_development | 10 | 旧偏好开发生成，保留历史诊断 |

每模型146条记录，共584条；其中只有145种不同system/prompt，同一模型相同输入可缓存复用。各组含重叠题，不能合并成独立测试准确率。数据均为开发诊断，没有最终未查看测试。

## 身份与复用

加载上一轮已核对的三分支训练summary，校验训练摘要、adapter、基线和评测数据哈希；按实际config.train_file加载训练数据。SFT训练暴露来自基线快照；A/B来自28对小集，C来自168对大集，不用大集冒充小集的训练暴露。新数据按ID/输入/精确AST隔离，旧括号保留已见SFT探针标记。

SFT旧90条读取2026-10-10已保存记录，核对基线/模型版本/生成设置/依赖版本、输入/目标/数据哈希，重算评分；相同历史输入必须有一致生成记录。当前generation.py与冻结快照规范化文本一致。再新加载SFT实际生成工具/括号/空格各1题，核对prediction、raw_prediction、token数量、上限命中四项。通过后复用旧90条。不是本轮重新生成这90条。

SFT的新56条与A/B/C的145种独特输入实际生成，额外有3条SFT重现探针。每条记录保存generation_source（fresh、same_input_cache或historical_sft），并写入摘要。相同输入缓存只在单个模型内共享，不跨adapter共享。

## 新开发集的成对统计

旧括号category形如`number_set/wording/form`，可按原逻辑分解。新开发集category只有form，必须从偏好provenance读取wording和contrast_id，不能套用旧字符串拆分。

48条算术组成24对：相同数字family、相同措辞、相同左右形式的一道grouped题和一道plain题。两道题语义通过才算这一对通过。8条问候单独汇总，不加入成对算术指标。这里的“成对”是两道互补题，不是单条偏好样本中的chosen/rejected。

严格通过保留AST要求；数值通过只保证当前数字结果，不能证明符号等价。问候检查结构和路由，不评价回答内容质量。

## 运行结果

产物目录：`artifacts/evaluations/dpo-v2-scaling-behavior-20261016-163551-cb6e9477`。`report.md`为四模型逐题输出，`summary.json`为总指标，25份`*_changes.json`保存五种对照×五组题的变化，20份模型/数据组JSONL保存原始预测与评分。

| 数据组 | SFT严格 | A小集28步 | B小集168步 | C大集168步 |
| --- | --- | --- | --- | --- |
| preference_v2_development | 36/56 | 33/56 | 7/56 | 4/56 |
| tool_calls | 19/20 | 19/20 | 4/20 | 2/20 |
| brackets | 33/48 | 34/48 | 4/48 | 1/48 |
| spacing_control | 12/12 | 12/12 | 3/12 | 0/12 |
| preference_v1_development | 7/10 | 5/10 | 1/10 | 1/10 |

| 模型 | 新开发算术数值通过 | 新开发算术成对数值通过 | 新开发问候严格 |
| --- | --- | --- | --- |
| sft | 36/48 | 16/24 | 8/8 |
| small_28 | 33/48 | 13/24 | 8/8 |
| small_168 | 3/48 | 0/24 | 4/8 |
| large_168 | 0/48 | 0/24 | 4/8 |

新56题按严格标准：SFT→A为0改善/3退步，SFT→B为0改善/29退步，SFT→C为0改善/32退步；A→B为0改善/26退步，B→C为0改善/3退步。A的旧括号题净增1题，并不代表其他组同步获益。B/C在多组题上严重退化。

| 新开发集检查 | SFT | A | B | C |
| --- | --- | --- | --- | --- |
| JSON合法 | 56/56 | 54/56 | 39/56 | 43/56 |
| schema合法 | 56/56 | 54/56 | 21/56 | 28/56 |
| 路由正确（含合法结构要求） | 56/56 | 54/56 | 10/56 | 4/56 |
| 命中生成上限 | 0 | 0 | 2 | 1 |

C的新56题失败分解：invalid_json 13、invalid_schema 15、wrong_route 24、通过4。C的48条算术没有数值通过，4条通过均为问候。B的7条通过为3条算术、4条问候。SFT的严格36/56与数值44/56不同，是因为8条算术输出数值正确但精确AST不一致。

## 结果核验

主运行完成后，逐条重算584条评分、4模型×5组汇总、25组对照和变化文件，核对20份预测文件及6份核心源码哈希，全部一致，见`verification.json`。SFT历史复用的3条实际生成探针一致。

为检查B/C严重退化是否与多adapter切换有关，另外分别启动全新基础模型，只加载该分支一个默认名adapter，各生成一道算术与一道问候。四项生成字段全部与主运行一致，见`single_adapter_verification.json`（B）及`single_adapter_verification_c.json`（C）。这是4条定点复现，没有重跑全套独立模型评测。

C算术复现例：输入`Evaluate 30*(103+61).`，实际输出`{"type":"finalAnswer","name":"evaluator","answer":"30*(103+61)"}`，不符合约定schema。C问候复现例：输入第一道问候，输出`{" " "Hope your week has been pleasant!" }`，JSON非法。这些失败在单独加载C时仍出现。

实际数据生成调用491次（SFT新56、三分支各145），另有SFT重现探针3次、B/C定点复现4次，共498次。584条正式记录中90条来自历史SFT，3条是各DPO分支内部重复输入复用；不把它们声称为584次新生成。完整89项测试通过，1条已有tiny PEFT配置提示。

## 本轮结论与下一步

固定偏好评分改善与真实生成出现背离：A/B/C的开发相对偏好改善分别45/56、53/56、55/56，而严格生成通过33/56、7/56、4/56。训练诊断已显示chosen的平均log probability也在下降，只是rejected下降更多；DPO相对目标改善不保证chosen绝对概率或greedy生成正确率提高。

A→B表明本次小集反复更新伴随严重退化；B→C在相同步数、近似回答token预算下扩大数据覆盖，也没有恢复生成能力。因此，当前证据不支持“只要增大样本数就能改善”的解释，也不能推出数据规模普遍无效。单种子、一个子集及共享合成模板仍限制因果与泛化结论。

可能的机制需另做验证：算术chosen/rejected都走calculator路由，主要对比分组；问候负例才直接对比final与tool_call。共享参数下，固定负例可能未充分约束格式与算术路由，chosen绝对概率下降也值得关注。现有结果尚不能确定各因素的贡献。

建议先阅读本次生成/汇总代码和具体错误，再设计一个明确控制实验，例如对同一v2数据做匹配预算的chosen-only SFT，或单独控制DPO更新强度。新v2数据的168步SFT对照尚未执行；不能用旧v1的28步SFT对照替代。保留全部失败结果，本步没有继续训练、挑选checkpoint或开启Phase 4。


## 代码阅读顺序

1. `src/agentic_post_training/evaluation/scaling.py`：training_keys按实际训练集建立暴露索引；behavior_record重新评分并补元数据；v2_subgroups分开算术成对和问候。
2. `scripts/compare_dpo_scaling.py`：身份核对、历史复用、四adapter切换、生成、逐题写入和恢复、五组对照。
3. 新增`tests/unit/test_scaling_behavior.py`：小/大训练暴露不同、24完整算术对、单条坏JSON只破坏对应算术对、数值等价但AST不一致的区别。完整89项测试通过，唯一提示仍来自已有tiny PEFT测试。
4. report.md四列逐题输出，再按对照名查看changes JSON。changes采用before_/after_字段，含义由文件名指定。

五种对照：SFT→A、SFT→B、SFT→C、A→B（重复更新）、B→C（同总步数的覆盖变化）。不根据固定答案偏好率挑选一个分支后只报告它。

## 运行方式

```powershell
& ".\.venv\Scripts\python.exe" scripts\compare_dpo_scaling.py --training-comparison artifacts\evaluations\dpo-v2-scaling-training-20261016-152209-97915019
```

默认每次创建新目录。长运行逐条写入并flush；中断后可使用`--output 原目录 --resume`恢复已完成的完整JSONL行前缀。恢复前核对权重/数据/设置/核心源码身份，重新校验已存记录，不重新生成完整前缀。需要原进程已经结束，不对同一输出目录同时启动两次；损坏或不完整JSON行会明确报错，不静默忽略。

原checkpoint、数据、训练摘要保持原样，本步没有重新训练或调参。单种子、单次小子集、共享措辞模板和合成负例仍限制结论；新开发集包含三位数操作数，不能与旧开发集难度直接等同。
