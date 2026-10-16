# Phase 3：DPO退化定位与稳定性恢复

日期：2026-10-16。

## 本轮结论

四轮修复后，`chosen_nll_weight=5、learning_rate=1e-5`的混合目标候选显著缓解了大规模退化：新开发严格36/56、数值44/56、成对16/24，恢复SFT水平；工具20/20、括号34/48、空格12/12、旧偏好7/10。新56条JSON/schema/路由均正确；全部146条各145通过，仍有1条新增坏JSON，无生成上限命中。

第四轮通过四项回归恢复检查，但新开发集无严格或语义净提升，未通过预先声明的第五项条件。因此全部五项质量验收仍未通过，不把恢复候选晋升为新baseline，也不宣称纯DPO全面优于SFT。旧工具/括号各净增1只代表小幅开发诊断收益。

## 全部结果

| 数据组严格通过 | 冻结SFT | 原C纯DPO | 监督1/LR1e-4 | 监督5/LR1e-4 | 监督5/LR2e-5 | 监督5/LR1e-5 |
| --- | --- | --- | --- | --- | --- | --- |
| preference_v2_development | 36/56 | 4/56 | 31/56 | 36/56 | 35/56 | 36/56 |
| tool_calls | 19/20 | 2/20 | 19/20 | 19/20 | 20/20 | 20/20 |
| brackets | 33/48 | 1/48 | 29/48 | 34/48 | 34/48 | 34/48 |
| spacing_control | 12/12 | 0/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| preference_v1_development | 7/10 | 1/10 | 5/10 | 6/10 | 7/10 | 7/10 |

| 模型 | 新56题数值/路由 | 算术成对数值（24对） | 新JSON/schema/路由（各56） |
| --- | --- | --- | --- |
| 冻结SFT | 44/56 | 16 | 56/56/56 |
| 原C纯DPO | 4/56 | 0 | 43/28/4 |
| 监督1/LR1e-4 | 31/56 | 5 | 53/45/45 |
| 监督5/LR1e-4 | 36/56 | 6 | 54/53/53 |
| 监督5/LR2e-5 | 42/56 | 13 | 56/56/56 |
| 监督5/LR1e-5 | 44/56 | 16 | 56/56/56 |

相对SFT，新56题严格和语义均为0改善、0退步。其他组的逐题改善/退步见末尾；分组总分持平不意味着逐条没有变化。各组含重复输入、已见表达式探针，不合并成独立测试准确率。成对要求grouped/plain两题同时数值通过，8问候不进入算术分母。数值正确不证明符号等价。

## 为什么会退化

1. **相对目标允许正确答案概率下降。** 原C开发chosen平均序列log probability从−3.643747降至−17.491135，rejected从−9.124099降至−41.487884。间隔扩大但正确输出本身更不容易生成，概率质量可能流向固定偏好对之外的回答。偏好accuracy衡量相对reference的改善，不等于chosen已比rejected更可能，更不是greedy生成通过率。
2. **公共格式缺少直接偏好信号。** 144条算术训练偏好平均共享18个回答token（17至19），包含JSON/calculator前缀；确定性teacher forcing下，共同前缀项在chosen−rejected中抵消。后续不同token的梯度通过共享LoRA参数仍可能损伤前缀。原C随后出现大量格式/路由错误。这是机制线索，不声称已量化各项独立因果贡献。
3. **更新幅度影响稳定性。** 只加监督权重1未修复；权重5仍未修复；固定权重5后降低学习率逐步恢复结构和旧能力。后三轮相邻对照只改变学习率。这支持该模型/数据组合对更新幅度敏感，不意味着1e-4对所有DPO都不合适。
4. **固定负例与真实错误存在覆盖缺口。** 原SFT已在49/56对中给chosen更高概率，但生成语义通过44/56；5道生成错题仍更偏好固定chosen。12道实际错误仅3道与预设rejected的AST或结果相同。更多同类合成负例不等于更针对实际错误；也不能据此证明来源是全部退化原因。逐条见baseline_pair_coverage.json。

未发现DPO符号、causal shift、EOS/PAD掩码、独立reference冻结或保存重载的明显实现错误。LoRA与attention dropout均为0。原B/C另有单adapter各两题复现，与主生成一致。严格与数值同时下降，不能靠放宽AST标准解决。

## 修复目标与控制

```text
L = L_DPO + alpha * (-log pi(chosen | prompt) / chosen_response_tokens)
```

DPO使用回答序列log probability总和；监督项用token平均，包含终止符、不含prompt和padding。alpha=1不代表两项尺度相等，算术回答约30 token；alpha=0保留原DPO损失及梯度。参考[TRL官方混合目标说明](https://huggingface.co/docs/trl/main/en/dpo_trainer#multi-loss-combinations)，本项目记录自己的归一化约定，不依赖安装TRL。

四轮都从同一冻结SFT重新开始，同168对训练/56对开发、168步、seed42、beta0.1、裁剪1、相同LoRA与CPU设置。每轮chosen4500、rejected4788个回答token；初始化、逐步样本顺序和token曝光均核对一致。

第一→二轮只改监督权重1→5；第二→三轮只改LR1e-4→2e-5；第三→四轮只改LR2e-5→1e-5。相对原C，最后方案同时改变监督目标和学习率，不能把全部收益归因于其中一项，也未证明两项各自必需。没有同数据同预算的纯SFT控制，不能把混合目标收益归因于DPO项。

## 预先质量门

在生成前声明：新严格至少36、数值至少44；新56条结构/路由全部通过；旧组严格至少19/20、33/48、12/12、7/10；并要求新严格或数值至少一项超过SFT。实际结果：

```json
{
  "new_strict_at_least_sft": true,
  "new_semantic_at_least_sft": true,
  "new_json_schema_route_all_valid": true,
  "legacy_strict_not_lower_than_sft": true,
  "new_strict_or_semantic_exceeds_sft": false
}
```

没有随结果放宽条件。第一至三轮失败记录全部保留。源码/失败诊断/方案提交49d976a、6d95dd3、f9174f8，后两轮方案在其生成前固定。当前显著缓解退化，但仍有结构退步且新增收益未完成，不能称为完整修复成功。

## 核验与缓存

- 完整103项测试通过，无skip；唯一提示来自已有tiny PEFT保存配置。
- 四轮reference无梯度且参数不变、仅LoRA更新、基础参数不变、重载logits最大差0。
- 每轮438条保存记录重新评分（含semantic），15份预测哈希、10组对照/changes、运行与训练源码、质量门均核对一致。438条包含重复引用的SFT/C，不是438道独立新题。四轮共580次新生成，584条候选记录含4次重复输入复用。
- 第二轮保存224对reference分数及SHA。第三/四轮复用前核对模型revision/初始参数digest/完整数据manifest/编码与评分实现、全部张量与冻结状态；新policy/reference各对train/dev首对前向，共4次复现。初始汇总明确复用相同模型已测值，训练后仍完整重新评分。
- 第三/四轮首步loss、chosen/rejected log probability、chosen NLL和grad_norm与第二轮精确一致，只有学习率不同。reference_log_probs.pt是固定分数，不是模型权重。
- 前两轮启动时的精确训练/比较源码保存在各评测source目录，SHA匹配当时清单；不能用扩展后的当前入口冒充历史源码。

## 产物与代码阅读

总汇：`artifacts/evaluations/dpo-repair-summary-20261016-201033-4f337b8d`。summary.json含全部模型/配置/控制与质量门；report.md给总表，15份相邻控制changes保留全部改善/退步；baseline_pair_coverage.json给固定负例覆盖诊断。

- 监督1/LR1e-4：训练`artifacts/checkpoints/dpo-lora-20261016-180833-e336ae0b`；评测`artifacts/evaluations/dpo-anchor-20261016-182453-d989d3f1`。
- 监督5/LR1e-4：训练`artifacts/checkpoints/dpo-lora-20261016-184519-79d1335a`；评测`artifacts/evaluations/dpo-anchor-20261016-190110-56ef6fae`。
- 监督5/LR2e-5：训练`artifacts/checkpoints/dpo-lora-20261016-191348-bfddf667`；评测`artifacts/evaluations/dpo-anchor-20261016-192955-24ca2895`。
- 监督5/LR1e-5：训练`artifacts/checkpoints/dpo-lora-20261016-194142-df9d5829`；评测`artifacts/evaluations/dpo-anchor-20261016-195656-ae7cb3a7`。

阅读顺序：

1. `trainers/dpo.py`的anchored_objective、train_dpo：原DPO、chosen平均NLL、反向/裁剪/更新。
2. `configs/dpo_v2_large_168_anchor*.json`：四轮只改声明的权重/LR。
3. `scripts/train_dpo.py`：目标日志、reference缓存身份/探针、冻结与重载；before/after loss仍是纯DPO诊断，train_log loss是训练总目标。
4. `scripts/compare_dpo_repair.py`：训练控制、历史身份、候选生成与预先质量门。
5. `scripts/audit_dpo_repair.py`：不加载模型，重新核对记录、源码、缓存、汇总和质量门。
6. `test_dpo_anchor.py`、`test_dpo_repair_control.py`和PEFT集成：零权重一致、公共前缀梯度、非法参数、单变量与曝光、冻结/更新/重载。

```powershell
& ".\.venv\Scripts\python.exe" scripts\audit_dpo_repair.py --behavior artifacts\evaluations\dpo-anchor-20261016-195656-ae7cb3a7
```

第四轮相对SFT的逐组严格改善/退步：

- preference_v2_development：改善0，退步0。
- tool_calls：改善1，退步0。
- brackets：改善1，退步0。
- spacing_control：改善0，退步0。
- preference_v1_development：改善0，退步0。

## 最终审计补充：总分会掩盖失败类型恶化

第四轮旧括号题`bracket-seen_large-sequence-left_plain`，输入`Multiply 12 by 3 first, then add that result to 41.`。SFT输出合法JSON但表达式为`12*3`，参数错误；候选输出`{"type":"tool_call","name":"calculator","arguments":{"expression":"12*3+"41"}}`，变成非法JSON。两者端到端都失败，因此总通过数与端到端transition没有显示新结构退步。

审计现在额外逐题拒收任何新JSON/schema/路由退步，测试覆盖总成功数和总结构数都不变仍有单题恶化的情形。该门是审计发现后新增的独立保护，不改写原五项预先验收历史；第四轮此门也未通过，不能替换冻结SFT。完整提升与该结构问题都仍需解决。本轮没有把失败的候选标为成功。

## 下一步边界

本轮不再扫当前合成数据的LR，也不把恢复到基线包装为新开发提升。优先在训练输入上采集冻结SFT真实错误，用可执行目标审计偏好，补齐错误覆盖；开发答案不进入训练。同时补同数据同预算chosen-only SFT控制，区分正例监督与偏好项贡献。方案冻结后还需真正未查看的最终测试。

单种子、共享措辞模板、训练多为两位数而新开发含三位数等限制仍在。v2 baseline、原C及全部失败实验保留；未启用GPU、付费资源或Phase 4。
正式拒收检查可在审计命令末尾加`--require-admission`；质量未达标时返回退出码2。默认审计退出0只表示记录一致，不代表模型质量通过。
