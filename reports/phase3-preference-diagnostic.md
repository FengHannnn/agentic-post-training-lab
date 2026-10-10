# Phase 3 后续：偏好概率分组诊断与受控对照设计

日期：2026-10-10。

## 为什么先诊断

首轮 DPO 的偏好 loss 降低，但实际行为有改善也有退步。直接增加训练步数无法区分：模型更认可正确答案，还是主要压低了负例；哪些题型获益，哪些题型仍弱。本步固定模型与数据，仅重新评分 28 条 train 和 10 条 development 的 chosen/rejected，不生成新数据、不更新参数。

## 看哪些量

```text
delta_chosen = DPO log π(chosen) − SFT log π(chosen)
delta_rejected = DPO log π(rejected) − SFT log π(rejected)
relative_margin = beta × (delta_chosen − delta_rejected)
```

delta 是同一条回答在两份模型下的变化：正数表示序列概率增加，负数表示下降。若 delta_chosen=-1、delta_rejected=-4，相对 margin 仍为正；这表示模型对两条回答的概率都降低，但更强地压低了 rejected。

统计每个 split 的整体均值，并按 category、rejected_source、expected_type 分组；同时统计相对 margin 为正的数量、chosen 概率增加的数量，以及 chosen 下降但相对 margin 提高的数量。逐条保留回答 token 长度和开发题实际生成变化，帮助区分概率与行为。

使用同一固定 base、SFT/DPO 两个 adapter、冻结 tokenizer、response-only 序列 log probability 求和。重新核对偏好文件哈希、DPO checkpoint 与行为比较的模型身份、训练后的 loss/margin/log probability 汇总。训练题分数是训练内诊断，不当成泛化证据。

## 解释限制

来源分组与题型混杂：真实模型错误负例主要来自无括号步骤/自然语言题，合成负例多来自括号或问候题。因此，即便两组变化不同，也不能直接下结论“合成负例优于真实负例”。回答长度也不同，序列求和分数不能当每 token 的质量比较。

模型通过压低某个 rejected 获得正 margin，也可能生成第三种错误回答。开发题实际生成 transition 来自已经完成的行为评测，不重新使用指定 chosen 当生成答案。

## 下一轮继续 SFT 对照：先定义比较条件

建议在相同 v2 起点上，用同一份偏好 train 的 chosen 做继续 SFT，与已有 DPO 对照。先明确控制口径为“chosen 样本曝光和 optimizer 更新次数”，不声称两种训练的 token 或计算预算相同。

| 条件 | 继续 SFT | 已有 DPO |
| --- | --- | --- |
| 起点 | 冻结 v2 | 冻结 v2 |
| 训练 prompt/chosen | 同一份28条 | 同一份28条 |
| rejected | 不用于 SFT loss | 用于 DPO loss |
| 更新次数 | 计划28 | 已完成28 |
| 每步样本 | 1条chosen | 1对chosen/rejected |
| 顺序 | 复用相同seed=42的randperm | seed=42的randperm |
| 学习率/optimizer | 计划1e-4/AdamW | 1e-4/AdamW |
| LoRA/max_length/clip | 继承v2/192/1 | 继承v2/192/1 |
| 目标 | response-only CE | sigmoid DPO，beta=0.1 |

计划记录实际前向 token、监督/评分回答 token、CPU 时间和保存重载；两种目标梯度尺度不同，即使学习率相同也不代表更新强度相同。此对照检验的是本机、此数据和此配置下的结果，不是方法的通用优劣。

沿用四组开发诊断，报告改善与退步、自然语言、成对一致性。首轮对照后再决定是否调整一个变量；没有最终未查看测试集前，仍只报告开发结果。本步仅完成诊断和对照方案，尚未运行这份继续 SFT 实验。

## 阅读与运行

1. 本文和结果 report.md。
2. `evaluation/preference_diagnostics.py`：summarize_preference_changes 与 grouped_preference_changes。
3. `scripts/diagnose_dpo_preferences.py`：模型身份核对、逐对评分、汇总复现。
4. 结果 `pair_scores.jsonl`：从一个相对改善但 chosen 下降的样本开始查看。

在项目根目录运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\diagnose_dpo_preferences.py --dpo-run artifacts\checkpoints\dpo-lora-20261009-205045-21dd3596 --behavior-run artifacts\evaluations\sft-dpo-20261010-150134-fc7e5e9a
```

输出到新建的 `artifacts/evaluations/dpo-preference-diagnostic-时间-编号/`。3项新增测试检查“chosen下降但相对改善”、分组数量和非有限值拒绝；完整78项测试通过。

## 本次结果

结果目录：`artifacts/evaluations/dpo-preference-diagnostic-20261010-181453-22baa0cd`。对38条偏好对分别计算两模型的序列分数，共76次两序列前向；重现训练后 loss、relative_margin、chosen/rejected 平均 log probability，误差在1e-5容差内。

| split | 相对改善 | chosen 概率增加 | chosen 下降但相对改善 | chosen Δ均值 | rejected Δ均值 |
| --- | --- | --- | --- | --- | --- |
| train | 25/28 | 16/28 | 9/28 | -0.570448 | -6.644581 |
| development | 9/10 | 5/10 | 4/10 | -0.178742 | -6.227544 |

整体相对偏好改善伴随着 rejected 概率更大幅下降；不能把25/28或9/10理解成正确答案绝对概率都增加。

| 训练类别 | chosen Δ均值 | rejected Δ均值 | 相对改善 |
| --- | --- | --- | --- |
| right_grouped | +0.168521 | -8.598712 | 6/6 |
| left_grouped | +0.306095 | -10.532454 | 6/6 |
| right_plain | -1.186339 | -2.541384 | 6/6 |
| left_plain | -1.941528 | -1.908568 | 3/6 |
| greeting | -0.013260 | -11.140387 | 4/4 |

括号题 chosen 平均概率上升，无括号题平均下降。left_plain 是当前训练内的薄弱组，平均相对 margin=-0.003296，开发该组也只有1/2相对改善、两题实际生成仍错。这里的样本小，类别与措辞绑定，不能证明模型对所有无括号表达式都退步。

例如 `Add 37 to the product of 16 and 10.` 的正确表达式是 `37+16*10`：chosen log probability 变化为 +0.077764，rejected 为 -6.895372，relative_margin=+0.697314，但实际生成仍错。指定两条回答的概率比提高，不保证正确回答成为贪心输出。

来源分组中，训练合成负例15/15相对改善，真实错误负例10/13相对改善；两组对应不同题型分布，不能据此判定数据来源的优劣。下一步按上表固定条件，运行chosen-only继续SFT对照；当前只完成方案，没有启动新的训练。
