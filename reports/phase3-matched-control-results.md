# Phase 3：38同题三组对照执行与结果

## 结论与实际生成结果

三次38步训练、共同生成与不加载权重的重算审计已全部完成。真实负例DPO没有超过本轮继续SFT；两组DPO均未满足预先额外收益条件。三组均新增同一条旧题坏JSON，恢复质量门也未通过，冻结SFT基线保持不变。按预定停止条件收尾本轮，不追加训练或超参数尝试。

| 数据组 | 冻结SFT | 继续SFT | 合成负例DPO | 真实负例DPO |
| --- | --- | --- | --- | --- |
| 新56严格 | 36/56 | 36/56 | 36/56 | 36/56 |
| 新56语义 | 44/56 | 44/56 | 44/56 | 44/56 |
| 旧工具严格 | 19/20 | 20/20 | 19/20 | 20/20 |
| 旧括号严格 | 33/48 | 34/48 | 33/48 | 34/48 |
| 空格控制严格 | 12/12 | 12/12 | 12/12 | 12/12 |
| 旧偏好严格 | 7/10 | 7/10 | 7/10 | 7/10 |
| 全组JSON合法 | 146/146 | 145/146 | 145/146 | 145/146 |

新开发算术数值36/48、算术成对数值16/24、问候8/8，四个模型相同；natural数值仍4/16，symbolic与sequence均16/16。各模型无生成上限命中。每模型146条记录含145个唯一输入，不能将重复、相关family或已查看开发题视作独立最终测试。

真实DPO与继续SFT的新56条预测完全相同。全部146条中只差1条文本：`bracket-unseen_small-natural-right_plain`，题目`Add 4 to the product of 3 and 7.`，期望`3*7+4`；继续SFT输出`3*(7)/2`，真实DPO输出`3*(7)`，两者都错。二者所有组严格/语义改善和退步均为0。真实DPO相对合成DPO在工具与括号各改善1题、无严格退步，但这两项收益在继续SFT中同样出现。

三组共同新增的结构退步是`bracket-seen_large-sequence-left_plain`：`Multiply 12 by 3 first, then add that result to 41.`。冻结SFT为合法JSON但表达式`12*3`；三组输出均含`"expression":"12*3+"41"`，成为非法JSON。它是每模型1条记录的结构退步，JSON/schema/route三个检查同时受影响，并非三个独立错误。端到端均为both_fail，因此只看通过数会遗漏恶化。真实负例中有两条坏JSON，也未在本轮阻止该回归；结构风险同样出现在继续SFT，不能全部归因于纯DPO。

## 发现、限制与收尾

本轮在相同输入、chosen、曝光顺序和更新设置下，替换26条负例文本，没有取得新开发集额外收益；真实DPO保留了继续SFT的旧组成绩，优于本轮合成DPO的两条旧题成绩。偏好概率变化与贪心生成收益仍不同步。

本轮两组纯DPO的chosen均值都提高，也未重现原C的新集4/56严重崩溃。与原C同时改变了数据子集、步数和LR，不能把稳定性恢复单独归因于其中一项。两组内部负例对照控制了输入和正例，但具体负例的内容、长度与难度也随来源变化，不能将效果归于“真实”标签本身。

38步、1e-5可能更新不足，真实组最终平均原始chosen/rejected分差仍为-7.56；数据分布狭窄且数字范围存在外推、模型/LoRA容量受限也可能影响学习。这些仍是可能解释，未做独立消融。结果不能证明增加真实多样数据没有价值，也不证明DPO普遍无效。当前工程闭环完整，模型性能提升目标仍未达到；不因结果未胜出而继续无边界调参。

## 实验范围

按已冻结的`data/preferences/dpo_v3_matched_errors/experiment_plan.json`执行三次训练：chosen-only继续SFT、合成负例纯DPO、真实负例纯DPO。计划SHA256为`a3520bf40a59e07e058b0cb0804571e4d999b197c3ca36de0b110b5601465405`。不追加LR、beta、anchor或步数尝试，不自动替换基线。

三组从同一冻结v2 SFT权重开始，各38步、38个chosen各曝光一次、相同ID顺序，LR1e-5、seed42、clip1、AdamW无weight decay。两组DPO beta0.1、chosen NLL权重0；只有负例来源不同，12条相同、26条改变。单种子、12个相关family、自然语言偏多、无问候训练、开发数据已反复查看等限制保留。同chosen曝光不等于同token算力。

## 入口与核验

新增入口不改旧训练源码，避免破坏历史实验身份：

1. `scripts/matched_control_common.py`：固定计划哈希，检查文件、输入/chosen一致性与语义/隔离/token审计；核对三组逐步ID、顺序、chosen token曝光和实际LR；按冻结规则判断恢复与DPO额外收益，并逐题检查新增结构/上限退化。
2. `scripts/train_matched_control.py`：复用既有`train_steps`/`train_dpo`，创建独立policy起点、固定reference评分，记录token量与更新耗时，验证基础参数/reference不变、LoRA更新、adapter保存重载。初始指标明确由相同权重reference的评分推导，并非逐臂重新运行完整policy评分。结束后检查三组曝光一致性。
3. `scripts/compare_matched_control.py`：核验训练与历史身份，先用三题新生成探针确认冻结SFT历史输出可复现，再分别加载三份独立adapter，用相同生成设置评估五组；六个模型对照各保存五份changes，共30份。`--audit-only`不加载权重，重算全部保存记录、指标、质量门和变化文件。
4. `tests/unit/test_matched_control_checks.py`：7项顺序/token/LR与收益门测试，检查持平不能称DPO收益、both_fail不能掩盖新增结构或上限退化。既有tiny PEFT集成覆盖实际SFT/DPO更新、冻结和保存重载。

全套125项测试通过，1条已有tiny PEFT保存提示。另用四份相同历史SFT记录自检评估汇总：六组对照、30份changes可生成，恢复门通过而DPO额外收益为false。

## 复用与恢复边界

训练目录为`artifacts/checkpoints/matched-control-v3-20261017-01`，各模型在同名子目录。`reference_log_probs.pt`保存两组各38对固定分数，不是模型权重。初始参数SHA256应完全相同；训练后adapter不同，优化器状态不保存。

训练入口的`--resume --output <目录>`只能跳过已完成且身份正确的臂；不会重新训练已有部分日志但未完成的臂，以免静默违反三次训练预算。评估入口支持核验已保存的有序前缀后继续生成，避免因中断重复生成。重审入口不重新生成。

数据清单中的`training_started=false`和`runner_adaptation_required=true`保留其冻结当时的记录，不改写原数据清单；实际训练执行状态与代码版本记录在新的run_manifest/summary和Git中。

## 三组训练结果

三组38步已完成，逐步ID、顺序、chosen token曝光及LR核对一致。初始policy/reference参数SHA256全部为`9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb`。各组基础参数/reference不变、reference无梯度、LoRA有更新、保存重载通过，logits最大差均为0。

| 组别 | chosen回答token | rejected回答token | 实际输入token | padding后输入token | 更新耗时秒 |
| --- | --- | --- | --- | --- | --- |
| 继续SFT | 1108 | 0 | 3300 | 3300 | 64.24 |
| 合成负例DPO | 1108 | 1116 | 6608 | 6628 | 120.88 |
| 真实负例DPO | 1108 | 1043 | 6535 | 6602 | 133.21 |

耗时只指更新循环，不含reference评分、训练后评分、保存重载或生成评估；CPU单次测量不据此比较算法吞吐。

继续SFT训练题chosen token平均NLL从0.321789降至0.285516。两组DPO训练对指标：

| 组别 | DPO loss 前→后 | chosen序列logp变化均值 | rejected序列logp变化均值 | 相对margin改善 |
| --- | --- | --- | --- | --- |
| 合成负例DPO | 0.693147→0.646207 | +0.426050 | -0.544117 | 33/38 |
| 真实负例DPO | 0.693147→0.648045 | +0.879285 | -0.050737 | 37/38 |

本轮两组chosen均值都上升，没有重复原C的chosen均值大幅下降现象。不同负例集合的loss/相对accuracy不能直接当作统一难度的性能排名。真实负例组最终平均原始policy margin仍为-7.563507，即chosen平均序列分数仍低于rejected；37/38只表示相对于reference的差改善，不表示37条生成正确，也不表示37条chosen已经胜过rejected。生成结论另看共同评估。

## 产物与重审

训练目录：`artifacts/checkpoints/matched-control-v3-20261017-01`。生成目录：`artifacts/evaluations/matched-control-v3-20261017-01`。原始`summary.json`、20份预测JSONL、30份`*_changes.json`、3个基线重现探针、`verification.json`均已保存。代码实现提交`0bf89a7`；报告与交接另行提交。

真实新生成435次，另3次冻结SFT重现探针，共438次；3个重复输入记录使用同一模型缓存。冻结SFT146条复用经过身份/版本/源码/评分检查的历史记录，没有伪称全部重新生成。总584条保存记录已重算，六组对照与30份changes一致，训练模型/日志/源码及固定分数缓存身份核验通过。

```powershell
& ".\.venv\Scripts\python.exe" scripts\compare_matched_control.py --training-run artifacts\checkpoints\matched-control-v3-20261017-01 --output artifacts\evaluations\matched-control-v3-20261017-01 --audit-only --require-admission
```

本次实际返回退出码2：重算审计通过，但DPO额外收益质量门未过；`verification.json`的`artifact_audit_passed=true`与`any_dpo_added_value=false`必须一起看。去掉`--require-admission`时，记录一致即可返回0，不能解释为模型达标。无需重跑训练或生成。
