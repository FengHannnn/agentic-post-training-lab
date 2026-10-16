# Phase 3：定位并修复DPO生成退化

日期：2026-10-16。

## 问题与已知证据

原C从冻结v2 SFT出发，用168对偏好训练168步，新56题严格通过从36降到4，48条算术数值通过为0。格式、schema与路由同时退化，不能当作正常的微小指标波动，也不能据此宣称DPO训练成功。

已核对损失正负号、单次causal shift、回答与EOS掩码、PAD不计损失、独立冻结reference、LoRA更新和保存重载；没有发现这些环节的明显错误。LoRA dropout和attention dropout均为0。B/C单adapter定点复现与原生成一致；这只覆盖各两道题。

C固定开发答案的平均序列log probability：chosen从−3.643747降到−17.491135，rejected从−9.124099降到−41.487884。因此相对间隔扩大，但正确序列绝对概率下降。训练前28步chosen每token NLL均值0.178523，最后28步0.718707；这些是不同训练题的在线轨迹，不是固定集前后CE比较。梯度裁剪一直启用，早28步全裁剪、末28步24次裁剪；裁剪不能代替生成质量约束。

标准sigmoid DPO优化chosen与rejected相对reference的对数概率差，不直接要求chosen绝对概率上升。算术正负答案共享JSON/calculator前缀，在确定性teacher forcing下，共同前缀项及其直接梯度会在差值中抵消。后续不同token产生的共享参数更新仍可能改变该前缀。问候负例对比路由，但算术负例只对比运算分组；数据也没有覆盖本次生成出现的全部坏格式与错误路由。这些是机制线索，尚未确定各自的因果贡献。

## 预先声明的单变量修复

保持原C的base、SFT起点、168条训练数据、56条开发数据、seed42、样本顺序、168步、学习率1e-4、beta0.1、裁剪1、CPU设置完全相同。从原冻结SFT重新出发，不从已退化C继续更新。

仅增加正确答案的回答token平均负对数似然：

```text
L = L_DPO + alpha * ( -log pi(chosen | prompt) / chosen_response_tokens )
alpha = 1
```

分母只含causal shift后被loss_mask选中的chosen token，包含终止符，不含prompt或padding。DPO继续使用序列log probability总和。chosen NLL复用同一前向图，不额外增加chosen曝光或前向次数。alpha=0保留原纯DPO的损失与梯度。alpha=1不代表两项尺度相等：DPO使用序列总和，chosen NLL使用token平均；算术回答约30 token，权重需要结合这个约定解释。

这是一项DPO＋chosen监督的混合目标；通过修复不能声称原纯DPO有效，也不能声称DPO项比纯SFT有额外收益。TRL官方支持组合sigmoid与sft目标，见[官方DPO说明](https://huggingface.co/docs/trl/main/en/dpo_trainer#multi-loss-combinations)；本项目明确记录自身token平均和权重约定，不依赖安装TRL。

## 生成前声明的验收条件

1. 新56题严格至少36、数值至少44，恢复到SFT水平；JSON/schema/路由56条全部合法正确。
2. 原工具、括号、空格、旧偏好开发严格通过分别至少19/20、33/48、12/12、7/10。
3. 新56题严格或数值至少一项超过SFT。
4. 起点、数据、逐步样本顺序、token曝光、冻结/更新/重载全部核验；实际生成按原CPU greedy64设置。

若只是比原C的4/56好，不能算完整修复；若恢复旧能力但新集未改善，须明确报告。全部题仍是已查看开发诊断，没有最终未查看测试，不通过生成后挑选子集改变验收标准。

## 实现与阅读顺序

1. `trainers/dpo.py`新增anchored_objective及可选chosen_nll_weight；train_log明确区分总loss、dpo_loss和chosen_nll。
2. `configs/dpo_v2_large_168_anchor.json`只修改声明的目标项和实验名称。
3. `scripts/train_dpo.py`记录目标；before/after的loss仍是纯DPO诊断，不能与训练总loss混称。
4. `scripts/compare_dpo_repair.py`核验原C与修复训练配置/起点/逐步曝光，复用已核验SFT/C记录，单独加载修复adapter生成145种输入，保存146条记录和逐项对照。
5. `test_dpo_anchor.py`检验零权重一致性、共同前缀梯度抵消及监督项的修复方向、非法参数；tiny PEFT集成同时覆盖纯DPO与混合目标的更新/冻结/重载。

## 本轮执行结果

第一轮alpha=1：训练`artifacts/checkpoints/dpo-lora-20261016-180833-e336ae0b`，生成评测`artifacts/evaluations/dpo-anchor-20261016-182453-d989d3f1`。

| 数据组严格通过 | SFT | 原C | alpha=1 |
| --- | --- | --- | --- |
| 新开发 | 36/56 | 4/56 | 31/56 |
| 工具 | 19/20 | 2/20 | 19/20 |
| 括号 | 33/48 | 1/48 | 29/48 |
| 空格 | 12/12 | 0/12 | 12/12 |
| 旧偏好开发 | 7/10 | 1/10 | 5/10 |

第一轮新算术数值23/48、成对5/24、问候8/8；JSON53/56、schema与路由45/56，所有组无生成上限命中。新开发按措辞的算术数值为symbolic13/16、sequence6/16、natural4/16。五项预先验收条件全部未通过，因此明确记作未修复。

第一轮reference不变、仅LoRA更新、重载logits最大差0；逐步曝光/配置/起点控制核验通过。开发chosen平均log probability提高到−2.850504，rejected降到−23.454333，偏好相对改善56/56，纯DPO loss0.327434。仍然不能用均值提高或偏好率100%代替生成验证。

438条记录重新评分、15份预测哈希、10组汇总/变化文件、3份入口/评测源码哈希均一致，见第一轮verification.json。实际新生成145次，同输入复用1次，其余292条SFT/C记录来自已核验历史。第二轮结果待完成后追加。



## 第二轮单变量修复计划

第一轮alpha=1已完成168步，chosen开发平均log probability改善到−2.850504，但初始实际生成已出现schema错误，违反预先声明的完整格式验收条件。保留第一轮完整五组生成与全部结果，不把它替换成成功记录。

2026-10-16 18:27启动第二轮：仍从同一冻结SFT出发、同168对/168步/顺序/学习率/beta，仅把chosen监督权重加强到5；配置`configs/dpo_v2_large_168_anchor5.json`。与原C相比依然只改变监督目标项。验收标准不变。不是从第一轮模型接着训练。

第二轮训练额外保存固定reference分数及其哈希，属于可复核产物，不改变目标或计算顺序。第一轮启动时的训练/比较入口精确源码已保存在其评测目录source子目录，SHA256与当时summary/run_manifest完全一致；后续入口增加可声明权重参数，避免把当前源码哈希冒充第一轮实际运行版本。
