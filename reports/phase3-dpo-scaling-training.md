# Phase 3：v2 DPO 规模对照训练

日期：2026-10-16。

## 本步范围

接入显式训练文件选择，运行三条CPU DPO分支，完成固定答案评分、参数冻结/更新检查、保存重载，以及对照条件核对。本步没有执行模型生成评测，不能据此报告实际工具调用通过率。

| 分支 | 配置 | 训练文件 | 独特偏好对 | 更新次数 | 遍数 |
| --- | --- | --- | --- | --- | --- |
| A | configs/dpo_v2_small_28.json | train_small.jsonl | 28 | 28 | 1 |
| B | configs/dpo_v2_small_168.json | train_small.jsonl | 28 | 168 | 6 |
| C | configs/dpo_v2_large_168.json | train.jsonl | 168 | 168 | 1 |

三者都来自`data/preferences/dpo_v2/`，共用56对development；policy/reference均从冻结v2 SFT adapter初始化。它们独立运行，B不是从A保存的adapter接着训练，而是重新初始化后完整执行168步，便于核对前28步一致。

A→B检查相同小集的更多更新/重复曝光；B→C固定总对数曝光与更新次数，比较覆盖更多family的效果。B每条训练样本使用6次，C每条使用1次。不可能同时固定总曝光次数和每条独特样本的曝光次数。

## 关键代码变化

`src/agentic_post_training/data/preferences.py`新增`load_preference_splits(folder, manifest, train_file)`：

1. `train_file`必须被manifest固定，且不能是development.jsonl。
2. 校验manifest全部数据文件哈希，即便其中一个文件本轮没有用于训练，也不能悄悄变化。
3. 文件路径必须留在数据目录，加载选中的训练文件与固定开发文件。

`scripts/train_dpo.py`使用`config.setdefault("train_file", "train.jsonl")`保留旧配置默认行为。v2还核对完整组合设计、旧v1文件身份及444条受保护目标内容。训练目标、LoRA配置、优化器、reference冻结方式、序列概率计算没有改变。

新增逐步日志：`epoch`、chosen/rejected回答token、非padding输入token、padding后输入token；新增summary：实际训练文件、训练循环耗时、token合计、初始参数摘要及训练脚本/核心trainer哈希。输入token包括两条序列各自的system/prompt；回答token只计loss_mask标记部分，包括EOS。`grad_norm`沿用裁剪前范数，因此日志中大于1不表示裁剪失效。

`epoch`按训练样本数划分，每轮使用独立生成器的下一次randperm；小集168步是六次遍历，不是永远按同一个排列重复。保持seed=42、lr=1e-4、beta=0.1、max_length=192、max_grad_norm=1.0、CPU线程2；AdamW且weight_decay=0。各分支有独立优化器，reference始终固定为原v2 SFT，不随轮次更新。

## 运行结果

三组训练与汇总核对已完成，86项测试通过（唯一警告来自已有tiny PEFT保存测试）。项目根目录：`C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab`。

训练产物：

- small_28：`artifacts/checkpoints/dpo-lora-20261016-141339-249a77f0`。
- small_168：`artifacts/checkpoints/dpo-lora-20261016-142317-5d529ace`。
- large_168：`artifacts/checkpoints/dpo-lora-20261016-145728-bc1fc34c`。

汇总目录：`artifacts/evaluations/dpo-v2-scaling-training-20261016-152209-97915019`。三组reference均冻结无梯度，policy冻结参数不变、LoRA有更新；保存重载logits最大差均为0.0。

本轮不生成回答，relative preference accuracy不是生成正确率。

| 分支 | 训练对数 | 更新次数 | 训练后loss | 开发后loss | 开发相对偏好改善 | 开发chosen Δ | 开发rejected Δ | 更新耗时/秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_28 | 28 | 28 | 0.547064 | 0.575274 | 45/56 | -1.790009 | -4.774380 | 78.52 |
| small_168 | 28 | 168 | 0.212014 | 0.258704 | 53/56 | -11.897777 | -29.692413 | 785.98 |
| large_168 | 168 | 168 | 0.251578 | 0.250463 | 55/56 | -13.847387 | -32.363785 | 882.57 |

控制核对通过：固定配置、基线、初始参数、代码和数据身份一致；开发初始评分一致；A/B前28步的样本顺序及数值指标一致。reference冻结无梯度，policy仅LoRA更新，三组保存重载检查通过。

| 分支 | chosen监督token | rejected监督token | 非padding输入token | padding后输入token |
| --- | --- | --- | --- | --- |
| small_28 | 750 | 798 | 4762 | 4822 |
| small_168 | 4500 | 4788 | 28572 | 28932 |
| large_168 | 4500 | 4788 | 28604 | 28964 |

token与耗时只统计policy训练循环，未计reference缓存、训练前后评分、模型加载/保存和重载；固定步数不等于完整运行计算量完全相同。

A→B改变同一小集的重复更新，B→C在相同总曝光下改变覆盖。单种子、单次小子集、合成负例与共享模板限制结论；尚未进行新开发集基线生成和三分支逐题行为对照。


## 如何理解这次结果

A→B使用相同28对数据，更多更新使开发相对偏好改善由45/56增至53/56，开发loss明显下降。B→C保持168步，扩充覆盖使开发loss从0.258704降至0.250463，相对改善从53/56增至55/56，增量较小。

但开发chosen平均log probability相比原SFT分别下降1.790009、11.897777、13.847387；rejected下降更多。固定答案偏好改善不能证明正确答案的生成概率全面提高，更不能把55/56当生成通过率。当前证据不支持断言“原先问题主要由样本少导致”，也不能用新v2与旧v1直接作纯规模归因。

B/C的回答token完全一致，padding输入仅相差32token，训练更新预算较接近；完整运行还需分别缓存reference和评分不同大小训练集，因此总成本不同。本机运行时间存在波动，不把三行耗时当通用性能benchmark。当前教学实现逐对评分、没有跨分支复用reference缓存，本轮耗时较长；已经训练完成，阅读结果无需再运行三次训练。


## 核对与指标边界

三组都必须验证：reference参数不变且无梯度，policy基础参数不变、LoRA有更新，adapter保存后用新base重载logits一致。

`scripts/summarize_dpo_scaling.py`进一步核对三组：固定配置、模型版本、基线、初始参数、数据和核心代码身份相同；开发初始评分一致；两条小集分支前28步顺序及loss/margin/log probability/梯度范数一致；每轮覆盖全部样本一次，曝光数与token账目符合日志。它只读取训练记录，不生成模型回答。

相对偏好accuracy统计相对冻结reference的margin>0。初始化policy/reference相同，平局按严格>0为false，因此初始accuracy为0；这不是原SFT生成准确率为0。chosen/rejected对数概率均为回答token的和，包括EOS，不是平均token交叉熵。

大/小训练集不同，训练loss均值不是同一批样本的均值，不能只比较train loss判断覆盖效果。开发集相同，固定答案评分可对照，但chosen下降、rejected下降更多时相对偏好仍能提高；仍需生成评测。

训练token/耗时只记录policy更新循环，不包括reference缓存、前后评分、加载与保存重载；尤其大集要评分更多训练样本，完整运行成本不相同。没有将同一步数描述为相同完整计算预算。

## 阅读顺序与运行方式

1. `src/agentic_post_training/data/preferences.py`的`load_preference_splits`及新增测试`test_preference_selection.py`。
2. 三份配置：重点比较`train_file`和`max_steps`。
3. `scripts/train_dpo.py`新增的选择、v2审计和日志部分；训练目标部分复用旧代码。
4. `scripts/summarize_dpo_scaling.py`的同起点、前缀、曝光与token核对。

项目根目录使用本项目虚拟环境，按需单独运行，每次会创建新checkpoint目录：

```powershell
& ".\.venv\Scripts\python.exe" scripts\train_dpo.py --config configs\dpo_v2_small_28.json
& ".\.venv\Scripts\python.exe" scripts\train_dpo.py --config configs\dpo_v2_small_168.json
& ".\.venv\Scripts\python.exe" scripts\train_dpo.py --config configs\dpo_v2_large_168.json
```

## 下一学习步骤

读完本轮代码后，先收集冻结v2在新56题上的实际生成基线，再在相同设置下比较A/B/C。原90题诊断继续保留，按数据组分别汇总，不合并成独立测试准确率。旧比较/概率诊断脚本仍有默认读取train.jsonl的假设，不能未经改动直接用于新小集分支的训练暴露或train指标重现；下一步需正确读取`selected_data_files`或`config.train_file`。

新开发集的48条算术记录还可按provenance的`contrast_id`组成24对，检查括号/无括号是否同时通过；8条问候单独汇总。不能直接把新category字符串交给旧括号分组逻辑，因为新category没有旧的number_set/wording/form三级结构。

重新汇总现有结果可运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\summarize_dpo_scaling.py --runs-file artifacts\logs\dpo-v2-scaling-20261016-141008-2f738fcf\runs.json
```

当前仍只有一个随机种子和一次小子集抽样，数据全部合成、模板共享，开发数字长度也有变化；没有最终未查看测试，不证明普遍泛化，也不把旧v1和新v2之间差异归因于纯样本数量。
