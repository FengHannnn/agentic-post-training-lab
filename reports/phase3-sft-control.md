# Phase 3：chosen-only 继续 SFT 对照

日期：2026-10-11。

## 为什么做这个对照

首轮 DPO 的相对偏好改善，真实生成有改善也有退步。我们需要区分：正确答案这批新增监督数据本身是否有效，以及同时使用 rejected 的 DPO 目标是否提供额外收益。该实验只回答当前小规模、单种子、固定参数设置下的问题；不证明 SFT 或 DPO 的普遍优劣，也不直接证明28对样本是否是主要瓶颈。

三条分支都是同一个冻结 v2 起点：原 v2 SFT 不更新；v2→DPO 使用 chosen/rejected；v2→继续 SFT 只学习 chosen。继续 SFT 从 v2 adapter 加载，而不是从 DPO adapter 开始，也不重新初始化 LoRA。

## 固定条件与差异

- 同一基础模型版本、冻结基线、tokenizer、28条训练 prompt 和 chosen。
- 使用 DPO 训练摘要中的配置，并逐步核对原 train_log 的样本索引；seed=42、28次更新、batch_size=1、AdamW、weight_decay=0、lr=1e-4、梯度裁剪1.0、max_length=192。
- 只更新已有 q_proj/v_proj 的 LoRA，230400个可训练参数；基础权重冻结。
- SFT 是回答 token 的平均交叉熵；DPO 是 chosen/rejected 的序列概率偏好目标。相同学习率不代表相同梯度尺度。配置保留的 beta 只用于追踪 DPO 设置，不参与 SFT 计算。
- chosen 使用次数和更新次数相同，输入 token 数、计算量、梯度尺度不同；不是等计算预算实验。
- 开发集只评分，未训练。90题仍为已查看开发诊断，部分含已见 SFT 表达式，不是最终测试集。

## 训练与保存检查

训练集回答 token 加权交叉熵：0.102634→0.041592；开发集：0.096143→0.044478。该 CE 与 DPO 的 loss 定义不同，不能比较两者 loss 数值大小判断方法优劣。

28次更新用时34.84秒（本机墙钟时间，包含逐步日志；不是CPU利用率或完整运行耗时）。训练输入共2277 token，回答监督共714 token，包括EOS，不计提示词损失。原 DPO 未记录同口径训练墙钟时间，不补造耗时比较。

确认基础冻结参数不变、LoRA参数有更新、样本顺序与DPO完全相同。保存后使用新加载的同版本 base＋新adapter重载，探针 logits 最大差为0.0。保存的是完整更新后adapter和tokenizer，没有优化器状态。

训练目录：`artifacts/checkpoints/continued-sft-20261011-181322-52e2bc03`。

## 行为评测

评测设置：同一模板、CPU float32、贪心、单beam、最多64个新token。原 SFT/DPO 复用上一轮已保存预测，本轮继续 SFT 新生成90题。复用前核对原模型摘要和adapter哈希、基线与评测文件哈希、三份tokenizer的vocab/模板/special tokens，以及各题token IDs；重算历史汇总并比对，核对逐题输入/目标与固定数据文件一致。

结果目录：`artifacts/evaluations/sft-control-20261011-181518-0ef09401`。

| 组别 | 原 v2 SFT | DPO | 继续 SFT | 继续 SFT 相对原 SFT 改善/退步 | 相对 DPO 改善/退步 |
| --- | --- | --- | --- | --- | --- |
| 工具请求 | 19/20 | 20/20 | 19/20 | 0/0 | 0/1 |
| 括号主诊断 | 33/48 | 33/48 | 33/48 | 3/3 | 3/3 |
| 空格控制 | 12/12 | 11/12 | 12/12 | 0/0 | 1/0 |
| 偏好开发生成 | 7/10 | 8/10 | 8/10 | 1/0 | 1/1 |

括号数值/路由通过：原SFT33/48、DPO33/48、继续SFT34/48；成对数值通过：11/24、10/24、12/24。空格成对通过：6/6、5/6、6/6。这些成对指标与严格AST通过不是同一种指标。

继续SFT的JSON/schema/route：工具20/20、括号47/48、控制12/12、偏好开发10/10；原SFT和DPO各组均全部合法。继续SFT无生成上限命中。新增无效JSON是 `bracket-seen_large-sequence-left_plain`，输出表达式片段为 `12*3+"41"`，破坏了JSON字符串。该题原SFT也未端到端通过，所以它未计入严格通过的退步，但JSON指标发生退步。

继续SFT修好了 `Add 37 to the product of 16 and 10.`，输出 `37+16*10`；DPO修好的是另一道 `10*(37+16)`。两个开发组同为8/10，不意味着修好了相同的问题。

在自然语言问法中，继续SFT修好“先乘再加”，同时把部分“乘以两个数的和”错写成无括号表达式。相对DPO，它保住了 `3*41+12` 的一些输出，但失去了 `2*(6+5)` 的一些正确输出。详细三列输出见结果目录 report.md，不仅看总体通过数。

## 当前能得出的结论

这批小数据对两种目标均有局部收益，也均有退步。当前设置下，DPO在工具组更好，继续SFT在空格与括号数值成对指标更好；两者没有全面胜出。直接chosen监督没有稳定解决自然语言顺序理解，所以当前问题不能只归咎于DPO通过降低rejected改善margin。

28对样本规模/覆盖仍是合理的待检验假设，本轮没有改变样本量，不能证明它是主要原因。下一轮若继续改进，应固定现有分支，构造题型与措辞更均衡、更多数字组合的数据，保留旧开发集并增加按family隔离的新开发集，用相同训练目标检查扩大数据后的变化；扩大数据的步数/曝光也需明确，避免把更多更新带来的效果全归因于样本量。先按学习节奏阅读本轮代码，再决定下一轮数据实验。

## 阅读顺序

1. `scripts/train_sft_control.py`：数据与起点核对、顺序匹配、训练、冻结检查、保存重载。
2. `src/agentic_post_training/trainers/sft_lora.py` 的 `evaluate_loss`、`train_steps`：复用原SFT目标与更新过程。
3. `scripts/compare_sft_control.py`：历史记录核对、新模型生成、逐题对照与报告。
4. 本轮 `report.md` 和各组 `*_vs_control_changes.json`：定位改善与退步。changes使用`baseline_*`与`control_*`区分文件名指定的对照起点和继续SFT；summary也使用baseline/control。

## 运行方式

在项目根目录使用本项目虚拟环境：

```powershell
& ".\.venv\Scripts\python.exe" scripts\train_sft_control.py --dpo-run artifacts\checkpoints\dpo-lora-20261009-205045-21dd3596
& ".\.venv\Scripts\python.exe" scripts\compare_sft_control.py --control-run artifacts\checkpoints\continued-sft-20261011-181322-52e2bc03 --behavior-run artifacts\evaluations\sft-dpo-20261010-150134-fc7e5e9a
```

每次运行创建新目录，不覆盖旧实验；如果重新训练，应将第二条命令的control-run换成新目录。当前78项测试通过，真实训练额外完成参数/保存检查；行为比较复用已有评分与比较函数，不增加重复实现同一逻辑的测试。
