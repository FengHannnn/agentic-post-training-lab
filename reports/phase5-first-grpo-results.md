# Phase 5：首次真实 GRPO 更新结果与独立复核

日期：2026-10-27。首轮有限真实更新完成，工程检查通过；严格行为指标未观察到提升，冻结 SFT 不晋升。尚未实现多次刷新 rollout 的在线策略循环，Phase 5 不在这里宣称全部完成。

## 产物与复核范围

用户回传 grpo-first-review-20261027.zip，SHA256=df068ba95ba4b03da76bd77e2bbff8bcbc32921c69f73e26e9ced389cf4e1e2f。共48个文件、解压约10.97 MB，归档于 artifacts/checkpoints/grpo-first-20261027-01。

本地检查46份产物哈希、26份源码快照及与当前可信源码的一致性，确认原始1152条来源、训练计划及checkpoint联合输入身份未变。核对全部16步的组ID、回答/token数、静态reward、梯度和停止界，以及16/17/528计数。读取两个optimizer.pt均使用weights_only=True，核对第1步/第16步、120个LoRA参数名、Adam步数与动量张量有限。

重新解码和评分更新前后各104条预测，共208条；按固定题目ID重算整份行为比较，与云端summary一致。逐步loss由microbatch的FP32标量加权累加，与policy_loss+beta*weighted_k3重组存在最大1.7877e-8的舍入差，使用1e-7容差核验；这不是对原始概率门或训练停止界的修改。

另在CPU只加载原始基座与原始/第1步/最终adapter，不执行模型前向或更新，独立重建全参数摘要及冻结基础参数摘要。local_audit.json与weight_audit.json归档于 artifacts/evaluations/grpo-first-review。云端reference运行状态和恢复探针的GPU计算未在本地重跑，依据已绑定源码及记录；不能把权重重建称为GPU数值重演。

## 真实更新与参数身份

| 项目 | 结果 |
| --- | ---: |
| GPU | NVIDIA L20 |
| 来源 / 候选优化组 | 144 / 62个混合组 |
| 候选回答 | 496 |
| 候选逻辑optimizer步 | 16 |
| 含被丢弃恢复探针的实际optimizer调用 | 17 |
| 模型backward调用 | 528 |
| 提前停止 | 无 |
| 发生变化的LoRA张量 | 120/120 |
| 最终LoRA最大元素变化（相对原SFT） | 0.0001601102 |
| 基础参数摘要 | 前后相同 |
| 原始 / 最终完整参数摘要 | 9b77c99e… / 853c4027… |

CPU重建的完整摘要与云端记录逐字匹配。第1步LoRA最大元素变化约1.00005e-5；原始/第1步/最终冻结基础摘要均为 e72fa38da3ed620a2f0879d8f11b0a6fe908e965f058b95b52738417bae1a4b6。它提供了真正更新adapter而基础权重保持相同的产物证据，不仅是读取policy_changed=true。

reference在云端保持冻结、无梯度且摘要不变，初始reference/policy与旧分数最大差为0。第1步checkpoint恢复后的下一步，adapter与Adam状态误差均为0，损失相同；最终adapter重新加载后，8条logp误差为0、8个greedy token检查全部一致，完整摘要匹配。

## 优化监控

所有LoRA梯度范数有限且非零，裁剪前范围0.18446–0.72163，没有触及最大范数1。sampled k3最高0.00147061，小于0.05停止界；ratio超出裁剪区间的比例最高约1.75%，小于50%停止界。少数token ratio最低约0.48064、最高约1.31669，再次说明PPO clip是目标激励约束，不把实际ratio硬限制在[0.8,1.2]。

sampled k3仍是存储历史上的近似惩罚，不是当前完整策略KL的无偏估计。每步组与回答不同，loss跨步变化不能直接当作单一固定目标的收敛曲线；日志中reward来自旧rollout，不能称为更新后reward。82个零方差组未优化但原始数据保留，训练分布选择已披露。

## 同条件生成对照

固定FP32、eager、相同模板、greedy、最多64 token，前后各生成48个训练诊断输入和56个开发输入。开发集是已暴露诊断数据，不是最终盲测；包含48道算术与8道问候。

| 指标 | 更新前 | 更新后 |
| --- | ---: | ---: |
| 训练诊断严格通过 | 28/48 | 28/48 |
| 训练诊断语义通过 | 35/48 | 36/48 |
| 开发严格通过 | 36/56 | 36/56 |
| 开发语义通过 | 44/56 | 44/56 |
| 新增JSON/schema/route退化 | — | 0 |
| length截断 | 0 | 0 |

严格指标没有提升，不能报告GRPO优于SFT。训练诊断只多1条语义正确：原表达式 (41*19)/75 改为 (41*19+75)，计算结果符合该题，但与严格目标AST仍不同，reward仍为0。开发有2条输出文本变化，严格状态均未改变；其中一条仅多余括号变化，不能作为能力提升。

工程验收true、行为安全门true、baseline_promoted=false。安全门通过只表示本轮没有新增结构退化且开发严格总数不降，不等于已证明收益。

## 耗时与成本

作业309.83秒，约5分10秒。按记录时价2.18元/时估算约0.1876元；不包含开机、空闲、存储或计费取整，不是完整平台账单。错误字段为空，无自动重试。

## 当前完成状态与后续方向

本轮已完成真实LoRA反向/更新、独立冻结reference、sampled KL惩罚、停止检查、完整Adam恢复验证、最终adapter重载，以及前后逐题行为比较。没有修改原始reward或旧跨引擎失败门，没有自动晋升候选。

首轮回答了“更新链路是否可信”，结果是通过；没有证明严格行为收益。当前只执行一次固定old-policy rollout上的有限minibatch更新，没有更新后重新采样。若继续Phase 5，下一步应从当前候选策略生成新的rollout、记录新的old-policy版本，reference继续固定原SFT，再做预先限定的下一策略迭代，而不是把同一1152条旧回答无限重复或立即提高学习率追逐正结果。

后续实验仍需保留SFT与当前候选，固定开发对照与停止规则。多轮数据刷新、策略/adapter版本治理及最终有限比较尚待实现；本报告不提前标记完成，也不把本轮无提升解释为算法必然无效。

本轮未改运行源码，未重复此前通过的305项测试。本地做的是输入/权重/状态/预测的完整产物复核，未重新训练真实模型。

## 代码阅读顺序

1. configs/grpo_first_update.json：62个混合组的一轮顺序、16步上限、参考/停止参数。
2. src/agentic_post_training/trainers/grpo.py → completion_objective/accumulate_minibatch/guarded_step：sampled惩罚、token分母、梯度与更新。
3. scripts/train_grpo_first.py → train：第1步checkpoint、第2步探针/恢复，16个逻辑步与17次实际调用的区别。
4. trainers/grpo.py → restore_optimizer_checkpoint/compare_optimizer_snapshot：模型参数以外的Adam状态与计数。
5. train_grpo_first.py → compare_evaluations：严格指标要求EOS、逐题结构回归、工程和行为判定分开。
6. tests/integration/test_grpo_first_update.py 与 unit/test_grpo_first_comparison.py：真正tiny LoRA更新/恢复和总分不变仍有结构退化的边界。
