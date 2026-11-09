# V2 首轮 SFT 结果复核

日期：2026-10-30（Asia/Shanghai）。运行：sft-pilot-v1-20261030。

首轮 SFT 完成 63 次更新，模型学会了先读规则和 schema，严格动作格式通过率从 54.2% 提升至 91.7%。4 个开发任务仍全部失败，当前需要补齐净收入 SQL、有效证据依赖和结构化提交。

| 指标 | 原始模型 | 本轮 SFT |
|---|---:|---:|
| 开发任务成功 | 0/4 | 0/4 |
| 故障任务最终成功恢复 | 0/2 | 0/2 |
| 严格动作格式通过 | 26/48（54.2%） | 44/48（91.7%） |
| 工具错误 | 39/48 | 26/48 |
| 已验证业务指标 | 0 | 0 |
| 有效月度 SQL | 0 | 0 |
| 有效 check_result / submit | 0 / 0 | 0 / 0 |
| 总动作轮数 | 48 | 48 |
| 生成 token | 3,659 | 2,210 |
| 首两步读取规则和 schema 的任务 | 0/4 | 4/4 |

严格格式检查的是 JSON 顶层 tool/arguments 协议；工具参数和业务正确性由环境进一步验证。

## 训练完成情况

训练使用 12 个训练任务的 36 条专家/恢复轨迹，形成 252 个动作样本。两遍共 504 次动作曝光，63 次 AdamW 更新；正常达到 epochs_complete。1,089,536 个 LoRA 参数参与训练，日志记录 112 个 adapter 张量发生变化，冻结参数梯度计数为 0。

训练循环耗时 132.51 秒，整个训练与评测作业 274.63 秒；峰值 allocated 显存 6.4437 GiB。前 8 步 batch loss 均值 0.0870，最后 8 步 0.0200；每步使用打乱后的不同动作批次。

保存了 7 个中间检查点和 final。final Adam 状态全部记录 step=63，进度 cursor=504。最终 adapter 摘要：

95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e

## 逐题行为与失败链

四个任务都从 read_policy → inspect_schema 开始，并查询实际存在的 orders/refunds 表。

- development-000：两次 SQL 返回结果，但业务验证为空；随后引用无效月度证据，并以数组提交 values。轨迹重复同一套流程两次。
- development-001：首次查询遇到预设临时故障，下一轮重试查询；查询返回后连续 8 次 calculate_change 引用无效证据。
- development-002：查询返回未验证指标，随后在 JSON 中直接写入 20406 - 2496 表达式，产生 4 次格式错误；两轮流程共 12 个动作。
- development-003：临时故障后重试查询，之后连续 8 次 calculate_change 引用无效证据。

故障任务均发生了实际查询重试；后续月度指标验证失败，最终恢复指标为 0/2。

错误共 26 次：无效月度证据 20、临时查询故障 2、非法 JSON 4。另有 6 次 SQL 成功执行并返回行，全部 verified_metrics=[]；development-000 的 4 次 check_result/submit 检查均未通过。

## SQL 的具体问题

模型在四个任务中采用相同结构：对 orders 与 refunds 做 JOIN，把 SUM(gross_cents) 标成 current_net_cents，把 SUM(refund_cents) 标成 prior_net_cents，按当前订单月份分组。

正确口径需要分别计算本月和上月：paid 订单收入减去该月实际发生的退款，退款通过原 paid 订单匹配 region/segment。当前生成的 SQL 缺少 paid 过滤、按退款发生月份统计和两个独立月份的净额计算，JOIN 聚合还会改变订单金额的计数范围。

环境因此返回空 verified_metrics。模型继续猜测 q1/q2 等引用，calculate_change 随后失败。提交端还把 values 写成数组；工具需要以三个指标名为键的整数对象，change_cents 证据应来自实际 calculate_change 返回的 cN。

## 本地复核

- ZIP CRC 校验通过；回传包 SHA-256：c274d22b64e493353c323bc956537aa3fcae94fb790a5b081cd0c57921a695d5。
- 20 个冻结源码/配置文件与运行 manifest、本地发布源码一致。
- 8 组检查点的 adapter 文件和 Adam 状态文件摘要全部匹配。
- 36 条训练示范重放成功；252 个动作样本与冻结源码重新生成的结果逐项一致，数据摘要匹配。
- 用已保存的固定 Qwen tokenizer 重新编码，最大训练上下文 1,430 token，监督 token 共 18,480；专家 SQL 动作为 136 token。
- 4 个开发任务、48 个动作在冻结环境中逐步重放，动作、观察、奖励、验证进度与终止状态全部一致。
- 48 个原生 prompt 与 chat template 一致；2,210 个生成 token 的解码及概率长度核对通过。48 次生成均以 EOS 结束，模型输出长度均在预算内。
- final adapter 含 112 个有限张量、1,089,536 个参数；Adam 状态读取及步数核对通过。

## 接续方案

保持本轮结果和检查点，下一轮先做小规模执行诊断：对原始模型和本轮 adapter 在相同训练任务上执行 greedy 轨迹，检查模型对已见任务的掌握程度；按 SQL、证据和提交动作分别统计 teacher-forced loss，并选少量相同 prompt 对齐训练侧与 vLLM 的 adapter 输出。

再根据诊断扩充训练任务与 SQL 正确示范，增加“SQL 执行成功但 verified_metrics 为空”后的修复、缺少 prior 月度证据后的补查、以及 values 对象和 cN 引用的示范。保留现有开发集与评测设置，训练任务承接新增修复样本。首个可完整交付的策略形成后再冻结共同 SFT 起点。

当前数据的 36 条轨迹来自 12 个基础任务；训练轮数与数据覆盖的下一轮配置随执行诊断确定。
