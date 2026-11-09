# 恢复偏好采集复核报告

日期：2026-11-02。运行：dpo-recovery-collect-v1-20261102。

结论：数据扩充完成。最终 169 对偏好通过本地重建与审计，覆盖 12 个训练任务，已归档为可训练数据集。当前冻结 DPO 在 SQL 未验证后重新查询、重复证据错误和检查失败后的恢复上存在集中短板，新增偏好覆盖了这些状态与对应正确动作。

## 数据组成

| 来源 | 对数 | 含义 |
|---|---:|---|
| frozen_sft_sample | 44 | 原首轮 SQL 偏好完整保留 |
| scripted_failure | 84 | 7 类恢复场景，每类覆盖 12 个 train 任务 |
| frozen_dpo_sample | 41 | 本轮冻结 DPO 实际采样失败，经去重与上下文数量限制后入选 |
| 合计 | 169 | 同一上下文下经过验证的 chosen/rejected |

chosen 动作分布：query_sql=104、calculate_change=41、check_result=24。包含 chosen token 17,581 个、rejected token 10,359 个；最长 prompt+response 为 1376 token，长度上限为 2048。

最终 pairs.jsonl SHA256：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。

## 252 次模型采样的行为

采样来自 84 个规则构造的恢复上下文，每上下文 3 次单动作采样。temperature=0.8，冻结 11 步 DPO 适配器，参数更新数为 0。各场景有 36 次采样；下表“有效进展”指新增规则/schema/已验证指标进展或结果检查通过。

| 恢复上下文 | 有效进展/36 | 失败或无进展/36 | 新模型偏好入选 |
|---|---:|---:|---:|
| 格式错误后恢复 | 27 | 9 | 4 |
| 未知工具后恢复 | 36 | 0 | 0 |
| SQL 未验证后重新查询 | 0 | 36 | 12 |
| 连续证据错误后恢复 | 0 | 36 | 0（与规则偏好完全重复） |
| 错误证据编号修复 | 34 | 2 | 1 |
| 连续 SQL 错误后恢复 | 0 | 36 | 12 |
| 检查失败后修复 | 0 | 36 | 12 |
| 合计 | 97 | 155 | 41 |

155 次失败或无进展中，1 次响应因长度限制未完整结束，保留采样记录；其余 154 次形成模型负例候选。候选包含工具错误 153 次、无进展 1 次。252 次采样共出现 7 次无效结构输出，未知工具输出为 0。场景名称描述其前缀状态，实际负例类型独立由环境执行结果判断。

本轮冻结模型在 48 次采样中原样重复上一轮动作；其中“连续证据错误”场景 36/36 次均重复了刚刚失败的 calculate_change。

## 关键偏好内容

1. SQL 未验证时：rejected 继续引用未验证证据执行 calculate_change；chosen 重新查询缺失的月度净额。
2. 连续证据错误时：rejected 重复之前失败的证据引用；chosen 查询缺失指标，获得新验证证据。
3. 已有正确月度证据但编号错误时：chosen 使用已有真实编号计算差额。
4. 连续 SQL 错误时：rejected 继续查询不存在的 metrics 表；chosen 使用真实 orders/refunds 表及业务口径查询。
5. 检查失败时：chosen 修正结果值，携带正确证据重新 check_result；模型负例中存在转回错误 calculate_change 的行为。

“连续证据错误”场景的 36 条模型负例与对应规则负例在 messages/chosen/rejected 上完全一致。去重优先保留规则来源，因此该类有 12 对偏好，实际模型重复行为通过 samples.jsonl 和行为分析记录保留。最终 41 对模型来源数据是新增的独特内容。

## 审计与绑定

prepare、collect、build、audit 四阶段全部成功，耗时 304.4 秒（约 5.1 分钟）。本地核对 51 个源码文件、11 个源输入绑定、5 个准备文件和 6 个 tokenizer 文件，重放 84 个恢复上下文，审计 252 条采样记录，完整重建全部 169 对偏好；本地审计与云端一致。

全部数据仅来自 train 的 12 个 task/family。规则前缀通过真实环境执行产生反馈；chosen 在同一动作前状态验证。规则负例明确标记 chat_template 编码，模型负例保留原生 token 与 raw log probability。原 44 对与此前已复核版本逐项一致。

冻结采样适配器 SHA256：fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d。

回传 ZIP SHA256：531dd4c4ec70ef36991183d94edeeae1ee7bde951caad6ba765b803e269d00a3

## 项目内归档

可训练数据：artifacts/datasets/dpo-recovery-v1-20261102/pairs.jsonl。dataset_manifest.json 固定数据身份、来源、数量和编码入口。

复核证据：artifacts/reviews/dpo-recovery-collect-v1-20261102 下的 local_review_summary.json、local_audit.json、recovery_behavior_analysis.json 和 representative_pairs.json。

## 下一步

进入恢复专项 DPO 训练时，使用 encode_recovery_pair() 统一编码这 169 对数据；它兼容原 44 对与新增两类 token 来源。先固定训练起点、参考策略及步数，再进行训练前后开发集任务评测和恢复上下文评测。重点跟踪未验证 SQL 后重新查询、重复错误调用、格式有效率、证据恢复与完整任务成功率。