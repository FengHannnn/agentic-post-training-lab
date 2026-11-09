# DPO 随机轨迹与偏好数据复核

日期：2026-11-02（Asia/Shanghai）。硬件：NVIDIA L20。运行：dpo-collect-v1-20261102。

随机采样完成 48 条轨迹，成功 23/48（47.9%），故障任务成功恢复 12/24（50.0%）。按同上下文环境验证、去重和每任务上限筛选，得到 44 对有效偏好，覆盖全部 12 个训练任务，已达到首轮 DPO 数据复核要求。

## 随机表现

| 指标 | 结果 |
|---|---:|
| 训练任务 / 每题采样 | 12 / 4 |
| 完整轨迹成功 | 23/48 |
| 故障轨迹恢复 | 12/24 |
| 严格动作格式通过 | 465/473（98.3%） |
| 有效 SQL | 49 |
| 已验证指标事件 | 72 |
| 工具错误 | 194，其中 24 次预设临时故障 |
| 生成 token | 37151 |
| 成功与失败兼有的任务组 | 12/12 |

使用冻结的 126-step SFT，temperature=0.8，top_p=1，每轮保留原生 token、原始 logprob 和采样种子。greedy 与随机采样分开报告：此前 greedy train 12/12、development 4/4、test 4/4；本轮显示随机生成下仍存在可改善的执行错误。

主要错误包括证据引用失败、SQL 列名/别名错误、不可用函数，以及 SQL 可执行但业务验证未通过。未接纳的失败检查和晚期动作保留在原始轨迹中。

## 偏好数据

- 分支验证产生 50 个候选，去重与每任务最多 4 对后，保留 44 对。
- 每任务 2～4 对，12 个任务全部覆盖；最大 prompt+response 长度为 1199 token，低于训练编码上限 2048。
- 理由分类：工具错误 24 对、SQL 无已验证指标 20 对。
- chosen 全部为脚本生成、经同状态分支执行验证的正确 SQL。
- rejected 来自冻结 SFT 实际采样：query_sql 40 对、非法 JSON 2 对、缺少有效证据的 calculate_change 2 对。
- 两个动作共享同一 messages 和原生 prompt；rejected 保留实际生成 token 和 EOS，chosen 按同一 chat template 编码。
- 标签粒度是当前动作能否取得可验证进展；完整轨迹的终局结果另外记录。
- 24 次预设临时查询故障均排除在偏好错误标签之外。

例如 train-000 的一个负例把退款列写为 t2.refund_cents，但 t2 指向 orders，执行后返回 no such column。chosen 查询在完全相同的前置状态得到 current_net_cents=19808，并通过原数据与扰动数据验证。

pairs.jsonl SHA-256：508464a0314982c7e6ade95c57afa3be081afc80f237e4fb20c8abe2079ffb78。

## 本地复核

- ZIP CRC 通过，37 个冻结源码/配置文件摘要一致，并匹配本地发布源码。
- 数据源 checkpoint 与已归档 SQL 加权 SFT、共同冻结基线的 adapter 身份一致；6 个 tokenizer/config 文件匹配。
- 48 条轨迹、473 个动作重新执行，原生 prompt、解码、观察、奖励、恢复和终止一致。
- 50 个反事实候选重新验证；独立重建后的 44 对偏好与回传数据逐项一致，汇总和 audit.json 完全一致。
- 采样全过程参数更新为 0，作业耗时 1201.07 秒；prepare、collect、build 均成功。

原始 ZIP SHA-256：9457f792cd1a64c4225790e49d658405c2da1b08f4ec6dcbe5af838bbf2a6843。

## 首轮训练安排

建议先用 44 对数据做有限规模 DPO：从相同冻结 SFT 开始，新建 optimizer，采用 1 遍数据、每 4 对累积一次，共 11 次更新；学习率 5e-6、beta=0.1、梯度裁剪 1。reference 概率固定为同一 SFT 的分数，动作 response 使用序列 logprob 求和。

评测关注随机成功、恢复、错误类型和 greedy 行为保持，采用新评测 seed 并重测同设置 SFT 基线。可选 chosen-only SFT 对照使用相同 chosen、相同起点和更新步数，用于比较不同训练目标。

## 文件位置

原始回传、提取数据和本地复核：artifacts/reviews/dpo-collect-v1-20261102/。

正式偏好数据：上述目录 extracted/dpo-collect-v1-20261102/pairs.jsonl；云端对应 artifacts/dpo-collect-v1-20261102/pairs.jsonl。

本地审计：local_review_summary.json、local_pair_summary.json、local_audit.json。
