# Fine GRPO 三方独立复评结果

记录日期 2026-11-13。对同一批 12 个独立任务族，**旧 local Fine GRPO（RL43）通过预先冻结的验收**，完整采样和恢复采样均超过 W。新收尾信用（RL47）取得最高完整采样成功数，恢复采样低于 W，也低于旧 local。复评选出旧 local 作为后续 Fine 候选，W 作为正式基线。[验收契约](grpo-fine-closeout-followup-contract-v1-20261112.md) · [选择记录](../artifacts/grpo-fine-closeout-followup-v1-20261112/selection.json)

## 运行设置与实际耗时

复评载入 W、旧 local 和新收尾信用的已有权重。三个策略在同一台 L40S 上运行，使用同一 `sql_semantics_v2` 验证器，按任务、场景、解码模式、rollout 编号、seed 和起点配对。每个策略评测 396 条标准轨迹、48 条复合缺别名专项轨迹和 14 次质量动作。任务族与训练、开发、历史 C4 及首轮独立复验隔离。本轮新增 SFT 和 RL 更新均为 0。[冻结配置](../artifacts/grpo-fine-closeout-followup-v1-20261112/config.json) · [运行状态](../artifacts/grpo-fine-closeout-followup-v1-20261112/final_status.json)

| 阶段 | 实际耗时 |
| --- | ---: |
| W 评测 | 1 小时 34 分 30 秒 |
| 旧 local 评测 | 1 小时 33 分 11 秒 |
| 新收尾信用评测 | 1 小时 33 分 17 秒 |
| 三支权重准备合计 | 25 秒 |
| 云端审计 | 45 秒 |
| 打包 | 7 秒 |
| **全程（含打包）** | **4 小时 42 分 59 秒** |
| 6 小时预算余量 | 1 小时 17 分 01 秒 |

全程采用打包收据的墙钟耗时，各阶段按秒四舍五入。[阶段耗时](../artifacts/grpo-fine-closeout-followup-v1-20261112/runner_state.json) · [打包收据](../artifacts/grpo-fine-closeout-followup-v1-20261112/returned_packaging_receipt.json)

## 标准评测与验收

| 指标 | W | 旧 local RL43 | 新收尾信用 RL47 |
| --- | ---: | ---: | ---: |
| 完整采样（主指标） | 24/48 | 26/48 | 27/48 |
| 恢复采样 | 177/252 | 183/252 | 176/252 |
| 完整 greedy | 11/12 | 11/12 | 11/12 |
| 恢复 greedy | 66/84 | 67/84 | 67/84 |
| 质量检查 | 通过 | 通过 | 通过 |

- **对 W 的验收**。旧 local 在完整采样提升 2 条（+4.17 个百分点），恢复采样提升 6 条（+2.38 个百分点），完整 greedy 持平，恢复 greedy 提升 1 条。新信用的完整采样提升 3 条，恢复采样少 1 条，四项均达到 W 的门槛未通过。
- **新信用对旧 local 的验收**。门槛为恢复采样净增至少 6/252。实际新增成功 3 条、失去成功 10 条，净减 7 条（−2.78 个百分点）。完整采样新增成功 2 条、失去成功 1 条，净增 1 条。两项 greedy 逐条成功状态一致。
- **复合缺别名专项**。三支各为采样 0/36、greedy 0/12，首动作认证和后续取得进展的轨迹数也均为 0。

[配对与验收明细](../artifacts/grpo-fine-closeout-followup-v1-20261112/selection.json)

## 恢复差异出现在什么地方

认证指环境认可某项指标的值和证据。检查通过指模型调用 `check_result`，环境对三项指标、证据引用和政策要求的检查均通过。成功提交指 `submit` 被接受。认证统计包含恢复起点已有的证据。

| 恢复采样阶段 | 旧 local | 新收尾信用 | 变化 |
| --- | ---: | ---: | ---: |
| 两个月净额均认证 | 229/252 | 223/252 | −6 |
| 三项指标均认证（含差额） | 215/252 | 206/252 | −9 |
| 检查通过 | 185/252 | 178/252 | −7 |
| 成功提交 | 183/252 | 176/252 | −7 |

两支都拿到三项认证的 205 条配对轨迹中，新信用新增 2 次成功提交，也失去 2 次，净变化为 0。其余轨迹新增 1 次、失去 8 次，形成了恢复成功数的净差。以任务 `fine-closeout-followup-002` 的重复证据错误场景、rollout 2 为例，两支首步都遇到相同的月度证据引用错误。旧 local 重查净额、计算差额并完成检查和提交，新信用六次重复计算并在回合耗尽时仅持有起点已有的上月认证。[旧 local 轨迹](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/local/evaluations/grpo_g0048.json) · [新收尾信用轨迹](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/closeout/evaluations/grpo_g0048.json)

| 恢复采样场景 | W | 旧 local | 新收尾信用 |
| --- | ---: | ---: | ---: |
| 格式错误恢复 | 33/36 | 35/36 | 33/36 |
| 未知工具恢复 | 36/36 | 36/36 | 36/36 |
| 未认证 SQL 重查 | 31/36 | 31/36 | 31/36 |
| 重复证据错误 | 4/36 | 7/36 | 3/36 |
| 旧证据恢复 | 35/36 | 35/36 | 35/36 |
| 重复 SQL 错误 | 21/36 | 20/36 | 19/36 |
| 检查失败恢复 | 17/36 | 19/36 | 19/36 |

重复证据错误贡献了 4 条净回退，格式错误恢复贡献 2 条，重复 SQL 错误贡献 1 条。预先关注的未认证 SQL 重查、旧证据恢复、检查失败恢复三项里，旧 local 和新信用的配对成功状态完全一致。[分场景汇总](../artifacts/grpo-fine-closeout-followup-v1-20261112/selection.json) · [W 轨迹](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/w/evaluations/weighted_sft.json)

上一轮新信用的开发面板完整采样为 11/16，W 为 9/16，首轮独立复验两支均为 8/16。这次扩大面板后，新信用在完整采样上再次领先 W，而旧 local 在四项标准指标的联合验收中胜出。[上一轮结果](grpo-fine-closeout-credit-results-v1-20261112.md)

## 本地复核

核对 ZIP 的字节数、SHA-256 和 1,397 个条目的 CRC 后，使用返回的冻结输入、评测记录和来源证据在本地重放。云端和本地的验收结果、逐任务配对和阶段计数一致。[结构化验证记录](grpo-fine-closeout-followup-results-validation-v1-20261113.json) · [本地审计](../artifacts/grpo-fine-closeout-followup-v1-20261112/local_audit.json) · [云端审计](../artifacts/grpo-fine-closeout-followup-v1-20261112/audit.json)
