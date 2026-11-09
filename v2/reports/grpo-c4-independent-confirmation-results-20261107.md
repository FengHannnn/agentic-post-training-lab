# C4 independent confirmation results

日期：2026-11-07。运行：`grpo-c4-confirmation-v2-20261107`。

**本轮完整完成，云端与本地审计通过；两支GRPO均未达到事前冻结的实用验收门槛。** outcome RL16在新任务上有小幅采样收益：完整成功比W多2/48，恢复多4/252；fine RL17完整多1/48，恢复持平。两支greedy恢复均比W少3/84。当前“至少一个GRPO分支达到最终超越W门槛”的项目目标继续待完成。

## 本轮效果与验收

| 模型 | sample完整（本轮主指标） | sample恢复 | greedy完整 | greedy恢复 |
|---|---:|---:|---:|---:|
| W 加权SFT | 28/48 (58.33%) | 171/252 (67.86%) | 12/12 | 61/84 (72.62%) |
| outcome RL16（C2） | 30/48 (62.50%) | 175/252 (69.44%) | 12/12 | 58/84 (69.05%) |
| fine RL17（C3） | 29/48 (60.42%) | 171/252 (67.86%) | 12/12 | 58/84 (69.05%) |

W的sample完整为28/48，尚未饱和，按已冻结顺序，本轮主指标为完整成功率。outcome提升4.17个百分点，fine提升2.08个百分点；greedy完整三者均100%。

实用验收同时要求恢复至少净增6/252（+2.38个百分点）、sample完整与两项greedy成功数不下降，并且主指标增加。逐项结果：

| 条件 | outcome RL16 | fine RL17 |
|---|---:|---:|
| sample恢复至少+6 | +4，未达标 | 0，未达标 |
| sample完整不下降且主指标增加 | +2，通过 | +1，通过 |
| greedy完整不下降 | 0，通过 | 0，通过 |
| greedy恢复不下降 | -3，未达标 | -3，未达标 |
| 最终实用验收 | **未通过** | **未通过** |

验收门槛沿用运行前冻结配置，运行后保持原样。outcome的sample收益优于本轮fine候选；W继续作为参照，outcome可作为后续受控优化的优先候选。

## 配对结果与任务聚类区间

所有比较按同任务、场景、mode和rollout配对。下表区间来自12个任务族聚类bootstrap，10,000次，作为描述统计。

| 候选对W | 指标 | 改善 / 退步 | 净成功数 | 提升及95%聚类区间（百分点） |
|---|---|---:|---:|---:|
| outcome | sample完整 | 2 / 0 | +2 | +4.17 [0.00, +10.42] |
| outcome | sample恢复 | 9 / 5 | +4 | +1.59 [-1.59, +4.76] |
| outcome | greedy恢复 | 0 / 3 | -3 | -3.57 [-9.52, 0.00] |
| fine | sample完整 | 2 / 1 | +1 | +2.08 [-4.17, +8.33] |
| fine | sample恢复 | 7 / 7 | 0 | 0.00 [-3.97, +3.97] |
| fine | greedy恢复 | 0 / 3 | -3 | -3.57 [-9.52, 0.00] |

fine相对outcome：sample完整改善1条、退步2条，净-1；sample恢复改善5条、退步9条，净-4；两项greedy成功标志完全相同。本轮比较的是事前选定的C2/C3最终候选；细粒度方法的同预算方法对照见C3报告。

## 新任务上的具体变化

outcome新增的两条sample完整成功为 `c4-confirmation-000 / rollout0` 和 `c4-confirmation-001 / rollout3`，没有完整任务退步。fine改善 `001 / rollout3`、`011 / rollout3`，并在 `003 / rollout2` 退步。rollout均使用原始0起始编号。

sample恢复按场景汇总，每格分母均为36：

| 场景 | W | outcome | fine |
|---|---:|---:|---:|
| format_repair | 32 | 35 | 33 |
| unknown_tool_repair | 36 | 36 | 36 |
| sql_unverified_requery | 25 | 25 | 26 |
| repeat_evidence_error | 12 | 12 | 8 |
| stale_evidence_repair | 34 | 34 | 34 |
| repeat_sql_error | 15 | 15 | 17 |
| failed_check_repair | 17 | 18 | 17 |

outcome的净恢复收益主要来自format_repair与failed_check_repair。fine在重复SQL错误中增加2次成功，重复证据错误减少4次，最终总恢复收益为0。

## 三条共同的greedy恢复退步

两支GRPO相对W退步的三个键完全相同：

| 任务 | 场景 | W行为 | GRPO行为 |
|---|---|---|---|
| c4-confirmation-001 | sql_unverified_requery | 重查→重算差额→检查→提交，4步成功 | 重查后跳过calculate_change，差额和证据错误，7步耗尽 |
| c4-confirmation-001 | failed_check_repair | 修正值后检查→提交，2步成功 | 保留差额-17474，正确值为-17475，重复检查5步耗尽 |
| c4-confirmation-003 | sql_unverified_requery | 重查→重算差额→检查→提交，4步成功 | 数值正确但差额证据误指q4，跳过重算，7步耗尽 |

例如001的重查得到current=7072、prior=24547，正确差额为-17475。W调用calculate_change产生差额证据后提交；GRPO沿用4617并把change证据指向只证明current的q4，收到失败检查后继续重复相同动作。003虽填写正确差额-8088，证据仍指向q4，最终同样失败。这里需要同时维护数值与证据的绑定，并根据失败检查采取修复动作。

## 完整任务的主要失败位置

三者sample完整失败分别20、18、19条，全部在12步预算耗尽后结束。按失败轨迹最终曾取得的有效指标数分类：

| 已验证指标数 | W失败轨迹 | outcome失败轨迹 | fine失败轨迹 |
|---|---:|---:|---:|
| 0 | 15 | 16 | 16 |
| 1 | 2 | 1 | 1 |
| 2 | 1 | 1 | 1 |
| 3 | 2 | 0 | 1 |

合计57条失败中，47条（82.46%）未取得任何被验证器认可的指标证据。轨迹中出现：SQL缺少要求的metric alias、使用不存在的表/列、调用白名单外的strftime/LIKE、动作顶层格式错误；随后在缺少有效月度证据时反复调用calculate_change。首次查询临时故障按原环境规则保留，错误分类中单独记录。

一个典型案例是 `c4-confirmation-002 / rollout1`：SQL已经算出当前22738、上期14460，却缺少current_net_cents/prior_net_cents alias，证据未获验证。后续模型继续绕开有效证据，最终耗尽预算。另一类是已得到三个正确指标后，在check_result中错置数值或丢失证据，再重复同一失败检查。

greedy完整三者均12/12，96个脚本专家完整/恢复起点也均在同样12步预算内完成。新任务上，随机生成时的SQL取证稳定性与错误后的流程修复是下一轮优先改进方向。

## 开销、身份与审计

本轮新增训练更新为0。三者独立进程、单LoRA编号1，共完成1,188条episode、6,056个生成动作，另有42次greedy质量动作（2,982个响应token）。常规episode响应token合计495,531，含质量动作总计498,513。

| 模型 | episode动作数 | episode响应token | 评测耗时 |
|---|---:|---:|---:|
| W | 2,026 | 165,992 | 1小时31分00秒 |
| outcome RL16 | 2,012 | 165,237 | 1小时29分15秒 |
| fine RL17 | 2,018 | 164,302 | 1小时27分19秒 |

准备约9秒，云端审计约38秒，总耗时 **4小时28分22秒**，低于7小时上限。三模型各一个推理session，均完成396条episode，无中断续跑。

本次GPU为NVIDIA L40S，UUID `f96adc6f-826c-805f-a41d-d34af1c0775a`。10份资源快照与三模型session保持同一身份。Python3.11.17、Torch2.8.0+cu128、CUDA12.8、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1。本轮W在同一环境重新评测，所有比较采用本轮配对结果。

模型权重沿用冻结身份：

| 标签 | adapter_model.safetensors SHA256 |
|---|---|
| W / SFT513-DPO11-RL0 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| C2 outcome / RL16 | 6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6 |
| C3 fine / RL17 | 1990f06d2eb8e3a1777273051bf2bce078c8bff67206533b88dd65780e5959c6 |

返回ZIP共1,457项、35,260,681字节，CRC通过。SHA256为 `4b1bb7c5b55e1af8cedaad0ead8656d81089a5fe90dc7c695307c211e7ac2f5d`；发布ZIP SHA256为 `293a694dc219a1d431ca7bb4e5cc3fc3ebf122c666ba63614039a0863bedc0c2`。

226个发布绑定文件、返回包211个冻结源码文件及26个冻结输入绑定均核验通过。使用包内冻结源码，本地重新验证全部1,188条轨迹、6,056个动作、42次greedy质量动作及重复token一致性、加载权重证明和96个专家起点；全部汇总、配对、验收判断与云端逐项一致。严格重放与模型效果验收分别记录：审计通过，效果验收未通过。

## 下一轮建议

优先以结果奖励GRPO做小规模受控优化：扩大训练端的数据实例与失败续跑覆盖，重点覆盖SQL alias/表列正确性、未验证证据重查、差额重算、失败检查后的数值/证据修复；在独立开发集筛选学习率和更新量。fine保留为后续对照。新增SFT/CE若采用，按既定规则另立混合方法记录。

本批C4结果保持冻结。若后续利用这些失败模式调参，本批转为诊断数据，下一次最终确认使用新冻结任务。当前不修改本轮模型权重或验收门槛，也不重跑本批数据来选择参数。

## 文件和代码阅读顺序

报告：`reports/grpo-c4-independent-confirmation-results-20261107.md`。项目复核材料：`artifacts/reviews/grpo-c4-confirmation-v2-20261107/`。完整返回包保留原始ZIP，结果证据另封装为 `grpo-c4-independent-confirmation-audit-20261107.zip`。

以下路径相对v2根目录：

1. `configs/grpo-c4-confirmation-v2.json`：冻结候选、12任务、7小时和验收门槛。
2. `src/apt_v2/grpo_confirmation.py`：任务、配对统计和验收判断。
3. `scripts/grpo_c4_common.py`：原生重放、权重验证和完整审计。
4. `scripts/grpo_c4_worker.py`：单模型生成、质量检查与逐episode事务。
5. `scripts/run_v2_grpo_c4.py`：作业调度、预算和打包。
