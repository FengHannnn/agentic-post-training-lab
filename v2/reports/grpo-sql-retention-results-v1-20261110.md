# SQL retention study v1 结果

本轮正常完成15步SFT和W/B配对评测，独立本地审计通过。新B的复合缺别名sample、greedy和首动作认证均为0；标准完整sample为13/16，低于本轮W的14/16，标准恢复sample为58/84，低于W的60/84。B门槛收尾，GRPO采样组和实际RL更新均为0，正式基线保留W。

新B的112个有效BF16 LoRA张量均相对W发生变化。目标轨迹中query_sql次数由W的36次增加到B的48次，全部48条SQL仍缺少current_net_cents和prior_net_cents名称，新增认证为0。当前重点继续放在warm-up的关键别名学习与标准能力维护。

契约见 reports/grpo-sql-retention-contract-v1-20261110.md，机器可读核验见 reports/grpo-sql-retention-results-validation-v1-20261110.json。

## 运行、归档和审计

- run_id：grpo-sql-retention-study-v1-20261110，attempt-001。
- error=null，stop_reason=b_gate_rejected，selected_candidate=null。
- 原始ZIP为86276964字节、991个唯一安全条目、172650138字节解压内容；CRC通过，文件大小和SHA256与回执一致。
- ZIP SHA256：0246e5f7b2fad61fecddc684d64e459fc71da43aa48dca1c9b1ea61d7eb98a62。
- 274项源码绑定和305项发布文件绑定通过；本地独立重放W/B各148条episode，共296条、1456个续接动作；另核验28次重复greedy质量动作。
- 15个SFT检查点、Adam步骤记录、实际权重变化、最终B导出及SFT528 / DPO11 / RL0血缘核验通过。Adam/RNG的state.pt继续留存云端，本地核验返回的权重、事务、计数和轨迹。
- GPU为与上一轮同一块NVIDIA L40S，Python3.11.17、Torch2.8.0+cu128、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1、SQLite3.54.1。

原始ZIP及回执位于 artifacts/returned/grpo-sql-retention-study-v1-20261110/；运行证据位于 artifacts/grpo-sql-retention-study-v1-20261110/，云端audit.json与本地local_audit.json分别保留。复核日志位于 .local/sql-retention-results-20261110/local-audit.log。

## 本轮实际配方

每个train任务5条监督：1条原复合修复、旧证据修复、失败检查修复、SQL错误恢复和检查后提交各1条。共12个任务、目标12条、维护48条，合计60条。

SFT从W重新开始，学习率2e-6、batch4、1epoch、15次更新、60例次，固定顺序seed2026111053。没有重复96条路线诊断，W重新评测。任务、sql_semantics_v2、原生8192/256预算、配对seed和验收门槛沿用冻结协议。

| 身份 | W | 新B |
| --- | --- | --- |
| 血缘 | SFT513 / DPO11 / RL0 | SFT528 / DPO11 / RL0 |
| adapter SHA256 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f | 38a9a6d68ed1a0ea0620aaa726ca6443fe701691a88c533634761663c6a72b31 |
| 有效BF16投影 | 112张量作为对照 | 112/112张量变化，worker加载证明与本地独立投影一致 |

branches/c的weighted_sft评测对应尚未启动RL的B，policy updates为0。上一轮SFT537的B单独留存，本轮独立从W开始。

## W与B配对结果

| 指标 | 本轮W | 本轮B | 成功数变化 |
| --- | ---: | ---: | ---: |
| 标准完整sample | 14/16 | 13/16 | -1 |
| 标准恢复sample | 60/84 | 58/84 | -2 |
| 标准完整greedy | 4/4 | 4/4 | 0 |
| 标准恢复greedy | 22/28 | 23/28 | +1 |
| 原复合sample终局 | 0/12 | 0/12 | 0 |
| 原复合greedy终局 | 0/4 | 0/4 | 0 |
| 原复合sample首动作认证 | 0/12 | 0/12 | 0 |
| 重复greedy一致性 | 7/7上下文 | 7/7上下文 | 均通过 |
| 标准及定向证据质量 | 通过 | 通过 | 均通过 |

B门槛中，两项标准sample非退步要求和目标首动作认证要求未通过；两项标准greedy和全部质量检查通过。按预设规则在B后正常归档。

## 目标行为：查询增加，别名仍缺失

W/B各16条无提醒原复合episode均耗尽12次总动作预算，续接动作各104次，新增有效认证均为0。

| 行为 | W | B |
| --- | ---: | ---: |
| 首动作query_sql | 8/16 | 10/16 |
| 首动作calculate_change | 3/16 | 0/16 |
| 首动作check_result | 3/16 | 2/16 |
| 首动作inspect_schema | 2/16 | 4/16 |
| 全轨迹query_sql次数 | 36 | 48 |
| SQL中包含必需月度名称 | 0/36 | 0/48 |
| 新增认证动作 | 0 | 0 |

常见输出延续前缀中的长SQL表达式，仍省略AS current_net_cents或AS prior_net_cents。随后calculate_change使用未认证证据，或重复check_result，最终预算耗尽。

对12条目标监督响应做原生token定位：每条完整响应136或142tokens，关键AS current_net_cents片段为5tokens，占3.52%～3.68%。当前action_loss对整个响应做均匀token平均；记录的目标训练损失均值为0.04574，维护为0.00452。这些数值是样本被训练时的损失记录。

下一步将单独记录关键片段的损失、分支概率和实际生成认证率，让目标学习是否发生具有直接检查指标。

## 标准恢复变化

| 场景 | sample W→B /12 | 新增成功 / 新增失败 | greedy W→B /4 |
| --- | ---: | ---: | ---: |
| format_repair | 11→10 | 0 / 1 | 4→4 |
| unknown_tool_repair | 12→12 | 0 / 0 | 4→4 |
| sql_unverified_requery | 11→11 | 0 / 0 | 4→4 |
| repeat_evidence_error | 0→0 | 0 / 0 | 0→0 |
| stale_evidence_repair | 12→12 | 0 / 0 | 4→4 |
| repeat_sql_error | 7→6 | 0 / 1 | 2→3 |
| failed_check_repair | 7→7 | 0 / 0 | 4→4 |

标准完整sample另有1条新增失败，合计sample新增失败3条、新增成功0条；greedy在repeat_sql_error增加1条成功。

三条sample回退分别为repair-development-000的format_repair rollout2、repair-development-003的完整rollout1及repeat_sql_error rollout2。行为包括使用未认证月度证据、临时查询错误后继续生成无效SQL、反复检查错误值或证据，均耗尽总预算。具体seed和动作统计见机器可读核验记录。

## 与上一轮的关系

| 配方及同轮对照 | 旧B24步轮 | 新B15步轮 |
| --- | ---: | ---: |
| 学习率 / epochs | 1e-5 / 4 | 2e-6 / 1 |
| 目标监督例次 | 48 | 12 |
| 维护监督例次 / 唯一任务 | 48 / 3 | 48 / 12 |
| W原复合sample→B | 0→8 /12 | 0→0 /12 |
| W标准完整sample→B | 14→13 /16 | 14→13 /16 |
| W标准恢复sample→B | 58→52 /84 | 60→58 /84 |

W独立进程复测的标准恢复为58→60、greedy恢复21→22，成功状态变化涉及5条轨迹；两轮内的重复greedy检查均通过。选优分别使用各自同轮fresh W，跨轮记录用于配方诊断。

本轮同时降低学习率和目标重复次数、扩大维护任务与状态覆盖。工作假设是目标别名学习强度偏低，下一步通过固定维护与学习率、单独改变目标监督强度来检验。关键token损失占比和实际生成结果分别记录。

## 时间和预算

本轮完整耗时3922.078秒，即1小时05分22秒。主要开销为W评测32分00秒、B评测31分51秒；SFT15步37.5秒，根审计15.1秒，review主体打包6.95秒。

两轮累计9500.092秒，即2小时38分20秒；距离既有21600秒上限还剩12099.908秒，约3小时21分40秒。回执中的本轮及合计budget_passed均为true。

## 下一步建议

继续修复warm-up，保持W为正式基线，B类候选通过门槛后再进入GRPO。当前两轮RL更新均为0，信用公式尚未参与这两轮权重更新。

先使用现成W、旧B24和新B15，在固定train起点补做小规模首动作诊断：记录实际别名生成、认证，以及别名片段与直接结束SQL的概率差。该检查区分目标动作在训练起点的掌握程度与development迁移。

随后优先固定本轮48条维护数据和较低学习率，只调整目标样本重复次数，先恢复目标修复的有效采样，再进入完整B验收。关键token加权作为单独候选因素登记。每个新配方使用独立身份与事前固定门槛，先做低成本首动作检查，再安排完整评测；后续GPU开销继续计入剩余总预算。
