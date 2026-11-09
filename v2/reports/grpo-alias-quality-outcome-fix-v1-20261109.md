# GRPO alias quality-outcome fix v1

## 运行状态与失败原因

第二次返回包中，两支训练仍为 48/48 组，实际更新 control 25 次、curriculum 37 次。control 最终评测已完成 132 条标准轨迹、32 条定向轨迹及 14 次质量生成，完整结果已保存。失败发生在最后的质量/日历证据门槛检查。

第一项：alias-development-003 的一条完整采样轨迹中，上月净收入 SQL 的退款子查询使用 SUM(r.refund_cents)，缺少 COALESCE(...,0)。原数据和仅扰动金额的数据仍有退款，所以环境认证成功；扩展日历校验将上月退款移走后，SUM 返回 NULL，正确净收入为 36080，触发 calendar_final_evidence=False。

第二项：alias-development-000 的 repeat_evidence_error 上，相同提示和 seed 的两次原生 greedy 输出不同，一次选择 calculate_change，另一次选择 query_sql，触发 repeated_greedy_tokens_identical=False。当前记录确认这一质量差异；底层数值原因留待专项诊断。

## 本次流程修复

保留 control 的原始评测、两项失败判定和原验收门槛。新流程复用已保存的完整评测，单独写入 quality-status.json，继续完成 curriculum 的既定评测、两支刷新及总审计。

结构或原生证据错误（配置、任务、seed、token、轨迹重放、权重绑定等）继续中止作业。模型质量和日历反例作为实验结果记录，不再中止另一支的评测。定向评测中的日历反例同样明确记录。

任一原质量门槛未通过时，alias_audit.json 的 passed 保持 false，evaluation_quality_passed 为 false，selected_candidate 为 null。作业完成状态为 completed_with_quality_rejections，表示采集与审计完成、实验质量验收未通过。门槛、任务、采样、训练配置、学习率、信用、模型权重均保持原值。

## 预算与续跑

当前累计 16783.15 秒（4 小时 39 分 43.15 秒），原六小时预算剩余 4816.85 秒（1 小时 20 分 16.85 秒）。control 完整评测直接复核复用，余下预计约 45–60 分钟；继续沿用原累计预算。

安装器校验第一版修复后的源码与 release 绑定、第二次失败状态及已提交 control 评测哈希，备份旧源码并同步根目录与两支冻结源码。原有实验文件、评测 JSON、权重、Adam/RNG 和计时在安装前后逐文件核验。安装记录：artifacts/grpo-alias-curriculum-v1-20261109/hotfixes/alias-quality-outcome-fix-v1/。

## 本地验证

85 项相关回归通过。使用与服务器相同的 SQLite 3.54.1，独立重放 W 与 control 共 328 条真实评测轨迹及各自质量记录；W 保持通过，control 的两项失败与服务器原审计逐项一致。已保存评测复用验证的 GPU 再生成次数为 0，原始评测 JSON 哈希不变。真实返回包副本安装前后 1,478 个原有产物哈希一致，训练组数、实际更新和累计计时保持原值。

## 使用方式

将 grpo-alias-quality-outcome-fix-v1-20261109.zip 上传到 /root/rivermind-data/AgenticPostTrainingLab，在原实例执行：

    cd /root/rivermind-data/AgenticPostTrainingLab &&
    unzip -o grpo-alias-quality-outcome-fix-v1-20261109.zip -d . &&
    cd v2 &&
    /root/rivermind-data/envs/apt-vllm/bin/python -u hotfixes/alias-quality-outcome-fix-v1/apply.py &&
    /root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_alias_curriculum.py --config configs/grpo-alias-curriculum-v1.json --resume

终端保留进度条，control 已保存评测显示 quality REJECTED 后继续其他评测。完成后返回 Download 指向的 review-attempt-003.zip。

## 代码阅读（仅本次新增或改动）

| 文件相对路径 | 大致作用 |
| --- | --- |
| scripts/evaluate_v2_grpo.py | 保留原检查，复用已提交结果并单独记录未通过的质量与日历门槛。 |
| scripts/audit_v2_grpo_alias.py | 汇总完整实验，任一质量门槛失败时保持验收失败并阻止候选选择。 |
| scripts/run_v2_grpo_alias_curriculum.py | 区分作业完成与质量验收结果，输出 completed_with_quality_rejections。 |
| tests/test_grpo_alias_quality_v2.py | 覆盖原始数据复用、质量拒绝、结构错误中止、空退款反例和候选阻断。 |
| hotfixes/alias-quality-outcome-fix-v1/apply.py | 备份和安装修复，迁移源码绑定，验证实验产物与累计计时保持原值。 |

报告项目路径：v2/reports/grpo-alias-quality-outcome-fix-v1-20261109.md。
