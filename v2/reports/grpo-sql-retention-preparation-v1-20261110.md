# SQL retention study v1 实现与运行说明

本轮已实现60条train监督动作、较小SFT更新、重新评测W与B，以及B通过后启动无提醒GRPO。正式基线保留W，云端作业由用户上传后执行。

完整配方和验收规则见 reports/grpo-sql-retention-contract-v1-20261110.md。上轮结果见 reports/grpo-sql-repair-results-v1-20261110.md。

## 固定变更

- 目标12条，维护48条；12个train任务各有旧证据修复、失败检查修复、SQL错误恢复和检查后提交四种维护状态。
- 学习率1e-5→2e-6，4epochs→1epoch，24步→15步；batch4，60例次，血缘SFT528 / DPO11 / RL0。
- 复用已审计路线结论；W重新评测。B门槛、最终C门槛、148条评测计划、质量检查与GRPO36组配方沿用冻结协议。
- 本轮累计最多16000秒，与上轮已用5578.013秒合计处于21600秒总预算内。实际打包计入预算。
- 独立run_id、源码和发布清单；终端显示SFT、训练组、评测、审计和打包进度。

## 本地验证

60条监督动作已逐条通过真实前缀和动作重放；最长提示1358tokens，最长响应142tokens，均在8192/256限制内。掩码和EOS通过原生token验证。任务隔离为12个train、4个development，与32个历史保留实例分离。

24个GRPO测试模块共324项通过；新旧repair/retention定向50项通过，干净解压后再次通过50项测试。四个新CLI入口、306个安全唯一条目及305项文件哈希核验通过。验证结果写入 reports/grpo-sql-retention-validation-v1-20261110.json，日志保存在 .local/sql-retention-20261110/。最终包沿用干净解压测试的源码、配置和冻结输入，补充回归与发布核验记录。本地真实LoRA/Adam更新及恢复由微型Qwen模型测试覆盖，云端作业记录1.5B模型的实际效果。

## 云端启动

先上传 grpo-sql-retention-study-v1-20261110-run.zip 到云端项目父目录，并在该目录执行：

~~~bash
(
set -e
unzip -o grpo-sql-retention-study-v1-20261110-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_retention.py --config configs/grpo-sql-retention-study-v1.json
)
~~~

对应作业中断后，在v2目录给同一命令添加--resume。已完成作业保留归档，下一次配方使用新run_id。完成后返回终端Download所示review ZIP和Receipt所示JSON，state.pt继续保留云端。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo_retention_study.py | 数据状态、固定SFT配方和来源预算验证 |
| grpo_repair_study.py | SFT528分支身份校验 |
| grpo_common.py | 新B血缘及输入绑定 |
| grpo_repair_worker.py | 共用SFT更新、恢复和审计 |
| run_v2_grpo_retention.py | W/B/C顺序和总预算调度 |
| audit_v2_grpo_retention.py | 独立验收重算 |
| validate_v2_grpo_retention.py | 执行、token和监督掩码验证 |
| build_v2_grpo_retention_release.py | 生成并核验云端包 |
| grpo-sql-retention-study-v1.json | 主实验配置 |
| grpo-sql-retention-branch-v1.json | 共用分支配置 |
| test_grpo_retention_v2.py | 本轮数据与事务测试 |
