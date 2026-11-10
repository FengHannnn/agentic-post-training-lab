# Warm-up probe v1 实现与运行说明

已实现W、旧B24、新B15的train首动作诊断。每模型12个目标起点各重复2次greedy，另有12个维护起点；同时记录关键别名分叉和teacher-forced原生token分数。本次不执行优化器更新。

配方见 reports/grpo-warmup-probe-contract-v1-20261110.md，上一轮结果见 reports/grpo-sql-retention-results-v1-20261110.md。模型和输入在 artifacts/inputs/grpo-warmup-probe-20261110/。

## 实现与本地验证

- 三份实际adapter及其SFT513/537/528血缘验证通过；W历史导出的adapter README与其哈希一并保留。
- 60条监督动作、24个诊断起点的专家动作及12个删别名反事实已通过真实执行。
- 最长专家提示1358tokens、响应142tokens，预算为8192/256；12个目标起点的别名片段均为5tokens，原生共同前缀和分叉位置一致。
- 本轮生成108次首动作，另有72次teacher-forced打分及72个附带生成token；原生分数按prefill诊断记录，实际动作逐条执行。
- 模型session绑定真实worker LoRA、GPU及输入；完成起点和模型均可验证复用。请求日志包含重试与中断记录。

18项新诊断测试与12项既有repair driver测试共30项通过，干净解压后再次通过30项；四个CLI入口、343个唯一安全条目和342项文件哈希核验通过。中断仿真发现的首次日志目录初始化缺失已修复。实际原生token、环境执行和adapter投影在本地核验，vLLM请求及恢复流程使用模拟引擎覆盖，云端作业记录三模型真实推理。最终包沿用干净解压测试的代码、配置和冻结输入。验证及发布记录写入 reports/grpo-warmup-probe-validation-v1-20261110.json，日志位于 .local/warmup-probe-20261110/。云端作业使用既有基座缓存和apt-vllm环境。

## 云端启动

上传 grpo-sql-warmup-probe-v1-20261110-run.zip 至云端项目父目录，在该目录执行：

~~~bash
(
set -e
unzip -o grpo-sql-warmup-probe-v1-20261110-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_warmup_probe.py --config configs/grpo-sql-warmup-probe-v1.json
)
~~~

中断后在同一v2目录给原命令增加--resume。已完成作业保留归档。返回终端Download所示review ZIP和Receipt所示JSON，用于本地独立重放及累计时间核验。

本轮最多30分钟，与前两轮已用时间合计受原6小时预算约束。结果回来后固定目标重复次数的单变量对照，维护数据、学习率和验收门槛继续保持一致。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo_warmup_probe.py | 数据计划、执行认证和别名分叉统计 |
| grpo_warmup_probe_worker.py | 推理、原生条件分数和独立审计 |
| run_v2_grpo_warmup_probe.py | 有限预算、恢复与打包 |
| validate_v2_grpo_warmup_probe.py | 输入及原生token验证 |
| build_v2_grpo_warmup_probe_release.py | 发布包核验 |
| grpo-sql-warmup-probe-v1.json | 固定诊断配置 |
| test_grpo_warmup_probe_v2.py | 执行与恢复测试 |
