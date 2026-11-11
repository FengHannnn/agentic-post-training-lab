# r16 中间学习率实验实现与运行

本轮从 W 独立训练 4e-6 一档。r16 的 60 步、192 个目标动作例次、48 个维护动作例次与批次顺序均固定。候选通过 train 筛选后进入完整 B 验收。[冻结契约](grpo-sql-lr-midpoint-contract-v1-20261111.md)

## 本地验证

验证了 60 条原生监督动作和 24 个 SQL 探针，最长提示 1358 tokens，最长响应 142 tokens。数据及批次逐项匹配 r16。新增收尾计数已在上一轮 264 条标准轨迹上复核，重现 21 条配对退化、其中 17 条已认证三项指标，以及 21 条未尝试提交。[机器验证记录](grpo-sql-lr-midpoint-validation-v1-20261111.json)

全部 28 个 GRPO 模块的 387 项回归通过，干净解压目录的 113 项定向复测通过。四个 CLI 入口、原生验证、模拟中断和续跑、累计计时及预算收尾通过。发布包的 391 个条目和 390 项文件哈希均已核验。

## 云端运行

将 `grpo-sql-lr-midpoint-v1-20261111-run.zip` 上传到云端仓库根目录，即 `v2/` 的上一级，然后执行。

~~~bash
(
set -e
unzip -o grpo-sql-lr-midpoint-v1-20261111-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_lr.py --config configs/grpo-sql-lr-midpoint-v1.json
)
~~~

中断后从 `v2/` 给 Python 命令加 `--resume`。训练、探针、完整评测、审计和打包在终端显示进度。参照上一轮 L40S 用时，若候选止于 train 筛选，预计 40–45 分钟。进入完整 B 验收预计约 1 小时 15 分至 1 小时 30 分。墙钟上限 3 小时 45 分。

结束后返回终端 `Download` 对应的 review ZIP 和 `Receipt` 对应的 JSON。用收据和 `final_status.json` 核对预算与选择。 `completion-diagnostics.json` 给出标准任务的取证、检查和提交计数。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo-sql-lr-midpoint-v1.json | 单档配方和预算 |
| grpo-sql-lr-midpoint-branch-v1.json | W 与 lr4 的评测配置 |
| grpo_lr_midpoint.py | 冻结契约、历史收据和收尾计数 |
| grpo_lr_study.py | 共用 r16 契约与顺序校验 |
| grpo_repair_study.py | 学习率分支身份与 SFT 血缘 |
| grpo_common.py | 评测前校验导出配方 |
| grpo_repair_worker.py | r16 监督与批次接入 |
| grpo_dose_worker.py | 诊断持久化和独立审计 |
| run_v2_grpo_dose.py | train 筛选、完整验收与诊断调度 |
| validate_v2_grpo_lr.py | 原生输入和配方验证 |
| build_v2_grpo_dose_release.py | 发布清单和 ZIP 校验 |
| test_grpo_lr_midpoint_v2.py | 配方、分支、门槛和诊断回归 |
