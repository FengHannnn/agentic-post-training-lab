# Agentic Post-Training Lab

面向大模型后训练与工具调用 Agent 的可复现实验项目，覆盖 SFT、DPO、GRPO、vLLM 采样、错误恢复与逐动作信用分配。

当前主线是 **V2：多轮 SQLite 业务对账**，模型为 Qwen2.5-1.5B-Instruct。V1 算术与工具调用实验已封存，源码和结果保留在仓库根目录。

## 从这里开始

- [V2 项目说明与本地验证](v2/README.md)
- [当前范围与验收规则](v2/PROJECT_REQUIREMENTS.md)
- [GRPO 后续工作交接](v2/handoff/PROJECT_HANDOFF_V2_GRPO_CONTINUATION_2026-11-09.md)
- [SFT / DPO 结果交接](v2/handoff/PROJECT_HANDOFF_V2_SFT_DPO_RESULTS_2026-11-09.md)
- [较小热启动实验结果](v2/reports/grpo-sql-retention-results-v1-20261110.md)
- [较小热启动与标准能力维护方案](v2/reports/grpo-sql-retention-preparation-v1-20261110.md)
- [SQL 认证修复门控实验结果](v2/reports/grpo-sql-repair-results-v1-20261110.md)
- [SQL 认证修复与门控实验方案](v2/reports/grpo-sql-repair-preparation-v1-20261109.md)
- [最新 alias 课程结果](v2/reports/grpo-alias-curriculum-results-v1-20261109.md)
- [本次代码与 Git 整理记录](v2/reports/project-consolidation-20261109.md)

## 当前结果（2026-11-10）

| 阶段 | 已完成工作 | 当前结论 |
| --- | --- | --- |
| SFT / DPO | SQL 加权 SFT、恢复偏好对照、短程 DPO 收尾 | W 为 GRPO 正式起点，血缘 SFT513 / DPO11 / RL0 |
| GRPO 工程 | 在线 K 轨迹采样、真实 LoRA 更新、策略刷新、事务恢复与概率审计 | 训练闭环已验证 |
| GRPO 效果 | outcome / fine、独立确认、信用与课程对照 | 正式基线继续保留 W |
| 最新 alias 课程 | 每支 48 组，control / curriculum 实际更新 25 / 37 次 | 课程支标准恢复 65/84，W 为 61/84；完整任务均 14/16；本轮质量验收未通过 |
| SQL认证修复门控实验 | 新版认证、96条W诊断、SFT热启动24步与W/B评测；全流程1小时33分 | B复合sample从0/12升至8/12；标准完整14→13/16、恢复58→52/84，B门槛收尾，保留W |
| 较小热启动结果 | 全12任务维护、SFT15步；标准完整13/16、恢复58/84，目标0/12 | B门槛收尾，RL更新0；继续修复warm-up，保留W；两轮累计2小时38分 |

## 目录与版本边界

| 路径 | 用途 |
| --- | --- |
| v2/src/apt_v2/ | V2 任务、环境、训练目标与评测核心 |
| v2/scripts/ | 训练、采样、评测、审计与发布入口 |
| v2/configs/ | 各轮冻结配置 |
| v2/tests/ | CPU 回归测试与固定证据样例 |
| v2/reports/、v2/handoff/ | 结果、契约、来源记录与交接 |
| v2/hotfixes/ | 历史作业修复与迁移证据 |
| v2/artifacts/ | 本地训练数据、权重、状态和审计回包 |
| src/agentic_post_training/、scripts/、tests/ | V1 历史实现与回归测试 |
| reports/、PROJECT_HANDOFF*.md | V1 历史报告与交接 |

## 本地验证

仓库根目录已有 .venv。V1 回归命令：

~~~powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v
~~~

V2 使用独立测试目录；SQLite 版本及命令见 [V2 本地验证](v2/README.md#本地验证)。云端执行采用各轮实际发布包中的冻结入口、配置和预算。

## Git 与历史留存

本仓库以一个 Git 根目录管理 V1 和 V2。源码、配置、测试、报告、交接、发布清单进入 Git；V2 的权重、优化器状态、完整轨迹、ZIP 和本机运行库留在本地。已入 Git 的 V1 历史证据继续保留。

2026-11-09 补交分为 V1 遗留工作、V2 整理前快照、最新 GRPO 源码与文档三部分。每次提交使用实际提交时间，文件中的实验日期保持原值。旧版根 README 已保存在 V1 补交提交中。

V1 历史入口：[阶段交接](PROJECT_HANDOFF_2026-10-27.md)、[最后一轮均衡示范结果](reports/phase5-balanced-warmstart-results.md)、[V1 封存回执](v2/V1_ARCHIVE_RECEIPT.json)。
