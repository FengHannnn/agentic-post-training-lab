# Agentic Post-Training Lab

基于 Qwen2.5-1.5B-Instruct 的多轮 SQLite 业务对账 Agent。模型调用 SQL 查询当月与上月净收入，计算差额，检查并提交带证据的结果。项目覆盖加权 SFT、DPO 和在线 GRPO，使用 vLLM 采样、LoRA 更新和 SQLite 语义验收。

## Fine GRPO 独立复评

以加权 SFT 和 DPO 后的 W 为起点，对比两种逐动作信用分配方案。12 个独立任务族使用相同起点和解码设置进行配对评测。

| 标准指标 | 基线 W | 原 local Fine GRPO（RL43） | 新收尾信用（RL47） |
| --- | ---: | ---: | ---: |
| 完整采样 | 24/48 | 28/48 | 33/48 |
| 恢复采样 | 177/252 | 183/252 | 196/252 |
| 完整 greedy | 11/12 | 12/12 | 12/12 |
| 恢复 greedy | 66/84 | 67/84 | 70/84 |

原 local 在完整采样和恢复采样中分别比 W 多成功 4 条和 6 条，通过预设验收，入选 Fine GRPO 候选，W 作为正式基线。新收尾信用在两批固定训练轨迹上各新增 3 个有更新信号的组。新信用训练出的模型取得最高完整采样成绩和恢复采样成绩。三方复评实际用时 4 小时 42 分 59 秒，含审计和打包。

- [信用分配对照与固定轨迹重算](v2/reports/grpo-fine-closeout-credit-results-v1-20261112.md)
- [V2 代码入口与本地验证](v2/README.md)

## 其他实验

- [当前范围与验收规则](v2/PROJECT_REQUIREMENTS.md)
- [目标监督剂量搜索结果](v2/reports/grpo-sql-dose-results-v1-20261111.md)
- [r16 收尾维护样本结果](v2/reports/grpo-sql-closeout-results-v1-20261112.md)
- [r16 中间学习率实验结果](v2/reports/grpo-sql-lr-midpoint-results-v1-20261112.md)
- [r16 热启动学习率实验结果](v2/reports/grpo-sql-lr-results-v1-20261111.md)
- [Warm-up首动作与别名诊断结果](v2/reports/grpo-warmup-probe-results-v1-20261110.md)
- [Warm-up首动作与别名诊断方案](v2/reports/grpo-warmup-probe-preparation-v1-20261110.md)
- [较小热启动实验结果](v2/reports/grpo-sql-retention-results-v1-20261110.md)
- [较小热启动与标准能力维护方案](v2/reports/grpo-sql-retention-preparation-v1-20261110.md)
- [SQL 认证修复门控实验结果](v2/reports/grpo-sql-repair-results-v1-20261110.md)
- [SQL 认证修复与门控实验方案](v2/reports/grpo-sql-repair-preparation-v1-20261109.md)
- [最新 alias 课程结果](v2/reports/grpo-alias-curriculum-results-v1-20261109.md)
- [本次代码与 Git 整理记录](v2/reports/project-consolidation-20261109.md)

### 实验沿革

| 阶段 | 已完成工作 | 当前结论 |
| --- | --- | --- |
| SFT / DPO | SQL 加权 SFT、恢复偏好对照、短程 DPO 收尾 | W 为 GRPO 正式起点，血缘 SFT513 / DPO11 / RL0 |
| GRPO 工程 | 在线 K 轨迹采样、真实 LoRA 更新、策略刷新、事务恢复与概率审计 | 训练闭环已验证 |
| GRPO 效果 | outcome / fine、独立确认、信用与课程对照 | W 为正式基线，旧 local Fine GRPO 通过 12 任务族独立复评 |
| 最新 alias 课程 | 每支 48 组，control / curriculum 实际更新 25 / 37 次 | 课程支标准恢复 65/84，W 为 61/84；完整任务均 14/16；本轮质量验收未通过 |
| SQL认证修复门控实验 | 新版认证、96条W诊断、SFT热启动24步与W/B评测；全流程1小时33分 | B复合sample从0/12升至8/12；标准完整14→13/16、恢复58→52/84，B门槛收尾，保留W |
| 较小热启动结果 | 全12任务维护、SFT15步；标准完整13/16、恢复58/84，目标0/12 | B门槛收尾，RL更新0；继续修复warm-up，保留W；两轮累计2小时38分 |
| Warm-up诊断结果 | W/B24/B15的train首动作认证0/11/0（各/12），维护单步均12/12 | B15在train仍偏向省略AS；优先目标剂量单变量对照；三轮累计2小时45分 |
| 目标剂量结果 | r4/r8/r16首动作认证均0/12；AS分叉差-3.47/-2.92/-1.55，SFT实际24/36/60步 | train门槛收尾，保留W；新增6小时已用48分18秒，下一步学习率单变量 |
| r16 学习率搜索 | 3e-6 两次 train 首动作认证 7/12、9/12，5e-6 均 12/12，开发集复合采样终局 10/12 | 5e-6 标准完整采样 13→8/16、恢复采样 60→48/84，保留 W |
| r16 中间学习率 | 4e-6 两次 train 首动作认证均 12/12，开发集复合采样终局 10/12 | 标准完整采样 13→9/16、恢复采样 60→52/84，保留 W |
| r16 收尾维护样本 | 复合采样终局 10/12，标准完整和恢复采样为 10/16、53/84 | 标准采样仍低于同轮 W，保留 W |
| Fine GRPO 收尾信用 | 开发完整采样 W 9/16 → 新信用 12/16，恢复 58/84 → 70/84 | 开发验收通过，独立复验主指标 8/16 持平，保留 W |

## 目录与版本边界

| 路径 | 用途 |
| --- | --- |
| v2/src/apt_v2/ | V2 任务、环境、训练目标与评测核心 |
| v2/scripts/ | 训练、采样、评测、审计与发布入口 |
| v2/configs/ | 各轮冻结配置 |
| v2/tests/ | CPU 回归测试与固定证据样例 |
| v2/reports/、v2/handoff/ | 结果、契约、测试记录与发布回执 |
| v2/hotfixes/ | 历史作业修复与迁移证据 |
| v2/artifacts/ | 本地训练数据、权重、状态和审计回包 |
| src/agentic_post_training/、scripts/、tests/ | V1 历史实现与回归测试 |
| reports/ | V1 历史报告 |

## 本地验证

仓库根目录已有 .venv。V1 回归命令：

~~~powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v
~~~

V2 使用独立测试目录；SQLite 版本及命令见 [V2 本地验证](v2/README.md#本地验证)。云端执行采用各轮实际发布包中的冻结入口、配置和预算。

## Git 与历史留存

本仓库以一个 Git 根目录管理 V1 和 V2。源码、配置、测试、报告和发布清单进入 Git。V2 的权重、优化器状态、完整轨迹、ZIP 和本机运行库留在本地。已入 Git 的 V1 历史证据继续保留。

2026-11-09 补交分为 V1 遗留工作、V2 整理前快照、最新 GRPO 源码与文档三部分。每次提交使用实际提交时间，文件中的实验日期保持原值。旧版根 README 已保存在 V1 补交提交中。

V1 的历史结果和归档可从以下记录查阅。[最后一轮均衡示范结果](reports/phase5-balanced-warmstart-results.md) · [V1 封存回执](v2/V1_ARCHIVE_RECEIPT.json)
