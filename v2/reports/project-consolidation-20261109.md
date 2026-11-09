# Project consolidation 2026-11-09

主项目已合入最新 alias / hard 源码、配置、测试和补丁，补齐 fine / hard 结果报告及两份原始交接。根 README、V2 README 和需求文档现已指向当前 GRPO 阶段；正式基线继续保留 W。

## 整理结果

| 工作 | 结果 |
| --- | --- |
| V1 遗留补交 | 55 个文件；源码、配置、报告、发布清单与原根 README |
| V2 整理前快照 | 517 个现有文件进入 Git；另保存忽略规则与字节保留规则 |
| 核心源码差异 | 19 个缺失文件补齐，6 个旧文件同步最新版本 |
| 最新回包导入 | 共 42 个新增 / 更新文件，包含 25 个核心差异、10 个补丁文件、6 个报告 / 验证文件及发布清单 |
| 结果与交接补齐 | 18 个补充报告 / JSON，加 2 份原始交接 |
| 项目入口 | README.md、PROJECT_REQUIREMENTS.md 与仓库根 README 已更新 |
| 正式模型 | W，SFT513 / DPO11 / RL0；保留全部历史实验结论 |

本轮整理采用第三次 alias 回包的实际源码，并保留 midpoint 与 quality-outcome 两次迁移记录。训练 verifier 的空退款边界和重复 greedy 数值机制继续作为后续工作登记。

## 来源与身份

来源回包为 grpo-alias-curriculum-v1-20261109-review-attempt-003.zip，290041670 bytes，SHA256 为 4a97429fae19131191e0dec3f53e21922091644ead514dddaacdbcf5ac3c1c34。本轮重新计算 ZIP SHA256，与已有回执一致。

发布清单 grpo_alias_release_manifest.json 的 291 个文件绑定在来源树和整理后的主项目中逐项通过。导入文件的 before / after SHA256、补充报告、原始交接和 SQLite DLL 身份保存在 handoff/grpo-source-import-20261109.json。

两份临时附件从原交接 ZIP 恢复，ZIP CRC 检查通过；原文写入 handoff/PROJECT_HANDOFF_V2_SFT_DPO_RESULTS_2026-11-09.md 和 handoff/PROJECT_HANDOFF_V2_GRPO_CONTINUATION_2026-11-09.md。原文中的待同步状态按本次整理结果更新理解。

## Git 提交与留存

| 提交 | 内容 |
| --- | --- |
| eea5d34 | 补交 V1 policy iteration、采样探索、奖励对照和示范热启动等遗留工作，共 55 文件 |
| 7f29ff6 | 保存整理前 V2 实现、配置、测试、报告、历史交接与 Git 规则，共 519 文件 |
| 本报告所在提交 | 导入最新 GRPO 源码、补丁与报告，更新项目入口 |

提交均使用实际整理日期，实验文档保留原实验日期。整理前版本可用 7f29ff6 读取；旧版根 README 保存在 eea5d34。Git 历史按新增提交顺序保留。

V2 运行产物、模型、Adam/RNG、完整轨迹、ZIP 和本机 DLL 继续存放本地，由 Git 忽略。artifacts/README.md 记录产物用途；发布清单记录输入与源码绑定。V1 已跟踪的历史证据继续保留。

V2 文件设置为字节保留，Git 格式检查按原始 CRLF / LF 行尾识别；干净检出的 281 项冻结源码与报告 SHA256 全部一致。Git 跟踪源码和文档；完整实验回包继续按各轮归档留存。

## 本地验证

本地环境为 Python 3.11.5、Torch 2.14.0+cpu、Transformers 4.57.6、PEFT 0.17.1；历史云端环境继续使用各轮 execution_environment。

- V1 全量回归：435 passed，1 条 PEFT 配置提示，59.92 秒。
- V2 全量回归：478 passed，4 条 PEFT 配置 / adapter 提示，966.17 秒（16 分 6 秒）。
- Python 语法检查：289 个文件通过，覆盖 src/、scripts/、tests/、diagnostics/ 和 hotfixes/。
- 严格 SQLite 运行库：3.54.1，位于 .local/sqlite3541/sqlite3.dll，SHA256 为 4b59d3cded9dc1608544160f3b4b10ac974c6ac0d6ffc23064e5f4c8dbd05aea。
- 新入口文档链接、Git 改动格式及索引字节一致性检查全部通过。
- 结构化验证结果：reports/project-consolidation-validation-20261109.json；原始测试日志：.local/project-consolidation-20261109/。

本轮执行本地整理与 CPU 验证。下一实验围绕 SQL 认证和首个有效动作设计对照、冻结预算。既有实验脚本中的终端进度条随源码保留。

## 本轮新增 / 改动代码入口

| 相对路径 | 作用 |
| --- | --- |
| src/apt_v2/grpo_alias_curriculum.py | 缺别名两阶段课程与定向评测 |
| src/apt_v2/grpo_hard_preflight.py | 困难恢复两档起点与冻结诊断 |
| src/apt_v2/grpo_curriculum.py | 将 alias 起点、任务与汇总接入公共课程分派 |
| scripts/run_v2_grpo_alias_curriculum.py、grpo_alias_common.py、probe_v2_grpo_alias.py | 双支调度、中点诊断与预算约束 |
| scripts/run_v2_grpo_hard_preflight.py、grpo_hard_common.py、grpo_hard_worker.py | 冻结困难预检执行链 |
| scripts/evaluate_v2_grpo.py、audit_v2_grpo.py、audit_v2_grpo_alias.py | 质量负结果收尾、合法零信号和验收审计 |
| scripts/grpo_common.py、grpo_worker.py、grpo_vllm.py | alias 配置、回放与同次前向概率证据接入 |
| scripts/build_v2_grpo_alias_release.py、build_v2_grpo_hard_release.py | 对应历史作业发布入口 |
| configs/grpo-alias-*.json、grpo-hard-recovery-preflight-v1.json | 对应冻结配置 |
| tests/test_grpo_alias_*.py、test_grpo_hard_preflight_v2.py | 课程、事务、质量和预检回归 |
| hotfixes/alias-midpoint-fix-v1/、alias-quality-outcome-fix-v1/ | 两次历史修复的安装器、payload 与迁移记录 |

逐文件清单及哈希以 handoff/grpo-source-import-20261109.json 为准。
