# Agentic Post-Training Lab V2

V2 研究真实多轮工具交互中的 SFT、DPO 与 GRPO。模型通过 SQLite 查询完成月度业务对账，使用执行证据认证本月净额、上月净额和差额，再提交结果。金额使用整数 cents，标准任务总动作预算为 12，恢复前缀计入预算。

## 下一轮：SQL 认证修复与门控实验

sql_semantics_v2 与96条冻结W诊断已实现。预检后选择无提醒GRPO、提醒渐退对照或SFT→GRPO路线，总GPU墙钟上限6小时。云端训练按新发布包启动。

- [本轮冻结契约](reports/grpo-sql-repair-contract-v1-20261109.md)
- [本轮实现与运行说明](reports/grpo-sql-repair-preparation-v1-20261109.md)

## 当前状态（2026-11-09）

**正式基线为 W，下一轮 GRPO 从 W 独立初始化。** W 的血缘为 SFT513 / DPO11 / RL0；最终短程 DPO12 是独立阶段成果，累计 DPO23。

| 项目 | 状态与结果 |
| --- | --- |
| SFT / DPO | 阶段收尾完成，权重身份及结果见结果交接 |
| C2 / C3 | 在线 outcome / fine GRPO、策略刷新、恢复与严格审计完成 |
| C4 | 12 个新任务族独立确认完成，两候选效果验收未通过 |
| outcome / fine 后续对照 | 学习率、legacy / local 信用及困难预检完成 |
| alias 课程 | control / curriculum 各 48 组，实际 RL 更新 25 / 37 次 |
| alias 标准采样 | W / control / curriculum：完整 14/15/14（各 /16），恢复 61/60/65（各 /84） |
| alias 原复合起点 | 三模型采样均 1/12，greedy 均 0/4 |
| alias 质量 | 两支空退款 SQL 日历检查失败；control 重复 greedy 6/7 一致；候选为空，W 保留 |
| 真实依赖扩展 | period_closing 环境与专家因果证明完成；模型开发评测均 0/4，训练覆盖待安排 |

当前优先事项是修复训练 SQL verifier 的空退款边界，并围绕原复合起点首次有效认证设计新实验。新实验的干预、对照、任务、预算和选择规则在运行前冻结。

## 阅读顺序

1. [GRPO 后续交接](handoff/PROJECT_HANDOFF_V2_GRPO_CONTINUATION_2026-11-09.md)：算法、历轮结果和待解决问题。
2. [本次整理记录](reports/project-consolidation-20261109.md)：主项目同步、验证结果与 Git 提交。
3. [最新 alias 结果](reports/grpo-alias-curriculum-results-v1-20261109.md)：标准恢复收益、复合迁移和质量失败。
4. [fine 信用对照](reports/grpo-fine-credit-results-20261109.md)与[困难恢复预检](reports/grpo-hard-recovery-preflight-results-20261109.md)。
5. [SFT / DPO 结果交接](handoff/PROJECT_HANDOFF_V2_SFT_DPO_RESULTS_2026-11-09.md)及[当前需求](PROJECT_REQUIREMENTS.md)。

两份原始交接按原文保留。其中本地未同步事项的完成状态以本次整理记录为准。历史报告与冻结配置保留各自实验当时的状态。

## 模型与训练契约

| 项目 | 固定信息 |
| --- | --- |
| 基座 | Qwen/Qwen2.5-1.5B-Instruct |
| revision | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| W adapter SHA256 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| W 输入 | artifacts/inputs/grpo-w-20261105/adapter |
| LoRA | q_proj / v_proj，r8、alpha16、dropout0；112 张量，1,089,536 可训练参数 |
| 采样 / 更新 | vLLM BF16；HF / PEFT FP32；每组 K4，最多一次 optimizer.step |
| reference / 优化器 | 冻结 W；新实验独立新建 AdamW |
| 信用 | outcome、potential_rtg_depth_v1、outcome_local_verified_v2；逐轮配置明确登记 |

原生响应 token 与行为概率 q 绑定；HF 重算 old / ref，更新使用 TIS、clip 和 sampled k3。训练、刷新、评测和重放保留权重、概率及事务证据。每轮以自身配置和 execution_environment 为准。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| src/apt_v2/grpo.py、grpo_credit.py、grpo_optimizer.py | 采样、动作信用、概率评分与更新 |
| src/apt_v2/grpo_alias_curriculum.py | 两阶段缺别名课程与定向评测 |
| src/apt_v2/grpo_hard_preflight.py | 两档困难恢复诊断 |
| scripts/run_v2_grpo_alias_curriculum.py | 已完成 alias 作业的双支调度与恢复 |
| scripts/evaluate_v2_grpo.py、audit_v2_grpo_alias.py | 评测、质量结果与验收审计 |
| hotfixes/ | 历史补丁和迁移证据 |
| grpo_alias_release_manifest.json | 第三次 alias 回包的发布绑定与两次迁移记录 |
| handoff/grpo-source-import-20261109.json | 本次导入的逐文件 SHA256 和整理前提交 |

最新源码已整理进本目录。历史实验复核使用该轮冻结源码及 source_revisions；新实验采用新 run_id 和新发布包。

## 本地验证

在本目录执行。已有虚拟环境位于 ../.venv。严格回放使用 SQLite 3.54.1；本机运行库位于 .local/sqlite3541/sqlite3.dll，由 Git 忽略。

Windows PowerShell：

~~~powershell
& "..\.venv\Scripts\python.exe" -u -c "import ctypes; sqlite_runtime = ctypes.WinDLL(r'.local/sqlite3541/sqlite3.dll'); import sqlite3, pytest; assert sqlite3.sqlite_version == '3.54.1'; raise SystemExit(pytest.main(['tests', '-v']))"
~~~

已使用 SQLite 3.54.1 的 Python 环境：

~~~bash
python -u -m pytest tests -v
~~~

测试输出显示逐项进度；实验脚本保留终端 tqdm 进度条，覆盖采样、评分、反向、评测和重放。环境与本次验证结果见整理报告。

## 产物与版本管理

源码、配置、测试、报告、交接、补丁和发布清单由 Git 管理。artifacts/ 保存数据、adapter、Adam/RNG、评分缓存和回包；.local/ 保存本机运行库和验证日志，两者的运行内容由 Git 忽略。权重与原始 ZIP 按独立归档保留；Git checkout 恢复源码和文档，训练输入按发布清单恢复。

V2 文件采用字节保留规则，确保 Git checkout 后的 SHA256 与冻结发布一致。原始训练记录、评分和失败标记继续保留。新报告使用英文文件名、中文正文、项目内相对路径，不自动打开。
