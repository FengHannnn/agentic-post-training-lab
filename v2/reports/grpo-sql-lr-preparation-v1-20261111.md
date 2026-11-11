# r16 热启动学习率实验实现与运行

本轮按 3e-6、5e-6 顺序从 W 独立训练。每档固定 r16 的 60 步、192 个目标动作例次和 48 个维护动作例次。候选通过 train 筛选后进入完整 B 验收，首个通过者作为下一阶段热启动。具体阈值见 [冻结契约](grpo-sql-lr-contract-v1-20261111.md)。

## 本地验证

原生 tokenizer 编码 60 条动作，最长提示 1358 tokens，最长响应 142 tokens。SQLite 3.54.1 重放 24 个探针，专家响应均通过，缺别名反例保持未认证。两档批次逐项匹配 r16，SFT 配方仅学习率变化。验证记录见 [机器结果](grpo-sql-lr-validation-v1-20261111.json)。

全部 27 个 GRPO 模块共 379 项回归通过，干净解压目录的 105 项定向测试通过。发布包包含 376 个条目，375 个文件哈希与清单匹配。四个 CLI 入口和原生验证均复测通过，模拟中断、续跑、累计计时和预算收尾通过。

## 云端运行

上传 `grpo-sql-lr-study-v1-20261111-run.zip` 到云端仓库根目录，即 `v2/` 的上一级，然后执行：

~~~bash
(
set -e
unzip -o grpo-sql-lr-study-v1-20261111-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_lr.py --config configs/grpo-sql-lr-study-v1.json
)
~~~

同一作业中断后，从 `v2/` 给 Python 命令加 `--resume`。终端显示 SFT 例次、探针、评测、审计和打包进度。按上一轮 L40S 各阶段耗时估计，两档都在 train 筛选收尾约 45–55 分钟，执行一次候选完整评测约 1 小时 15–25 分，执行两次约 1 小时 45 分至 2 小时。本轮上限为 5 小时 10 分。

完成后返回终端 `Download` 对应的 review ZIP 和 `Receipt` 对应的 JSON。用收据和 `final_status.json` 核对预算与选择，再读取 `ready-B.json` 的来源和权重哈希。Adam/RNG 的 `state.pt` 保存在云端用于恢复。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo-sql-lr-study-v1.json | 候选顺序和剩余预算 |
| grpo-sql-lr-branch-v1.json | W 与 B 的评测配置 |
| grpo_lr_study.py | 学习率配方、r16 输入和预算契约 |
| grpo_dose_study.py | 共用顺序选择逻辑 |
| grpo_repair_study.py | 候选学习率与 SFT 血缘 |
| grpo_common.py | B 来源配方检查 |
| grpo_repair_worker.py | r16 顺序接入 SFT |
| grpo_dose_worker.py | 热启动、探针和独立审计 |
| run_v2_grpo_dose.py | 共用训练与验收调度 |
| run_v2_grpo_lr.py | 学习率实验入口 |
| validate_v2_grpo_lr.py | 输入、tokenizer 与 SQL 复核 |
| build_v2_grpo_dose_release.py | 发布清单和压缩包核验 |
| test_grpo_lr_v2.py | 配方、来源和有序选择回归 |
