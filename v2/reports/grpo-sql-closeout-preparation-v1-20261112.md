# r16 收尾维护样本对照实现与运行

从 W 独立训练一档，学习率 4e-6、60 步。原 192 个目标动作例次和 48 个维护动作例次保持相同，按原 r16 批次索引替换 12 条新证据检查和 12 条检查后提交的维护样本。[冻结契约](grpo-sql-closeout-contract-v1-20261112.md)

## 本地验证

60 条监督动作完成原生 tokenizer 编码和 SQLite 重放。最长提示 1988 tokens，最长响应 142 tokens。24 个原始探针、r16 批次索引与目标样本均逐项匹配。每条新增检查样本的正确动作只改一个证据 ID，每条提交样本复用通过检查的参数。[验证记录](grpo-sql-closeout-validation-v1-20261112.json)

9 个相关 GRPO 模块的 156 项回归通过，干净解压目录的 51 项定向测试通过。四个 CLI 入口、模拟中断和续跑、累计计时及预算收尾通过。发布包核验 405 个条目和 404 项文件哈希。

## 云端运行

将 `grpo-sql-closeout-v1-20261112-run.zip` 上传到云端仓库根目录，即 `v2/` 的上一级，然后执行。

~~~bash
(
set -e
unzip -o grpo-sql-closeout-v1-20261112-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_lr.py --config configs/grpo-sql-closeout-v1.json
)
~~~

中断后，从 `v2/` 给 Python 命令加 `--resume`。训练、探针、评测、审计和打包在终端显示进度。参考上一轮 L40S 用时，止于 train 筛选预计 40–45 分钟，进入完整验收预计 1 小时 15 分至 1 小时 30 分。上限为 2 小时 30 分。

作业结束后回传终端 `Download` 指向的 review ZIP 和 `Receipt` 指向的 JSON。标准轨迹的收尾计数保存在 `completion-diagnostics.json`。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo-sql-closeout-v1.json | 单档实验、历史收据和剩余预算 |
| grpo-sql-closeout-branch-v1.json | W 和候选评测配置 |
| grpo_closeout_study.py | 新维护样本生成和冻结契约 |
| grpo_repair_study.py | 候选评测血缘 |
| grpo_repair_worker.py | 接入替换后的监督样本 |
| grpo_lr_midpoint.py | 共用标准轨迹收尾诊断 |
| grpo_dose_worker.py | 沿用原探针并独立审计 |
| run_v2_grpo_dose.py | 训练、筛选和验收调度 |
| validate_v2_grpo_lr.py | 验证数据、索引顺序和原生 SQL |
| build_v2_grpo_dose_release.py | 打包并核验发布清单 |
| test_grpo_closeout_v2.py | 替换范围、证据修复和门槛回归 |
| test_grpo_lr_midpoint_v2.py | 共用诊断与中间学习率兼容性回归 |
