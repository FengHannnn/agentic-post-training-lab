# Fine GRPO 收尾信用实验运行说明

按冻结契约从 W 独立训练旧 local 和新收尾信用两支。统一使用 `sql_semantics_v2`、W reference、`K=4`、学习率 `1e-6`，每支最多 48 组。冻结 W 先评测并完成最多 25 分钟的训练起点预检，两支采用同一份预检后计划。新增收尾信用按检查和提交分配，每条轨迹上限 0.2。[实验契约](grpo-fine-closeout-credit-contract-v1-20261112.md)

## 上一阶段热启动的瓶颈

- 较小更新和较低目标剂量的 B15 在 train 和开发复合起点仍为 0/12。固定 2e-6 增加目标剂量后，r16 的平均 AS 分叉差由 W 的 −4.29 改善到 −1.55，首动作选择 SQL 增至 11/12，生成仍省略别名。条件概率改善先于可执行动作的变化。[目标剂量结果](grpo-sql-dose-results-v1-20261111.md)
- 固定 r16 改学习率后，4e-6 和 5e-6 都取得两次 train 12/12 和开发复合采样终局 10/12。标准任务的瓶颈转到多步收尾。4e-6 的 17 条配对退化里，14 条已认证三项指标，17 条均耗尽动作且没有提交。[中间学习率结果](grpo-sql-lr-midpoint-results-v1-20261112.md)
- 24 条收尾替换样本保住复合收益，标准完整和恢复采样各增加一条。候选在同一开发起点仍连续引用未认证的旧证据 ID。训练前缀里的旧证据来自缺别名查询，该开发前缀的旧证据来自业务口径错误的查询。维护的单步正确率与多步收尾分别评估。[收尾样本结果](grpo-sql-closeout-results-v1-20261112.md)

这些热启动轮次实际 RL 更新为 0。信用公式的影响由本轮两支在线训练直接比较。

## 验证

本地原生 SQLite 重放两套任务面板的专家动作，校验完整评测与专项评测计划，并用原生 tokenizer 校验 12 个预检起点的专家续跑。历史 local 的 46 组重算验证两种公式仍有 37 个有效组，32 次首次有效检查、32 次成功提交由零优势转为正优势。[验证记录](grpo-fine-closeout-credit-validation-v1-20261112.json)

139 项相关回归和干净解压后的 15 项定向测试通过。四个 CLI、模拟中断和续跑、8 小时预算记账及回传打包通过。核验 418 个 ZIP 条目及 417 个绑定文件哈希。

## 云端运行

将 `grpo-fine-closeout-credit-v1-20261112-run.zip` 上传到仓库根目录，即 `v2/` 的上一级，然后执行。

~~~bash
(
set -e
unzip -o grpo-fine-closeout-credit-v1-20261112-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_fine_closeout.py --config configs/grpo-fine-closeout-credit-v1.json
)
~~~

同一作业中断后，在 `v2/` 给 Python 命令加 `--resume`。墙钟总上限为 8 小时，训练与评测终端有进度条。历史双分支用时约 5 小时 33 分。本轮增加 W 预检及可选独立复验，按实测耗时成对停止训练。结束后返回终端 `Download` 对应的 review ZIP 和 `Receipt` 对应的 JSON。

`selection.json` 保留信用对照的配对净变化、两支相对 W 的门槛和收尾计数，以及独立复验状态。每支实际 RL 更新数与已提交采样组数分别登记。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo-fine-closeout-credit-v1.json | 主配置和预算 |
| grpo-fine-closeout-credit-branch-v1.json | 共用训练和评测分支模板 |
| grpo_credit.py | 旧 local 和新收尾优势 |
| grpo_fine_closeout.py | 课程、候补、面板和选择 |
| grpo_curriculum.py | 接入训练与评测的上下文 |
| grpo_repair_study.py | 冻结当前 verifier 的任务与评测计划 |
| grpo_common.py | 方法和分支配置校验 |
| grpo_repair_worker.py | W 单会话预检及部分组重放 |
| run_v2_grpo_repair.py | 共用阶段预算与续跑基础 |
| run_v2_grpo_fine_closeout.py | 双分支训练、复验与审计 |
| build_v2_grpo_fine_closeout_release.py | 原生验证和发布核验 |
| test_grpo_fine_closeout_v2.py | 信用、课程与预算边界回归 |
