# SQL retention study v1 冻结契约

本轮从正式W重新开始，用较小SFT更新和覆盖更广的维护状态构建B；B通过原门槛后继续无提醒GRPO。run_id为grpo-sql-retention-study-v1-20261110。本契约与本地验证在云端执行前冻结。

## 依据与预算

上一轮结果见 reports/grpo-sql-repair-results-v1-20261110.md：B复合缺别名sample从0/12升至8/12，标准完整14→13/16、恢复58→52/84，按B门槛收尾。旧维护12条仅覆盖3个train任务。

本轮使用新run_id和新发布清单；复用上轮已审计的warm_start路线结论，冻结来源证据于 artifacts/inputs/grpo-sql-retention-20261110/prior-evidence.json。本轮重新评测W，固定执行一个SFT配方，不再重复96条路线诊断。

既有总预算21600秒，上轮回执已用5578.0134817191865秒，本轮所有attempt累计上限16000秒（4小时26分40秒）。两轮最大合计21578.013481719187秒，包含准备、训练、评测、刷新、审计与实际打包。resume继续累计本轮已用时间及归档收尾，外部receipt同时记录两轮合计时间。

## 数据和SFT配方

正式W为SFT513 / DPO11 / RL0，adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。

仍使用sql_semantics_v2、同一12个train和上一轮4个repair-development任务（000至003），动作总预算12，原生上下文8192、单次响应256。监督仅来自train；新数据逐条按真实环境重放前缀和正确动作。

| 每个train任务的状态 | 监督动作 | 数量 |
| --- | --- | ---: |
| 原复合缺别名 | 认证本月净额的query_sql | 12 |
| stale_evidence_repair | 重新calculate_change并使用有效月度证据 | 12 |
| failed_check_repair | 以正确值和证据重新check_result | 12 |
| repeat_sql_error | 正确query_sql，覆盖含临时错误的任务 | 12 |
| 已通过检查、尚未提交 | 使用真实证据submit | 12 |

合计60条，目标12、维护48；每个任务5条，每种维护状态覆盖全部12个任务。固定seed2026111053打乱一次，batch4、1epoch、15次AdamW更新、60例次、学习率2e-6、weight_decay0、梯度裁剪1。按响应token监督并保留EOS；只更新policy LoRA，实际FP32更新与checkpoint哈希逐步核验。每步保存adapter和Adam/RNG，中断恢复后续步骤。

相比上轮学习率降至1/5、更新次数24→15、数据例次96→60，目标监督例次48→12；训练配方一次固定。新B血缘为SFT528 / DPO11 / RL0。B与上一轮SFT537的权重分别记录。

## 评测与B门槛

W、新B以及最终C各执行标准132条加无提醒原复合16条，共148条episode；另有7个上下文×2次重复greedy质量动作。sample温度0.8，greedy温度0，配对任务与seed延续上一轮协议。

B须同时满足：

- 标准完整sample、标准恢复sample、标准完整greedy、标准恢复greedy四项成功数均不低于同轮W。
- 全部质量检查通过。
- 无提醒原复合sample首动作新增月度认证至少3/12，且高于同轮W。

B未通过时正常归档收尾，W保留正式基线。报告B−W的指标变化、配对新增成功和失败以及具体错误行为。

## 通过B后的GRPO

C从新B初始化，冻结B reference，新建Adam，SFT血缘528、DPO11，单独累计实际RL更新。最多36组，K4、local信用、每组最多1次更新、学习率1e-6；维护18组、原复合目标18组，始终无提醒。共9组完整维护、9组标准恢复维护，覆盖7种恢复场景与旧/新train池。

沿用已验证的同次前向采样概率、worker真实LoRA证明、可恢复组事务与objective独立重算。先前B的SFT Adam/RNG保留在其检查点，C只继承权重。

最终C须满足既有门槛：四项标准指标不低于W；主指标严格增加；原复合sample终局至少比W增加2/12且greedy不下降；全部质量检查通过。W完整sample若为16/16，主指标为恢复sample，否则为完整sample。另要求C相对B四项标准指标均不下降，并在至少一项标准或复合终局指标增加。

只有完整36组配方、实际RL更新且FP32权重和有效BF16投影均变化的C进入正式选优。报告B−W与C−B。B单独取得的改善登记为SFT阶段成果。

## 资源停止和审计

先用W本轮实际评测耗时估计后续开销。每个候选预留max(2400秒,1.25×W评测耗时)，另留刷新、审计、180秒打包。训练每2组为一个配对周期，周期开始检查预算，完整周期边界可提前停止并做收尾评测。不足36组时归档为预算结果。

基础设施错误保留部分证据、状态和review包，候选撤销。质量负结果按门槛收尾。源码、配置、数据、GPU UUID与软件版本绑定；恢复核验相同发布与状态。外部receipt记录最终ZIP哈希、实际打包后本轮时间及两轮合计时间。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo_retention_study.py | 固定配方、60条维护与目标动作、来源和预算验证 |
| grpo_repair_study.py | 新配方的SFT528分支身份校验 |
| grpo_common.py | 检查新B的15次更新与60例次血缘 |
| grpo_repair_worker.py | 新旧SFT配方分派、逐步保存、恢复和权重审计 |
| run_v2_grpo_retention.py | 重新评测W、构建B、门控C和累计预算收尾 |
| audit_v2_grpo_retention.py | 独立重放与B/C验收结果重算 |
| validate_v2_grpo_retention.py | 监督执行、原生token预算与掩码验证 |
| build_v2_grpo_retention_release.py | 发布包CRC、唯一条目和逐文件哈希验证 |
| grpo-sql-retention-study-v1.json | 主实验、较小SFT更新及总预算配置 |
| grpo-sql-retention-branch-v1.json | W/C共用GRPO分支配置 |
| test_grpo_retention_v2.py | 数据覆盖、门控、血缘、检查点和预算恢复测试 |
