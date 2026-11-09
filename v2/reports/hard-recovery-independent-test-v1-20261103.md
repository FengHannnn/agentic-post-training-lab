# 困难恢复 SFT：独立测试集评测说明

日期：2026-11-03。
运行 ID：hard-recovery-independent-test-v1-20261103。

本轮比较父模型 chosen-only SFT 与已审阅的困难恢复加权 SFT，在固定 test 任务上评测完整任务、已有恢复场景和新的双指标缺验证场景。评测只做推理，模型权重固定；训练更新为0。

## 模型与测试集

| 项目 | 设置 |
|---|---|
| 父模型 | chosen-only-sft-v1-20261102，累计SFT255／DPO11 |
| 候选模型 | hard-recovery-sft-weighted-v1-20261103，累计SFT513／DPO11 |
| 测试任务 | test-000～test-003，family-3000～family-3003 |
| 数据隔离 | task ID、family、seed与train/development分离；测试上下文均位于169条训练messages之外 |
| 数据与参数 | 读取已审阅训练来源，固定两个adapter SHA256 |
| 生成 | BF16 vLLM，max_new_tokens256，max_model_len8192，top_p1、top_k=-1，关闭prefix cache |
| 随机采样 | temperature=0.8；前后模型同设置、同种子 |
| 时限 | 14,400秒，4小时 |

父adapter SHA256：
078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55

候选adapter SHA256：
f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f

本轮沿用4个固定test任务，便于与历史结果比较；双指标缺验证组合为新增评测上下文。

## 评测范围

### 完整任务：72条轨迹

每个模型在4个test任务上：

- greedy每任务1条，共4条。
- 随机每任务8条，共32条。

两个模型合计72条。统计最终提交成功率、步骤数、工具错误、证据错误和前后配对结果。

### 已有恢复场景与首动作：288次

保留7类恢复场景：格式错误、未知工具、SQL未验证后重查、重复证据错误、已有正确证据时修正引用、连续SQL错误、检查失败后修复。

另增加2类场景：

1. both_sql_unverified：当月与上月SQL均执行成功，但verified_metrics为空。
2. both_sql_unverified_repeated_evidence：在上述状态下，进一步注入两次引用无效证据的calculate_change。

4任务×9场景=36个上下文。每个模型每上下文greedy1次、随机3次，合计288次首动作。分别统计工具选择、证据ID、SQL目标验证、严格格式与重复上一动作。

### 新场景完整恢复：64条续跑

对新增的8个上下文，除首动作外，继续运行模型直到提交或达到原任务12步预算。两个模型合计64条恢复续跑；首动作由续跑的第一步直接提取。

统计：

- 是否先后补齐两个指标并完成提交。
- 完整恢复成功率及前后配对改善／退步。
- 恢复所用步数及新增工具错误。
- 每步原生prompt／response token、概率、采样种子和环境观测。

首动作与最终恢复成功分别报告。新场景前缀占4～7步，专家路径均需5步完成，最长恰好在12步预算内完成。

## 跨月退款补充SQL校验

主表沿用当前Environment的成功判定。补充表对生成SQL检查两类数据库变化：

1. 让上月订单在当月退款、当月订单在上月退款。
2. 把目标地区／分群的paid订单退款移到当月，涵盖其他月份的原订单。

每个挑战数据库都重新计算正确指标，并执行模型生成的SQL。它检验退款是否按退款月份统计，以及SQL是否额外限制了原订单月份。原环境中的数值扰动校验继续沿用。

补充结果包括：

- 原环境已验证查询在跨月挑战中的通过情况。
- 最终提交所引用证据的SQL来源是否一致。
- 完整任务与新场景续跑各自的calendar_consistent_success数量。

最终来源会沿calculate_change证据继续追踪到两个查询。早期查询若存在口径问题，随后被正确查询替换，补充结果按最终提交所使用的来源评判，同时保留所有查询的诊断记录。

## 本地验证

已完成真实数据预检：

- 两个实际adapter哈希与已审阅模型一致。
- 4个test任务的ID、family、seed与其他划分分离。
- 36个测试上下文均未出现在训练messages中。
- 36个专家首动作全部有效，8个新增上下文的专家续跑全部成功。
- 最长专家路径prompt1458 tokens，加256生成预算为1714，低于8192上限。
- 标准metric_sql在两类跨月挑战中全部通过。

新增9项测试通过，覆盖测试集隔离、双指标恢复、跨月SQL、最终来源追踪、原生生成接口、端到端审计、检查点固定和失败回传。完整回归初次128项通过，另1项旧测试因错误提示文字匹配而失败；提示兼容性修正后，原失败项与训练／测试隔离保护补测2项均通过。129项测试覆盖完成，compileall通过。

为支持独立恢复评测，build_context增加显式test划分入口；diagnose_action增加显式上下文构造器和允许划分参数。默认训练准入仍为train，测试上下文进入make_pair时继续被拒绝。

## 运行命令

把 v2-hard-recovery-independent-test-v1-20261103.zip 上传到云端：

/root/rivermind-data/AgenticPostTrainingLab/v2/

然后执行：

```bash
unzip -o \
  /root/rivermind-data/AgenticPostTrainingLab/v2/v2-hard-recovery-independent-test-v1-20261103.zip \
  -d /root/rivermind-data/AgenticPostTrainingLab &&
cd /root/rivermind-data/AgenticPostTrainingLab/v2 &&
/root/rivermind-data/envs/apt-vllm/bin/python -u \
  scripts/run_v2_hard_recovery_test.py
```

ZIP内部包含v2/，解压目标为项目目录。阶段顺序：prepare → greedy → sample → recovery → audit。终端显示完整任务和恢复进度条，日志同步写入artifacts运行目录。恢复进度每完成一个首动作／完整续跑后更新。

完成后回传：

/root/rivermind-data/AgenticPostTrainingLab/v2/hard-recovery-independent-test-v1-20261103-review.zip

## 回传与审计

回传包包含测试任务与上下文清单、72条完整任务轨迹、288条恢复首动作、64条新场景续跑、分项计数CSV、跨月SQL校验详情、前后配对汇总、源码快照、来源哈希及审计结果。

审计逐条检查原生token与文本／上下文对齐、EOS与长度、采样种子、动作与观测、最终提交、恢复前缀和汇总计数。各阶段均重新核对两个模型的固定文件哈希，模型权重保留在原训练目录。

主要新增文件：

- src/apt_v2/independent_recovery_test.py：测试上下文、完整恢复重放与跨月SQL挑战。
- scripts/test_v2_hard_recovery.py：固定输入、完整任务、首动作与续跑生成。
- scripts/audit_v2_hard_recovery_test.py：独立重放、来源追踪和结果统计。
- scripts/run_v2_hard_recovery_test.py：运行时限、实时日志与回传打包。
- tests/test_hard_recovery_independent_v2.py：9项新增测试。

报告原件：v2/reports/hard-recovery-independent-test-v1-20261103.md。
预检结果：v2/artifacts/diagnostics/hard-recovery-independent-test-v1-20261103/preflight_summary.json。
