# SFT 独立 test 评测 v1

日期：2026-11-02（Asia/Shanghai）。当前阶段：对 SQL 加权 SFT 的 126-step 候选做独立 test 评测；本地实现、验证与运行包已完成。

## 目标与固定输入

模型候选在既有 train 12/12、development 4/4 成功，现评测保留的 4 个 test 任务，其中 2 个有首次查询临时故障。test 的 ID、family 与训练任务隔离，训练数据来源在运行前核对。

固定候选为 sql-sft-weighted-v1-20261101 的 final adapter，累计 126 次更新，SHA-256：e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0。

同时评测 63-step 父 adapter，SHA-256：95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e。

两者使用同一个 vLLM BF16 引擎和原有 greedy 设置，seed=2026102901，max_model_len=8192，每动作生成上限 256。共 8 条完整轨迹，训练更新次数为 0。

## 记录与验收

预先记录的候选验收目标：4/4 成功、2/2 故障恢复、至少 8 条有效月度 SQL、4 次有效 check_result、4 次成功 submit，严格动作格式全部通过。

脚本保留所有成功和失败结果，不按得分自动重试。审计会重建原生 prompt、解码输出 token、执行工具并核对每轮观察、奖励、结束与恢复状态。评测前后核对源 adapter 和模型身份，保持权重一致。

脚本输出 eligible_for_local_review 判定；回传本地复核后再记录最终冻结决定。当前候选状态仍为 pending_independent_test。

## 云端运行

上传 v2-sft-independent-test-v1-20261102.zip 到 /root/rivermind-data/，执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
ls -lh /root/rivermind-data/v2-sft-independent-test-v1-20261102.zip
unzip -o /root/rivermind-data/v2-sft-independent-test-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_sft_test.py
```

依次运行 prepare、evaluate、audit，总预算 1200 秒并预留打包时间。终端逐阶段显示日志路径。查看推理进度：

```bash
tail -f /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/sft-independent-test-v1-20261102/evaluate.log
```

回传文件：/root/rivermind-data/AgenticPostTrainingLab/v2/sft-independent-test-v1-20261102-review.zip。

回传包保留配置、身份摘要、冻结源码、tokenizer、评测轨迹、统计和日志，跳过 adapter 权重与 optimizer 文件，减少下载体积。使用云端现有候选和模型缓存，复用 L20 环境。输出目录与 ZIP 独占创建，已有结果保留。

## 本地验证

V2 测试共 70 passed，源码编译通过。新增验证覆盖：固定四题 test、train/family 隔离、拒绝 development 配置或训练模式、拒绝缺题和重复任务、成功与失败的验收判定、完整八轨迹 token 上下文重放、等待本地复核的冻结状态，以及失败时保留日志且排除大权重文件。

## 代码阅读

本轮复用 evaluate_v2_sql_sft.py 的 paired evaluator 和 sql_sft.py 的 audit_episode，已有训练代码可跳过重复阅读。新增 src/apt_v2/sft_test.py 定义 test 隔离与汇总验收；scripts/test_v2_sft.py 组织输入校验、评测及审计；scripts/run_v2_sft_test.py 负责有限预算和回传打包；配置为 configs/sft-independent-test-v1.json。

报告位于项目 v2/reports/sft-independent-test-v1-20261102.md；发布包与凭据位于 artifacts/releases/sft-independent-test-v1-20261102/。
