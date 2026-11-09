# DPO 阶段首轮：随机轨迹与偏好数据

日期：2026-11-02（Asia/Shanghai）。用户选择：先采样现有 12 个训练任务，再构建同一上下文下的动作偏好对。代码、测试和 L20 采集包已完成。

## 本轮执行内容

从冻结 SFT sft-sql-weighted-v1-20261102 出发，每个 train 任务随机采样 4 条完整轨迹，共 48 条。temperature=0.8，top_p=1，top_k=-1；每动作最多 256 token，上下文上限 8192。每条轨迹及每轮使用记录在案的不同伪随机种子。

策略固定为累计 126-step adapter，SHA-256：e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0。云端直接从 artifacts/sql-sft-weighted-v1-20261101/final/adapter 加载，并在采集前后校验摘要。原模型和 SFT 权重保持固定，本轮 optimizer_updates=0。

使用原有 L20 环境和 Qwen2.5-1.5B 缓存。总预算 3600 秒，prepare、collect、build 共用；超时或失败保留已有日志和轨迹并打包。

## 偏好对的定义

标签粒度为单个动作。每对共享完全相同的 messages 和原生 prompt token：

- rejected：冻结 SFT 在该上下文实际采样的动作。
- chosen：脚本专家根据相同执行状态生成的修复动作，在从同一历史重建的独立环境中执行验证。

接纳的 rejected 类型为工具执行错误、SQL 执行后没有已验证指标、检查或提交未通过。chosen 要取得新的规则/schema/指标进展，或通过检查/提交。预设首次查询临时故障单独排除，保持偏好标签对应动作质量。

例如模型查询返回 verified_metrics=[]，脚本在同一查询前状态给出正确月度 SQL，实际执行并验证后，保存两种动作及各自的工具观察。偏好标签描述这一步的验证结果，完整轨迹的终局成功率另外报告。

每条轨迹最多提取 2 对，每任务最多保留 4 对，按 messages/chosen/rejected 摘要去重。原始轨迹全部保留，最终最多 48 对。训练编码上限 2048，超长上下文整体跳过，不截断证据链；rejected 使用实际生成的原生 token，包含真实 EOS。截断输出单独留在轨迹中，不进入本轮偏好对。

偏好数据复核就绪指标为至少 12 对、覆盖至少 4 个训练任务。数量不足时生成对应统计，下一步据此调整采样或任务覆盖。development/test 保持在本轮采集之外。

## DPO 损失内核

已实现 response_logps、encode_pair 和 dpo_loss，并用微型真实 Qwen + LoRA 验证梯度与更新。标准目标使用 action response 的序列 logprob 总和：

```text
-log sigmoid(beta * ((logπ(chosen|x) - logπ(rejected|x))
                   - (logref(chosen|x) - logref(rejected|x))))
```

prompt 和工具观察作为条件，只有动作 response 参与概率求和；reference 分数保持 detached。正式 1.5B DPO 更新在偏好数据复核后配置学习率、beta、批量与有限步数，策略和 reference 均从相同冻结 SFT 建立。

本轮保存原始模型 logprob，用于数据身份和后续评分核对；采样记录标记 dpo_sample，behavior_token_log_probs 留空。DPO 采用策略/reference 重算概率；后续 GRPO 使用具备真实行为概率契约的独立采集接口。

## 云端运行

上传 v2-dpo-collect-v1-20261102.zip 至 /root/rivermind-data/ 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
ls -lh /root/rivermind-data/v2-dpo-collect-v1-20261102.zip
unzip -o /root/rivermind-data/v2-dpo-collect-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_dpo_collect.py
```

另一个终端查看采样进度：

```bash
tail -f /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/dpo-collect-v1-20261102/collect.log
```

回传：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-collect-v1-20261102-review.zip。

回传包含完整随机轨迹、采样身份、同上下文偏好对、分支验证观察、汇总、审计、源码与 tokenizer，不重复携带模型权重和 optimizer。

## 本地验证

V2 全部测试 79 passed，新增 DPO 测试 9 项。覆盖同上下文分支执行、可执行但错误的 SQL、临时故障排除、原生 token/EOS 对齐、重复和 train 隔离、格式错误负例、轨迹篡改检测、DPO 损失及梯度方向、微型真实 Qwen + LoRA 更新后偏好 margin 提升、冻结底座/reference、完整偏好构建审计和失败留证。

另用历史 63-step 训练失败样例作真实 Qwen tokenizer 编码检查，验证上下文与原生 rejected token 对齐；正式本轮数据由冻结 126-step SFT 在云端重新采集。

## 代码与产物

- src/apt_v2/dpo.py：状态修复动作、同状态分支验证、偏好去重、token 编码和 DPO 损失。
- scripts/collect_v2_dpo.py：准备输入、48 条随机轨迹、偏好构建与审计。
- scripts/run_v2_dpo_collect.py：总入口、时间预算、日志和回传打包。
- configs/dpo-collect-v1.json：本轮采集配置。
- tests/test_dpo_v2.py：DPO 核心和数据流程测试。
- artifacts/releases/dpo-collect-v1-20261102/：发布包与凭据。

报告保存于项目 v2/reports/dpo-collect-v1-20261102.md。
