# GRPO Outcome Curriculum Same-Forward Probe Repair

本次停止仍发生在 collect-control-0018。第一版补丁已经生效；其“同 seed 重复解码的概率近乎逐位一致”假设与本次运行记录不符。第二版改为核对同一次采样前向的 logits、原生采样器概率和记录的行为 q。

项目路径：reports/grpo-outcome-curriculum-same-forward-repair-20261108.md

云端路径：/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-outcome-curriculum-same-forward-repair-20261108.md

**故障和保留进度**

实际失败请求的提示长 313 tokens，响应长 22 tokens，首动作为 read_policy。跨请求 prefill 与 decode 的最大 log-probability 差为 0.3287812173；同 seed 重复解码得到相同响应 tokens，概率最大差仍为 0.1711665392，超过第一版 1e-5 门槛。这是概率验证路径失败。第 18 组尚未提交。

两支各已提交 18 组、完成 10 次实际更新，最新检查点均为 checkpoints/group-0014。control 权重 SHA256 为 08bfd26ccb54a13195fd403baafbe63f1ec6b1c8f23946f076c5b5cf143ccfb3；lr10x 为 01654b067de2d537b9e42de003b226ddc48837f2dd62794ba3547c194b4abcf6。176 个已完成训练与评测文件在两次回包中逐字节一致。

累计用时 6780.423881 秒，约 1 小时 53 分钟。原 6 小时预算剩余 14819.576119 秒，约 4 小时 7 分钟，包含原有 180 秒打包预留。配置、学习率、奖励、任务、K=4、优化目标及效果验收沿用本轮原设置。

**第二版验证方式**

每个采样组创建新 vLLM 引擎，在预热完成后安装观察器。它只观察首动作的前 16 个响应 tokens；响应更短时检查全部 tokens。观察器复制同一次 sampler.forward 的原始 logits，核对实际温度为 1、无截断、无惩罚，再确认进入 TopKTopP 采样器的整个 logits 张量与原始 FP32 logits 完全一致。

随后对复制的原始 logits 独立执行 FP32 log_softmax，核对选中 token 的概率、原生采样器返回概率和 generation 中记录的 q，最大误差门槛为 1e-5。FP64 的 logit−logsumexp 另存为精度诊断。全词表测试发现 FP64 与原生 FP32 的差可达 3.14e-5，因此准入比较采用采样器实际使用的 FP32 运算。

首动作生成后立即读取观察记录并校验，接口或概率异常会在首动作停止。4 条轨迹生成完成后，观察记录再与原有 112 个 LoRA 张量证明绑定；提交组和后续审计均独立复核该记录。观察器沿用原 sampler 返回结果，测试同时检查 logits、输出和 RNG 状态保持一致。原行为 q 继续用于 TIS 与 GRPO。

原 prefill 探针及 0.25 诊断阈值继续记录；新的准入证据来自同一次前向。第一版的跨请求重复解码从本轮生产采样路径移除，已有组仍按各自记录的验证方式复核。每组仅保留原有 1 个 prefill 探针生成 token。两支统一应用此修订。

核对依据是 [vLLM 0.11.0 Sampler 官方源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/sampler.py)和 [TopKTopPSampler 官方源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/ops/topk_topp_sampler.py)。源码中 processed_logprobs 使用 FP32 log_softmax；本补丁观察这条原生路径。

**验证记录**

完整 GRPO 回归 198 项通过，其中包括 23 项同次前向校验和 8 项新安装器测试。实际 attempt-002 夹具准确复现旧门槛失败。

直接执行 vLLM 0.11.0 官方 Sampler 类代码，并为依赖隔离提供 CPU Torch 采样后端；152,064 词表在 FP32、BF16、FP16 三种输入精度下均通过，新概率校验最大差均为 0。

第 18 组的 4 条记录响应通过真实 Qwen tokenizer、SQLite 环境以及完整 collect/commit/verify 入口；33 个动作、3400 个响应 tokens 与原轨迹对应，奖励一致。该端到端检查用合成 one-hot logits 驱动 CPU 采样器，验证观察器和证据绑定的完整连接。

真实回包副本完成安装：590 个保护文件哈希一致，原始、第一版与第二版源码快照各三处通过核验，重复安装通过；原 36 个已提交训练组再次通过新审计器回放。实际 L40S 上的同次前向检查在续跑首动作执行。

安装器核对第一版补丁的有效发布 manifest 和三份运行 manifest，保存原始、第一版和第二版源码快照。云端安装额外核对最新 state.pt 中的 Adam/RNG 文件哈希；安装前后保护文件树必须一致。补丁来源记录位于 artifacts/grpo-outcome-curriculum-v1-20261107/source_revisions/grpo-probe-same-forward-v2/。

**安装与续跑**

上传第二版 ZIP 后解压到 /root/rivermind-data/AgenticPostTrainingLab，再运行：

```bash
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u hotfixes/grpo-probe-same-forward-v2/apply.py
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_curriculum.py --config configs/grpo-outcome-curriculum-v1.json --resume
```

继续使用原实例的 artifacts、检查点与 state.pt；安装器沿用原预算。第 18 组按原计划和 seed 重新采样。终端继续显示 rollout、score/update 和总进度条。下一次回包文件为 grpo-outcome-curriculum-v1-20261107-review-attempt-003.zip。

**代码阅读顺序**

| 相对路径 | 大致作用 |
|---|---|
| scripts/grpo_sampler_probe.py | 观察同次前向的 logits 与原生概率，校验 q |
| scripts/grpo_vllm.py | 安装 worker 观察器、首动作检查和概率探针 |
| scripts/grpo_worker.py | 生成 4 条轨迹，绑定张量证明并提交采样组 |
| hotfixes/grpo-probe-same-forward-v2/apply.py | 校验旧状态、保存源码修订、安装补丁 |
| tests/test_grpo_sampler_probe_v2.py | 原生接口、全词表精度、RNG 和概率篡改测试 |
| tests/test_grpo_sameforward_hotfix_v2.py | 连续修补、预算及检查点保护测试 |

回包 SHA256：39a8aff80edd79c736385f6cf3152efb970f50d14739078a0532f8b7f138022b
