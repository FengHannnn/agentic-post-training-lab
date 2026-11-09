# GRPO Outcome Curriculum Probe Repair

本轮在 collect-control-0018 的概率探针阶段停止。两支各已提交 18 组、完成 10 次实际 LoRA 更新；现有训练和 W 基线评测通过本地审计。修复后在原 L40S 实例沿用原配置与检查点，从组 18 续跑。

项目内路径：reports/grpo-outcome-curriculum-probe-repair-20261108.md

云端绝对路径：/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-outcome-curriculum-probe-repair-20261108.md

**失败原因和进度**

组 18 已生成 4 条完整轨迹，随后概率探针比较逐 token decode 的行为概率 q 与整段 prefill 的 prompt log probabilities。该请求的最大 log-probability 差为 0.328790456，超过 0.25 门槛，因此该组尚未提交，也尚未执行参数更新。该次未提交采样产生 3400 个响应 tokens 和 1 个额外探针 token，其用时已计入原预算。

实际失败的首动作是 read_policy，请求包含 id=revenue-p001，共 313 个提示 tokens、22 个响应 tokens。已保存此原生请求和概率记录作为回归夹具。原实现把两种计算路径的概率差作为单一准入条件；修复增加同请求、同 seed 的解码复核。

| 分支 | 已提交组数 | 实际更新次数 | 最新 checkpoint | adapter SHA256 |
|---|---:|---:|---|---|
| control，LR=1e-6 | 18 | 10 | checkpoints/group-0014 | 08bfd26ccb54a13195fd403baafbe63f1ec6b1c8f23946f076c5b5cf143ccfb3 |
| lr10x，LR=1e-5 | 18 | 10 | checkpoints/group-0014 | 01654b067de2d537b9e42de003b226ddc48837f2dd62794ba3547c194b4abcf6 |

已耗 6632.679 秒，约 1 小时 50 分 33 秒。原总预算仍为 21600 秒，剩余 14967.321 秒，约 4 小时 9 分 27 秒；其中继续保留 180 秒打包预留。安装补丁与续跑不会重置累计用时。

运行环境：NVIDIA L40S，UUID 2ac022d0-8ede-c73c-a21e-ccb5f7a3c88e；Python 3.11.17，Torch 2.8.0+cu128，vLLM 0.11.0，Transformers 4.57.6，PEFT 0.17.1。

**概率探针修复**

保留原 prefill 探针、0.25 门槛与实际最大差值。原门槛通过时沿用原验证；超限时，对同一 adapter、完整原始提示、原始 SamplingParams 和原始 seed 重复解码。

重复解码必须满足：完整响应 token 序列相同，结束原因相同，每个响应 token 的 log probability 与原行为 q 的最大差值不超过 1e-5。独立回包校验同时绑定请求 seed、提示 tokens、响应 tokens、采样参数、原 generation 摘要与额外探针 token 数。任何一项不满足仍会停止采样组。

原行为 q 原样用于 GRPO/TIS。探针重复解码仅作验证，额外 token 单列计费。原 prefill 差值不会被覆盖，验证方式与两条路径的结果一起写入 probability_probe。最终 112 张量 worker 检查继续绑定不可变 adapter。

本轮学习率、任务、奖励、K=4、LoRA、clip/KL/TIS 设置、组数和效果验收条件保持原值。两支统一使用修复后的验证逻辑。原 36 个已提交组与所有 Adam/RNG 状态、分数缓存、评测结果和配置保持原哈希；组 18 按原 seed 完整重采样 4 条轨迹。

**本地验证**

回包 ZIP 的 CRC 检查通过。两支 144 条已提交训练轨迹完成原生 token 与真实环境回放；20 次更新的目标函数重新计算，最大指标差分别为 5.3644e-8 和 3.7253e-8。W 的 132 条标准开发轨迹、24 条针对性恢复轨迹和 14 次质量动作通过复核。

原故障夹具在修复前准确触发 0.32879 > 0.25 的失败；修复后检查同请求重复解码。完整 GRPO 回归 160 项通过，安装器的云端 Adam 哈希、预算保护、检查点保护、日志恢复、重复安装与补丁身份检查共 7 项通过。

第 18 组的 4 条记录响应通过实际 Qwen tokenizer、真实 SQLite 环境和完整收集/提交入口回放，33 个动作与原部分文件完全一致；保留 prefill 差值 0.328790456，记录响应的同请求概率差为 0。此本地测试使用保存的原生响应；续跑时由 L40S 实际执行重复解码与 worker 张量验证。

真实回包副本的安装检查已通过：582 个本地保护文件安装前后哈希一致；原始与有效源码快照三处均通过核验；补丁重复安装检查通过。安装后两支原 36 个训练组再次通过新验证器回放。云端安装继续核对原实例上的 Adam/RNG 文件。

**安装与续跑**

修复 ZIP 顶层为 v2/hotfixes/grpo-probe-replay-v1/，解压目标是 /root/rivermind-data/AgenticPostTrainingLab。包内 payload 由安装器校验后写入对应路径。保留原实例上的 artifacts/grpo-outcome-curriculum-v1-20261107，尤其是 state.pt。

```bash
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u hotfixes/grpo-probe-replay-v1/apply.py
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_curriculum.py --config configs/grpo-outcome-curriculum-v1.json --resume
```

安装器先核对原发布 manifest、三份运行 manifest、所有回包绑定文件、两支 18 组事务和最新云端 Adam/RNG 文件哈希；保存原 manifest 与原源码，生成独立的有效源码快照，再更新源码绑定。安装前后保护文件树必须一致。

来源和安装记录保存在 artifacts/grpo-outcome-curriculum-v1-20261107/source_revisions/grpo-probe-replay-v1/。两支目录也各保存有效源码快照。终端进度条与原运行入口保持一致。续跑完成或失败均生成 review-attempt-002.zip，继续回传该包审计。

**代码阅读顺序**

| 文件相对路径 | 大致作用 |
|---|---|
| scripts/grpo_vllm.py | 原生采样、prefill 探针与同请求重复解码 |
| scripts/grpo_worker.py | 采样组提交、请求 seed 绑定与独立探针校验 |
| hotfixes/grpo-probe-replay-v1/apply.py | 保护两支检查点和预算，安装源码修订 |
| tests/test_grpo_probe_replay_v2.py | 实际失败夹具、概率一致性和篡改拒绝 |
| tests/test_grpo_probe_hotfix_v2.py | 安装事务、云端状态哈希和中断恢复 |

原始回包 SHA256：c27dedfd55790620807cee463d84a1992142f072fca500f5ab143f29be8789de

原运行发布 ZIP SHA256：b7110c17f1ed85629d3c80d240a961d1a8e784315579b5a08a37e23dba782600
