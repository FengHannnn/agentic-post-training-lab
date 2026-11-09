# E2：均匀权重 DPO 对照执行说明

2026-11-04，Asia/Shanghai。运行 ID：dpo-uniform-control-v1-20261104。

E1 已复核：86 步随机完整任务 13/16、greedy 完整恢复 21/28，258 步为 0/16、6/28。当前继续隔离困难权重的影响；C1 未通过，GRPO 暂缓。

## 唯一训练变量

原加权 DPO 的四个困难场景原始权重为 4，其余为 1，全数据均值归一到 1。E2 将 hard_weight 改为 1，169 个样本归一化权重均为 1。困难 84 对占总权重由 79.81% 降为 49.70%。intervention.json 自动核对训练字段，仅接受 hard_weight：4 → 1；任何父模型、学习率、种子、轮数、beta 或评测协议变化都会停止。

| 固定项 | 数值 |
|---|---|
| 父模型 / frozen reference | chosen-only-sft-v1-20261102，078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55 |
| 偏好数据 | 已审阅 169 对，SHA256 c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329 |
| 训练量 | 6 epochs，43 更新/遍，共 258 更新，1,014 对曝光 |
| 优化器 | fresh AdamW，LR 5e-6，weight_decay=0，max_grad_norm=1 |
| DPO 目标 | beta=0.1，response sequence-sum logp，batch mean |
| 批次 / 种子 | 每次更新 4 对（尾批 1 对），seed=2026110301，每遍 seed+epoch |
| 起点履历 → 终点履历 | SFT255/DPO11 → SFT255/DPO269 |
| 检查点 | 43/86/129/172/215 步和 258 步 final；固定 final 评测 |
| 整体时限 | 14,400 秒；实时输出；失败/超时也归档 |

实现继续使用 optimize_weighted_dpo，weights 全 1。算法记录沿用 hard_recovery_weighted_dpo，training_stage=uniform-control-6ep，experiment=E2_uniform_weight_control；报告明确显示均匀权重。真实小型 LoRA 更新用于验证其与原 unweighted optimize_epochs 的参数、Adam 状态、更新和曝光完全一致。

## 同轮四模型评测

parent、weighted_sft、weighted_dpo（原加权末步）、dpo（本次均匀末步）。三个控制 adapter 与完成元数据绑定到已复核 E1，实际云端文件和基础模型缓存由 prepare 验证。对照只做推理。

每模型完整任务 greedy 4 条 + sample 16 条；完整恢复 greedy 28 条 + sample 84 条。合计 80 条完整任务、448 条完整恢复，共 528 条。固定 4 个 development 任务、7 个恢复场景、原种子与温度，恢复前缀仍计入原 12 轮预算。test 未使用。

四模型各重算相同 169 对逐 token 分数和同一 238 个专家探针 CE。先验证本次父 reference 与原加权 DPO reference 一致，门槛 1e-4；原始逐 token 分数和模型 SHA 随包保存。评分汇总中的 optimizer_updates=0 专指推理评分阶段，整个作业计划训练 258 更新。

主比较是 uniform 对 weighted_dpo 的完整任务及完整恢复终局成功，按任务/rollout 或 context/sample 配对列改善、退化、共同成功、共同失败。同时检查 stale chosen CE、calculate_change/check_result/submit CE、有效首动作到最终成功的转换、重复查询和提交次数。E2 复核后按既定顺序准备 E3 chosen CE 保真与 E4 完整续跑覆盖，各实验独立冻结。

## 日志与资源采样

入口依次运行 prepare → train → scores → complete → recovery → audit。原始数据和模型保留；已有同名输出会拒绝重复运行。

resource_snapshots.jsonl 在阶段前后只读记录 nvidia-smi、free、df、ps，带 UTC 时间与阶段。各命令超时 2 秒，缺失命令或查询失败写入记录后继续。采样只用于定位 Jupyter 卡顿，不调整线程数、显存比例、内核或训练/评测参数。Adam/RNG 状态与中间 adapter 留在云端，review ZIP 回传最终 adapter、全部完成记录、输入绑定、源码、逐 token 评分、轨迹、日志和 audit。

## 云端执行

上传 dpo-uniform-control-v1-20261104-run.zip 到云端 v2 目录后，在终端运行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-uniform-control-v1-20261104-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_uniform.py
~~~

预期回传：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-uniform-control-v1-20261104-review.zip。

前置产物是原 chosen-only、加权 SFT、加权 DPO、169 对数据采集，以及刚完成的 E1 artifacts 目录；prepare 在任何训练更新前逐项核对。

## 本地验证

真实输入预检已通过：三个控制权重及完成记录、30 份溯源文件、169 个全 1 权重、与原训练相同的逐遍样本顺序、28 个开发恢复上下文、238 个探针（最长 1,527 tokens）。完整基础模型权重在云端，由云端 prepare 核对实际文件；本地绑定已审阅模型身份和真实 tokenizer。

全套 148 项测试通过（330.08 秒），源码编译通过，预检通过。测试明细见随包 handoff/dpo-uniform-local-validation-20261104.json；运行包 SHA256 见单独交付的 dpo-uniform-release-receipt-20261104.json。本地完成实现与验证；GPU 训练和本次新结果等待云端运行。
