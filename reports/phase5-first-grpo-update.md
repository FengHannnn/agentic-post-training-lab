# Phase 5：有限真实 GRPO 更新、reference、保存恢复与生成对照

日期：2026-10-27。用户已读完数据与目标函数代码并要求进入下一步。本轮实现首次真实模型更新入口，CPU 测试与输入预检已完成；135M 模型云端更新尚待执行，不提前宣称训练完成或行为提升。

## 固定实验范围

复用已验收的 phase5-signal-144x8-v1 数据，原始 summary SHA256=272dc316970eca29fd9bbcd0beadfa196dbfa31a7eb257f29764b53004a42d52。初始 policy、old 与 reference 都对应原始 FP32 SFT 参数身份；sampler q 仍是此前 vLLM 实际采样路径的保存分数。原始数据与历史 training_admitted=false、旧跨引擎门失败记录不回写。本次用新契约 grpo-first-update-v1，在额外身份、参考与梯度检查后进入有限更新。

从全部 144 组中，以预先冻结的顺序使用 62 个混合奖励组，每组八条全部保留。31 个全对与 51 个全错组不进入本轮优化，原始记录不删除。选择混合组是显式的有效组调度，改变了优化分布；不把它冒充在全144输入分布上的无偏训练，KL 也只在所选历史上计算。

每个 minibatch 最多4组，最后一批2组，共16个逻辑优化器步骤、496个候选回答，各组只访问一轮。无重复 epoch，不根据开发表现换参数，不自动再生成一批或继续长训。这是一个固定 old-policy rollout 上的单次有限策略迭代，不是已经实现多轮在线 rollout 刷新。

| 设置 | 固定值 |
| --- | ---: |
| 参数 | 原始 FP32 SFT，只有 q/v LoRA 可训练，230400 个 |
| 学习率 / 优化器 | 1e-5 / AdamW，weight_decay=0 |
| PPO clip epsilon | 0.2 |
| sampler token TIS cap | 2 |
| beta | 0.02 |
| 最大梯度范数 | 1 |
| 更新前 minibatch sampled k3 停止界 | >0.05 |
| 更新前 ratio 超出 [0.8,1.2] 的 token 比例停止界 | >0.5 |
| 最大作业时间 / 估算作业费用 | 约30分钟 / 10元 |

这些是执行前固定的首轮工程参数，不宣称最优。eval 模式下保留梯度图，所有 dropout=0；FP32、eager、use_cache=False 的评分路径和禁用 TF32 保持不变。eval 不等于关闭梯度。每个回答独立前向并反向累积，按整个 minibatch 的有效 token 数归一化，避免同时保留32条完整计算图。

## reference 与 KL 项

reference 是单独加载并完全冻结的 SFT 模型。更新前对496条候选回答重新计算 reference 与初始 policy 分数，要求二者与保存 old logp 的最大误差均不超过1e-5。评分后只缓存所生成 token 的 reference logp，不缓存整个词表。old 固定用于策略比值，reference 固定用于偏离惩罚，初始化相同不代表角色相同。

```text
w = stopgrad(min(2, p_old/q_sampler))
r = p_new/p_old
x = log p_reference - log p_new
k3 = exp(x) - x - 1
L = -mean_token[w * min(r*A, clip(r,0.8,1.2)*A)]
    + 0.02 * mean_token[w * k3]
```

代码以双精度 expm1(x)-x 改善接近零时的数值精度，保留 new 的梯度。形式参考 [TRL 官方 GRPO 文档](https://huggingface.co/docs/trl/v0.26.2/en/grpo_trainer) 中的 sampled KL 项；本项目没有安装或接入 TRL trainer。

重要限制：这些 token 来自固定 sampler 的历史，k3 在本轮作为样本惩罚使用，不称为对当前完整策略分布 KL 的无偏估计；token TIS 也不完整校正前缀/组联合分布。sampled_k3 停止门只是代理指标，不保证真实 KL 小于0.05；clip同样不是实际概率变化的硬边界。

## 真实更新与停止检查

每步先清梯度、逐回答累积策略/KL损失的反向，再检查基础参数没有梯度。若聚合 sampled k3 或 clip 比例超界，则清空已算梯度、停止本轮，不执行该步 optimizer.step。要求 LoRA 梯度有限且非零，clip_grad_norm 后执行更新。非有限分数、损失或梯度直接报错，保留已有输出，无自动调参/重试。

日志中的 static_rollout_reward_mean 来自原始固定回答，不是当前模型更新后的新 reward，不能用它证明训练期间 reward 提高。每步记录 loss、policy loss、sampled/weighted k3、ratio范围、clip比例、梯度范数、token数与组ID。

完整基础参数摘要只在训练前后比较，reference 摘要也前后比较；每步只检查梯度状态，不每步遍历全部参数元素比对。原始SFT、已有产物和开发数据均不覆盖。

## 保存与恢复验证

逻辑第1步后保存 adapter 和完整 Adam 状态、步数、参数顺序、RNG，并绑定原始输入摘要与训练配置的联合身份。先在不中断分支执行第2步并暂存结果；丢弃该分支，重新加载第1步 adapter+optimizer，再执行同样第2步。

要求新参数最大误差不超过1e-6，Adam动量/状态最大误差不超过1e-7，步数计数精确一致，损失差不超过1e-6。随后使用恢复分支继续逻辑第3—16步。

因此完整运行时候选轨迹16步，但实际 optimizer.step 调用17次，其中1次属于被丢弃的恢复探针；实际模型 backward 通常为528次，即496条候选回答加额外32条探针。输出分别计数，不能将17次都称为候选训练步。若提前停止，以实际日志为准。

最终再保存 adapter、optimizer 与 tokenizer，在全新基座上加载最终 adapter，比较8条回答 logp（1e-5）和8个开发输入的greedy token，并核验全参数摘要。这是本轮自动内部恢复与重载验证，不宣称已实现任意断点/分布式任务的通用恢复调度器。

## 行为对照与判定

更新前后在相同FP32 GPU、相同模板下各做一轮 greedy 生成：48个既有训练诊断输入、全部56个偏好开发输入（48算术+8问候），最大64 token，不修复输出。严格计数还要求正常EOS完成；问候只判断结构/路由和非空回答，不评价开放语义。

训练与开发 prompt 隔离。开发集已在历史实验中查看过，是已暴露开发诊断，不是最终盲测；训练诊断集不用于泛化结论。记录逐题严格/语义、EOS/length和结构退化，即使总分持平也拒绝掩盖单题JSON/schema/route回归。

engineering_checks_passed 检查真实更新、基础/reference冻结、恢复与最终重载；behavior_safety_gate_passed 单独检查无新结构退化且开发严格总数不降。两者不混为一谈。baseline_promoted 始终false，不自动将候选替换为冻结SFT，也不要求GRPO必须胜出才结束首轮实验。

## 云端执行

上传 grpo-first-update-20261027.zip 到 /root/rivermind-data，使用保存了1152条原始产物的L20实例。新文件增量加入，不覆盖旧云包哈希文件，不重新采样。

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e grpo-first-update-20261027.zip .
cd AgenticPostTrainingLab
python scripts/train_grpo_first.py --hourly-price 2.18
```

时价若变化则替换。已按用户预算放宽取消累计200元阻断，仍保留本次作业约30分钟/估算10元上限和清理时间。费用估算不含开机、空闲、存储或取整。运行中打印评测和优化进度，出错/超时停止整个子进程组，保留checkpoint和日志，尽可能打包；不自动重试，不自动关机。

输出目录 artifacts/checkpoints/grpo-first-20261027-01 包含源码快照、计划身份、前后预测、逐步history、第1步与最终checkpoint、resume/reload检查、summary、费用记录。项目根目录自动生成：

```text
/root/rivermind-data/AgenticPostTrainingLab/grpo-first-review-20261027.zip
```

下载后关机、上传ZIP复核。若失败，提供已有包/错误，不直接重复启动。云端真实训练结果仍待执行。

## 本地验证

新增16项测试，全套 python -m pytest tests -q 为305 passed，无skip，1条已有tiny PEFT提示。tiny Llama+真实PEFT在CPU完成非零梯度和实际参数更新、基础与reference冻结、Adam恢复后的下一步参数/动量一致、不同回答长度的microbatch梯度与全token均值等价、clip停止时不更新。张量测试覆盖beta=0回归、惩罚方向、有限差分梯度、非法输入；评测测试覆盖总分相同仍有结构回归、length不算完成及ID错配。

真实1152条输入预检已通过，确认62个候选组/496回答、16批、48+56评测输入均在长度预算内。预检没有加载135M模型或执行GPU依赖检查；仅替换GPU检查返回值以测试CPU输入通路，并核验冻结基线，未伪称本地重现Linux cloud bundle路径检查。记录归档于 artifacts/evaluations/grpo-first-preflight/local_preflight.json。入口语法编译通过。

## 代码阅读顺序

1. configs/grpo_first_update.json：固定62组顺序、16步、学习率/beta、停止界与评测输入。
2. src/agentic_post_training/trainers/grpo.py → completion_objective：策略项、sampled k3、两种比值和梯度；accumulate_minibatch：单回答反向与全minibatch token分母。
3. 同模块 guarded_step：反向、阈值检查、梯度裁剪与实际optimizer更新；save/restore_optimizer_checkpoint：恢复哪些状态与身份。
4. scripts/train_grpo_first.py → context/train：旧数据重审、独立reference、首次评分对齐、第1步保存与第2步恢复探针。
5. 同入口 evaluate/compare_evaluations、最终reload：同条件行为对照、逐题退化与工程/行为验收分开。
6. tests/integration/test_grpo_first_update.py：真实tiny LoRA更新、恢复动量、microbatch梯度与停止；再读两个unit文件的数学/行为边界测试。

首轮取回结果后才决定是否需要下一策略版本的新rollout与有限后续实验。本次不自动扩展到多轮训练、0.5B模型或完整工具Agent。
