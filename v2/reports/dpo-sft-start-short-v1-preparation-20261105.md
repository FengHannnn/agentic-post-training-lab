# 从最佳加权 SFT 出发的保守短程 DPO

日期：2026-11-05（Asia/Shanghai）。运行 ID：dpo-sft-start-short-v1-20261105。

## 目标与本轮范围

上轮 DPO86 与加权 SFT 的同轮直接对照已复核：采样完整任务 12/16 对 16/16，采样恢复 51/84 对 60/84；贪心完整任务均 4/4、恢复均 21/28。本轮按用户确认进入一次保守短程 DPO，争取达到“比当前加权 SFT 更好或差不多”的阶段目标。

一条训练路径，从最佳加权 SFT 开始；同一模型同时作为 frozen reference。此前 E1–E4 机制排查保留在独立扩展记录，后续按用户要求纳入交接文档。

## 固定训练设计

| 项目 | 本轮固定值 |
|---|---|
| 策略起点与 reference | hard-recovery-sft-weighted-v1-20261103/final |
| 起点 tensor SHA256 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| 起点累计履历 | SFT513 / DPO11 |
| 数据 | 已审 169 对 train 偏好数据，内容与顺序保留 |
| 数据 SHA256 | c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329 |
| 学习率 | 5e-7，为此前 5e-6 的十分之一 |
| Pair 权重 | 全部为 1，uniform DPO |
| DPO β | 0.1 |
| 梯度累积 | 每次更新 4 对，单对 forward/backward 后累积 |
| 优化器 | 新 AdamW，weight_decay=0；节点之间精确续接 |
| 梯度裁剪 | max_grad_norm=1 |
| 训练种子 | 2026110301，固定一次打乱 |
| 累计检查节点 | 4、12、24 步，逐节点决定是否继续 |
| 累计 pair 曝光 | 16、48、96 对，最多约 0.568 遍 |
| 最大新增更新 | 24 次；累计履历最多 SFT513 / DPO35 |
| 辅助 CE | 0，保留纯 DPO 目标 |

起点完整继承加权 SFT 已学能力。reference 在同一作业中用该起点的 FP32 eval 模式重新计算，在第一次优化器更新前固定；沿途复用并核验该缓存。历史 chosen-only 父模型的 reference 缓存不进入本轮训练。

损失沿用响应 token logp 的序列求和 DPO：每对 chosen/rejected 先分别求序列 logp，再减去 frozen reference 的相对差，进入 -log sigmoid；batch 对 pair loss 求均值。新训练只更新原结构的 1,089,536 个 LoRA 参数。

## 运行顺序与停止规则

1. 校验源码、原 SFT、基座模型、偏好数据及上一轮来源；重新评测同轮 SFT。
2. SFT 完整任务采样应为 16/16、贪心为 4/4；若发生变化，保留评测结果并在 0 次训练更新处停止。
3. 从起点训练至第 4 步，保存 adapter、Adam、Python/Torch/CUDA RNG；释放训练进程后单独评测和严格重放。
4. 达标即停止；未达标且未触发严重退化规则时，沿同一状态续至 12 步，再到 24 步。
5. 全部节点结束后仍未达标，报告实际差距；候选按采样完整任务、采样恢复、贪心完整任务、贪心恢复、较早步数依次排序。

固定的“差不多”实用验收规则，四项同时满足：

- 采样完整任务至少 15/16，且比同轮 SFT 至多少成功 1/16，即最多落后 6.25 个百分点。
- 采样完整恢复比同轮 SFT 至多少成功 4/84，即最多落后约 4.76 个百分点。
- 贪心完整任务成功数不下降。
- 贪心完整恢复成功数不下降。

首个通过轨迹审计、必要质量检查与上述性能规则的节点就是收尾候选。完整任务比同轮 SFT 少成功 4/16 或更多时，提前停止后续训练。非有限损失/梯度、输入或状态身份差异、轨迹审计失败均停止并归档。

这套容差用于当前固定 development 任务的阶段决策。每个节点使用相同 development 评测进行选择，test 保持未使用。后续最终确认可按项目总计划使用冻结的新任务/种子。

## 评测和审计

每模型完整任务：贪心 4 条、采样 16 条；完整恢复：贪心 28 条、采样 84 条，共 132 条轨迹。模型为同轮加权 SFT 加最多 3 个累计节点；通常首个达标节点停止，主轨迹数为 264–528 条。每模型另做 7 个固定上下文的 2 次贪心首动作质量检查。

沿用上轮种子与温度：完整任务采样与恢复采样温度均 0.8。主指标是采样完整任务成功率，恢复为辅助指标；同步报告逐任务/场景结果、配对改善/退化、轮数、token、工具/SQL/证据错误与首动作有效性。

每模型一个独立推理进程，单 LoRA、固定名称 fixed_adapter 和编号 1；vLLM BF16、eager、max_num_seqs=1、无 prefix cache。私有配置只排序 target_modules。所有原生 token、prompt、种子、EOS、预算、动作、观察、奖励、产物和日历 SQL 均严格重放；节点按审计后的成功数自动判定。

整个作业上限 14400 秒，实时输出和阶段日志，保留资源快照；同名结果保护。即使中断或失败也返回 ZIP，记录已完成的累计更新数。优化器/RNG 文件留在云端，各节点真实 adapter、状态哈希、Adam 步数证据、训练记录、完整评测和冻结源码进入返回包。

## 本地验证

**11 项本地检查通过，用时 57.75 秒。** 覆盖真实 LoRA adapter 保存重载、Adam/RNG 续训与连续训练逐项相同，参考缓存/冻结参数保持，部分遍历曝光与训练履历，性能容差和自动停止，失败归档，原生轨迹重放、阶段审计到最终候选选择，以及单 LoRA 质量检查。4 个阶段命令入口与源码编译检查通过。

本地核对真实 SFT tensor：112 个张量、1,089,536 个参数，SHA 与已审起点一致；全部 169 对数据通过来源校验与固定 tokenizer 编码，最长单侧上下文加响应为 1376 token；28 个 development 上下文与训练 prompt 无重叠。基座完整权重由云端 prepare 在训练前逐文件核验。

本地交付为已验证的代码、配置和运行包；GPU 训练与实际性能结果在云端执行后产生。

## 云端执行

将 dpo-sft-start-short-v1-20261105-run.zip 上传到云端 v2 目录后执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-sft-start-short-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_sft_start.py
~~~

使用上面的绝对 Python 路径即可，不依赖终端是否显示 base 或已激活虚拟环境。

运行后下载并回传：

/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-sft-start-short-v1-20261105-review.zip

本轮依赖云端保留原加权 SFT、169 对已审偏好数据、上一轮 early86 对照产物及固定 Qwen2.5-1.5B 基座缓存。prepare 会在任何训练前检查这些来源。
