# C2 outcome GRPO 首轮正式实验契约

冻结日期：2026-11-06。运行标识：grpo-outcome-formal-v1-20261105。

## 起点与目标

首轮正式实验从已复核 W 独立开始，W 的累计谱系为 SFT 513 / DPO 11 / RL 0；新建 AdamW 状态。W 同时作为冻结 reference。起始 adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。预检 RL 2 的 adapter 保留为预检产物。

C2/C3/C4 为必做。最终至少一条 GRPO 分支超过 W；完整成功率饱和时以恢复成功率为主。数值提升与回归阈值留待后续确定。本轮记录开发集上的完整配对结果与候选排序，C4 执行新任务最终确认。调参已获授权，混合 SFT/DPO/CE 方法另行登记；本包为纯 outcome GRPO。

## 训练

24 个训练组，normal/recovery 各 12 组，每组 K=4，共 96 条训练轨迹；最后另采样 4 条最新策略轨迹验证刷新。至多 24 次更新，至少 1 次真实更新通过运行审计。零方差组直接跳过，不做拒绝重采样；组预算与实际更新数分别记录。

同组使用相同任务和起始状态，normal 与 train recovery prefix 按组 1:1。prefix 不进入训练目标，prefix 的动作计入原始 max_turns=12。回报 R=success−0.01×总动作数，组内 population std 归一化，epsilon=1e-8；同一 episode 各动作共享 advantage。目标按 token 均值→action 均值→episode 均值归约。

采样 T=1、top_p=1、top_k=-1，最多 256 个响应 token，8192 上下文；真实 q 使用 vLLM processed_logprobs。HF FP32 重打分 old/ref；old/fresh gate=1e-5。AdamW LR=1e-6、weight_decay=0，clip=0.2、beta=0.02、TIS cap=2、grad clip=1。sampled k3>0.05 或 clip fraction>0.5 时跳过更新且不推进 Adam。每组至多更新一次，然后从不可变 checkpoint 启动新采样进程。每次加载验证实际 worker 的 112 个 BF16 LoRA 张量。

## 开发评测

W baseline 与第 8、24 个已提交训练组对应 checkpoint 各运行一次。每个节点 132 条 episode：4 greedy complete、16 sample complete、28 greedy recovery、84 sample recovery；另有 7 个上下文×2 次 greedy 单动作质量检查。共 396 条开发 episode 与 42 次质量动作。

评测沿用已定开发任务、seed 与采样协议，所有分支一模型一进程、一 adapter。样本逐条严格重放，恢复轨迹从冻结 prefix 完整续跑，calendar evidence 独立核验。输出按任务、场景、模式、分支汇总，以及相对 W 的 paired improved/regressed/both-success/both-failure。W sample complete 饱和时以 sample recovery 排序，否则以 sample complete 排序；排序作为开发选择记录，数值验收阈值仍为空。

## 资源、进度和恢复

这是最多 3 个正式作业中的第 1 个，总预算 12 GPU 小时；本作业含评测和审计的 wall-time 上限 14400 秒，预留 120 秒打包。仅接受 NVIDIA L40S 且显存至少 48,000,000,000 bytes；同一作业内保持 GPU/runtime 身份绑定。首次运行从 W/新 Adam 开始；中断恢复使用 --resume，累积已耗时间继续计费。

终端显示总阶段、采样动作/响应 token、评分、反传、完整评测与审计进度。异常或超时保存日志、已提交 checkpoint 与 review ZIP；state.pt 留在云端用于 Adam/RNG 恢复，review ZIP 返回其校验与提交证据。

v2 预检耗时 461.9235 秒，采样 16 episodes/118 actions/9410 response tokens；两次更新耗时 42.6818 秒。L40S 更新峰值 allocated=22,002,655,232 bytes（20.49 GiB）、reserved=50,266,636,288 bytes（46.82 GiB）。按预检采样开销外推，本轮训练与三次评测约需 3–4 小时；运行器以实际累计时间执行硬预算。

## 内存实现与验证

HF eval 模式下原 gradient_checkpointing 开关未触发重计算。正式包增加 decoder layer 的显式 nonreentrant activation checkpoint，模型保持 eval 模式，保持参考/策略评分路径一致；动作完成后可释放 CUDA cache。测试验证实际 checkpoint 调用、分数一致、全部 LoRA 梯度一致，包括 Qwen2 SDPA 且 A/B 梯度均非零的用例。云端指标记录峰值 allocated/reserved 和 checkpoint 层数，用实际运行衡量内存变化。

## 输入来源与本地验证

GPU profile 来源为已复核 v2 review ZIP，SHA256=6addd0ca11afb9477c7571ea01470ddd4cc755b63d21730ada65e45a2b03620f；audit/status/environment/local audit 四份文件随包冻结并校验 SHA256。实际 review 的 183 个冻结 source 文件、15 项输入绑定均匹配发布记录；16 条轨迹完整重放，4 次 worker LoRA 独立投影均验证 112 张量，两个更新的目标重算最大误差低于 2e-8。

128 项本地测试通过，覆盖既有环境/recovery/DPO 路径、GRPO 数值与事务、字符串 RPC、正式评测计划/配对/审计、防篡改、评测中断恢复，以及 activation checkpoint 实际执行与梯度一致性。正式 GPU 结果由返回的 review ZIP 继续复核。
