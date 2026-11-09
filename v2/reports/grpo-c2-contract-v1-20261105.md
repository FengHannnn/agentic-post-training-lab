# C2 GRPO 首版算法与实验边界

确认日期：2026-11-05。阶段：C2 本地实现与 GPU 预检准备。

## 目标与后续阶段

C2 完成结果奖励 GRPO 在线闭环；C3 从同一个 W 独立训练细粒度 GRPO，并增加一种“月度对账→验证结账生成结转记录→下一账期核对”的真实依赖任务；C4 完成新任务确认和交付。至少一条 GRPO 分支最终超过 W。完整任务成功率饱和时，以完整恢复成功率作为主要判断依据。提升幅度、回归容差与最终评测规模暂存为待定项，最终确认前冻结。

允许调整学习率、K、采样、KL、奖励权重、更新安排和训练分布。加入 SFT、DPO、辅助 CE 的配方另记为混合方法。细粒度主对照固定共同起点、数据起点分布、终局奖励、优化和预算口径，主要比较逐动作信用。

## 模型与输入

- 基座：Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。
- 策略和 KL reference 共同起点：W，累计 SFT513 / DPO11。
- W adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。
- LoRA：q_proj/v_proj，r8、alpha16、dropout0，可训练参数 1,089,536。
- 新建 AdamW，weight_decay=0；RL 更新单独计数，保留 SFT/DPO 履历。
- 发布包附带已复核 W 的 adapter、complete、model_identity、历史训练配置及 audit。云端 prepare 核对完整基座文件哈希，并冻结当次 GPU 与软件身份。

## 分组与奖励

每组使用相同 train 任务、相同错误前缀、相同初始环境状态、相同采样 checkpoint；独立采样 K=4 条完整续跑。按组交替选择正常任务和恢复任务，比例 1:1；七种恢复场景循环调度。任务、前缀、环境状态和步成本共同构成 start_id。

每条轨迹的回报为 R_i = success_i - 0.01 × T_i。T_i 包括错误前缀及后续动作，使用原任务 12 步总预算。前缀只提供上下文，训练只对后续原生生成动作计损失。同组前缀成本相同。

mu = sum(R_i)/K；sigma = sqrt(sum((R_i-mu)^2)/K)；A_i = (R_i-mu)/(sigma+1e-8)。该轨迹所有动作共享 A_i。完全相同回报的组跳过更新，记录全部采样及零方差统计。全成功但长度不同的组可产生有效优势。

每个新组最多执行一次优化器更新。更新成功后保存 checkpoint，启动新推理进程重新采样；零方差或 KL/clip 门槛跳过均不增加 RL/Adam 更新计数。

## 原生采样与 token 准入

首版训练采样 temperature=1、top_p=1、top_k=-1、min_p=0；重复惩罚1，presence/frequency惩罚0，min_tokens=0。最大响应256 token、推理上下文8192。训练采样和未来评测采样分别配置。

q 使用 vLLM processed_logprobs，直接绑定实际生成的 prompt/response token。vLLM 0.11 将 processed 值定义为应用采样处理后的值：[官方参数文档](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html#logprobs-mode)。每次采样进程对首个动作最多16个 token 做同权重 teacher-forced 概率探针，记录 BF16 数值差异；初始最大误差门槛0.25。

EOS 停止保留真实 EOS；长度停止保留真实256个 token，执行所得动作并记录环境反馈。非法 JSON 是环境动作错误。上下文溢出、服务异常和基础设施中断保存 partial evidence，整组退出训练准入。原生 token、解码、EOS、长度、模式和种子逐项复核。

## 更新目标

p_old 在采样 checkpoint 的 HF FP32 路径重算并 detach；p_new 为同路径可求梯度分数；p_ref 来自冻结 W adapter。共用冻结基座并维护 policy/reference 两套 adapter，每次切换后显式设置可训练参数。old/ref/q 缓存绑定整组、采样权重和原生 token。

ratio = exp(log p_new - log p_old)

TIS = min(2, exp(log p_old - log q))，detach。

k3 = exp(log p_ref - log p_new) - 1 - (log p_ref - log p_new)。

每 token 损失为 TIS × [-min(ratio×A, clip(ratio,0.8,1.2)×A) + 0.02×k3]。依次按动作内 token、轨迹内动作、轨迹间平均。梯度裁剪1。sampled k3>0.05 或 clip_fraction>0.5 时清梯度并跳过更新。k3/clip 统计采用相同的轨迹均衡权重；同时保留 TIS、概率差异、动作和 token 计数。

学习率初值1e-6。训练前验证同 HF 路径 old/fresh 最大差≤1e-5。response_scores 只物化响应后缀词表 logits，并以完整 logits 测试其 causal 对齐。reference 张量哈希在更新前后核对。

## 刷新、恢复与审计

每个采样阶段独立进程、单 LoRA、固定 ID1/name、关闭 prefix cache，读取不可变 checkpoint。vLLM worker 对实际登记且激活的 LoRA 与从目标 checkpoint 新读取的对应张量逐项核对，保存112个 A/B张量的形状、dtype和哈希。

更新后保存 adapter、Adam、Python/Torch/CUDA RNG、group_cursor、策略版本、采样版本、实际更新数和下一组调度。checkpoint 目录提交与 transition 日志构成事务边界；恢复按已提交 transition 重建状态，保留中断采样并从原组起点重采。每次恢复共享原作业累计时限。prepare 成功且输入身份完整的运行支持 --resume。

云端审计重放动作、观察、证据、奖励、终局与 artifact；复算优势、缓存身份、实际 LoRA 变化、Adam计数及刷新后采样绑定。state.pt 留在云端，review ZIP 返回其 SHA、云端核验记录、adapter、轨迹、配置、冻结源码、日志和统计。本地按返回冻结源码复核实际拿到的文件。

## 预检额度与终端体验

首个 GPU 预检上限3600秒，预留120秒收尾打包；最多4个训练组、2次实际更新，至少1次实际更新及最后保存版本的再次采样。最后的刷新采样单列为 probe 组，并计入全部生成成本。

C2 首批正式作业最多3个、合计12 GPU小时，单作业最多4小时；正式组数、更新数、评测节点及停止规则由预检数据冻结。C3预算随后确定。

终端显示总阶段、逐组 rollout、逐动作评分/反向和严格重放进度条；子进程输出实时转发至终端并写入日志。失败、中断与超时均生成 review ZIP，保留状态与已产生证据。
