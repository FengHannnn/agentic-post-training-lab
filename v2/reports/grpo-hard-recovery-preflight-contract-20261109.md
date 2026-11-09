# Hard Recovery Preflight Preparation

日期：2026-11-09。运行：`grpo-hard-recovery-preflight-v1-20261109`。

本轮按用户确认的方向，优先排查困难训练起点为何缺少有效学习信号：固定最佳加权 SFT 模型 W，采样 96 条训练轨迹，离线按当前 local 规则重算信用。累计作业上限 **2 小时**，包含准备、采样、重放审计和打包；新增 RL、SFT、DPO 更新均为 0。

报告项目内路径：`reports/grpo-hard-recovery-preflight-contract-20261109.md`。上传解压后的云端位置：`/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-hard-recovery-preflight-contract-20261109.md`。

**要回答的问题**

上一轮 legacy 与 local 在三类困难训练场景中均为 0/32 成功，没有新增验证进展，相关 8 组全部跳过。本轮比较同一任务、同一类问题的“状态简化”与“原复合”起点，观察 W 能否取得新的有效证据、完成恢复，以及 local 能否产生非零信用。

状态简化条件的具体改动在采样前固定；其实际恢复难度由采样结果描述。原复合条件直接沿用原课程的上下文、动作和 observations。

**任务与采样计划**

| 训练任务 | 来源 | 初次查询故障 |
|---|---|---|
| train-000 | 原训练任务 | none |
| curriculum-train-000 | 扩展训练任务 | none |
| train-001 | 原训练任务 | transient_query |
| curriculum-train-001 | 扩展训练任务 | transient_query |

三个错误类别 × 两种条件 × 四个任务 × K=4 = **96 条完整继续执行轨迹**，对应 24 个信用组。每个“类别 × 条件”包含 16 条轨迹、4 个任务组。只使用训练任务，后续正式方法比较另设开发评测。

同一任务、错误类别与 rollout 在两条件中使用相同 seed；条件执行顺序按任务与类别交替平衡。采样 T=1、top_p=1、top_k=-1，无额外截断；单动作最多 256 tokens，上下文最多 8192 tokens，与原训练采样设置一致。

**两种起点的具体差异**

| 错误类别 | 原复合 compound | 状态简化 simplified |
|---|---|---|
| missing_alias_pair | 本月、上月查询均缺少指标别名，随后无效计算差额 | 上月查询保留正确别名，本月仍缺少别名；相同位置仍有无效差额计算 |
| schema_query_repeat | 连续两次错误 SQL，随后无效计算差额 | 用 inspect_schema 替换第二次错误 SQL，保留第一次错误和后续无效计算 |
| requery_stale_change | 补查本月后，最后检查同时使用错误差额数值和错误差额证据 | 只把最后检查的差额数值修正，差额证据仍错误指向本月查询 |

schema 简化条件同时减少一次重复 SQL 错误，并再次展示表结构；后续 query ID 也随查询次数改变。这些变化作为该条件的整体定义记录。requery 两条件都只验证了本月与上月指标，仍缺少有效 change 证据；最后的检查反馈完全相同，均只有 change_cents 不通过。

每对条件的前缀动作数完全相同，因此剩余步数相同。**12 步是包含脚本前缀的总预算。**无瞬时故障时，三个类别的前缀分别为 5、5、7 步，剩余分别为 7、7、5 步；带瞬时故障的任务，前缀各增加 1 步，剩余各减少 1 步。

24 个起点均已通过专家继续执行检查，最长总长度 11 步。缺别名条件中，状态简化需要补齐一个月的有效证据，原复合需要补齐两个月；其余两类的专家继续执行动作数在配对条件中相同。

**固定模型与记录指标**

W adapter SHA256：`f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f`。基座为 Qwen2.5-1.5B-Instruct，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。全程单进程单 LoRA 编号 1，模型权重保持不变。每次采样会话核对 worker 原生 LoRA 张量，并使用既有 same_forward_sampler_v2 观察首次动作的原生概率。

local 定义保持 `A[i,t] = A_outcome[i] + 0.2 × 本动作新验证指标数`。根据每个条件自己的 K=4 组计算；不同起点的轨迹分别分组。

记录：

- 最终成功数、至少取得一项新验证进展的轨迹数。
- 起点已有指标数、后续新增指标数，以及对起点尚缺指标的完成比例。
- local 有效组数、非零动作优势数、有局部加成的组数、终局回报有方差的组数。
- 工具错误、失败检查、重复失败动作、继续执行动作数、前缀与剩余预算。
- 相同任务/类别/rollout 的成功与进展配对改善、退步数。

例如缺别名简化条件已在前缀验证上月指标，这一指标不计入继续执行的新增进展；归一化完成比例使用“后续新增指标数 / 起点尚缺指标数”。

本轮结果用于诊断训练起点。结果优先按类别和四个任务分别查看：简化条件能产生进展而复合条件缺少进展时，后续考虑逐级课程；两条件均无进展时，重点查看首次有效 SQL 取证；有进展但未成功时，重点检查差额证据、失败检查修复和提交。后续训练方案根据回包共同确定。

**预算、进度与续跑**

累计上限 7200 秒，预留 180 秒打包。参考上一轮三类困难采样及全体采样耗时，96 条采样估计约 35–60 分钟，连同准备、审计和打包约 60–90 分钟。简化起点的实际生成长度由本轮记录。

总阶段进度条覆盖 prepare、collect、audit；采样显示 96 条 episode 的进度及当前类别、条件、动作数、响应 tokens；审计显示逐轨迹重放进度。每条完整且验证通过的 episode 原子提交，中断后按 episode 续跑，已提交轨迹保持原内容，累计预算沿用。

可在新 L40S 实例开始本作业；作业内和续跑时固定 GPU UUID、驱动与软件版本。沿用 vLLM 0.11.0、Transformers 4.57.6、PEFT 0.17.1。保留原基座缓存；完整运行包附带 W adapter、身份文件和全部新增入口。

**运行命令**

将 `grpo-hard-recovery-preflight-v1-20261109-run.zip` 上传到 `/root/rivermind-data/AgenticPostTrainingLab/`，执行：

```bash
cd /root/rivermind-data/AgenticPostTrainingLab
/root/rivermind-data/envs/apt-vllm/bin/python -m zipfile -e grpo-hard-recovery-preflight-v1-20261109-run.zip .
cd v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_hard_preflight.py --config configs/grpo-hard-recovery-preflight-v1.json
```

同实例中断后，在 v2 目录续跑：

```bash
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_hard_preflight.py --config configs/grpo-hard-recovery-preflight-v1.json --resume
```

首轮返回包为 `v2/grpo-hard-recovery-preflight-v1-20261109-review.zip`；续跑带 attempt-002 等后缀。完成或异常退出后，终端均输出 Download 路径及已提交轨迹数。

**本轮新增文件阅读顺序**

| 相对路径 | 大致作用 |
|---|---|
| configs/grpo-hard-recovery-preflight-v1.json | 冻结 W、96 条计划与 2 小时预算 |
| src/apt_v2/grpo_hard_preflight.py | 配对起点、专家证明、采样计划和信用统计 |
| scripts/grpo_hard_common.py | 配置、原生轨迹、权重与概率证据审计 |
| scripts/grpo_hard_worker.py | 准备、冻结 W 采样、逐轨迹提交与审计入口 |
| scripts/run_v2_grpo_hard_preflight.py | 阶段调度、累计计时、续跑和回包 |
| scripts/build_v2_grpo_hard_release.py | 构建并逐文件校验完整运行包 |
| tests/test_grpo_hard_preflight_v2.py | 起点、信用、原生重放、篡改拒绝与续跑验证 |

原有文件通过调用复用，本轮运行包中这些文件保持已发布版本。

**本地验证结果**

干净解压运行包后，233 项 GRPO 回归测试全部通过，耗时约 3 分 53 秒；其中本轮新增 13 项测试，覆盖24个配对起点、专家预算、模拟采样与逐轨迹续跑、全失败零进展信用、真实 W 哈希、概率证据和篡改拒绝。265 个绑定文件与三个新增脚本命令入口校验通过。与上一发布包共有的256个文件保持原字节。

真实 Qwen tokenizer 检查覆盖24个起点：最长起始提示1590 tokens、最长专家上下文1967 tokens、最长专家动作142 tokens，满足8192/256限制。本地验证使用CPU环境、模拟采样和历史原生证据；本轮W的实际恢复表现由以上L40S采样命令生成。
