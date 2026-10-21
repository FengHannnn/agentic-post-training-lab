# Phase 5 第一步：144×8 随机 rollout 与训练信号准备

日期：2026-10-21。Phase 4 已完成工程验收，数值限制仍保留。本轮进入 Phase 5 的数据与学习信号准备，不执行真实 GRPO 更新。实现和本地预检已完成，真实新采样待云端执行。

## 目标与冻结范围

此前 48×4 实验中，19 组全对、19 组全错、10 组混合。本轮不根据这些结果挑题，直接使用 dpo_v2 全部 144 个算术训练输入，每题 8 个回答，共 1152 条。覆盖同一 12 个数学 family 的全部问法/结构组合；natural、sequence、symbolic 各 48 题，四种 category 各 36 题。这是训练分布诊断，不是 144 个新数学 family 或盲测，不使用 development 回答参与采样。

固定 SFT 基线、tokenizer、严格 arithmetic-strict-v1 reward 保持不变。所有请求保留，无去重、按分筛选或失败重试。每条回答的原始 token、sampler raw logq、EOS/length 和文本均保存。

同一新批次分别统计前 4 条回答与全部 8 条回答的混合组数，并记录多出的混合组数。该比较复用同八条中的前四条，不额外生成，也不假称它们与旧 48×4 的回答逐条配对。增加组内回答数可能揭示偶发成功/失败，但不保证此前全对/全错的输入都会出现差异。

## 采样与评分契约

新契约 phase5-signal-144x8-v1。输入 ID 预先按字典序固定；seed=20261021+group_index*1000+sample_index，避免复用旧批次种子。逻辑组大小为 8，并发上限仍为 4，二者不是一回事。

vLLM 仍使用固定版本、FP16、原生 LoRA、eager、无 prefix cache、chunked prefill、max_model_len=512、max_num_batched_tokens=512、显存比例 0.25。T=1、top_p=1、无 top_k 过滤/惩罚、每条最大 64 token；请求 n=1，八条回答由八个显式 seed 请求组成。

为了保留中间进度，分 18 次 engine.generate 调用，每次 64 个请求；总共 1152 个请求，没有额外 benchmark。分块调度不同于 Phase 4 整批调用，原始 q 按实际执行保存，不宣称跨调度 raw logp 一致。EOS 原样保留，length 依据原 reward 契约为截断，不补 EOS、不删除。

生成进程退出释放 vLLM 后，独立进程加载原始 FP32 SFT+LoRA、eager、禁用 cache/TF32，对每条原始序列计算 old/no-grad 与 new/有梯度图分数；无更新时 logp 最大差不超过 1e-5。只允许 230400 个 LoRA 参数求梯度，检查全部参数前后摘要与原始身份一致，所有 parameter.grad 为空。不执行模型 backward 或 optimizer.step。

继续分别保存 sampler 与 trainer 分数，计算固定 token TIS 策略损失诊断。它是带上限的条件 token 有偏近似，不表示旧跨引擎门通过，也未实现组级/序列级无偏校正。reference/KL、真实反向、保存恢复和行为对照仍是下一实现步骤。training_admitted=false。

## 输出与判读

默认目录 artifacts/trajectories/phase5-signal-20261021-01：

- run_manifest.json 与 source/：模型/数据/新配置身份，以及依赖源码的哈希与快照。
- rollouts.jsonl：全部 1152 个原生回答及严格评分。
- generation_checks.json：实际引擎、采样设置、18 次分块与时间/token 数。
- trainer_scores.jsonl、trainer_checks.json、trainer_score_batch.pt：训练侧分数、逐条 ratio、参数不变、长度 mask 与独立 sampler 分数。
- reward_audit.json：全部组的 reward 与按 8 条计算的 advantage。
- summary.json：有效组数、全对/全错/混合、前四条与八条对比、分问法统计、严格/语义通过、JSON/长度、概率差异、校正权重与诊断 loss。
- cost_and_run.json：阶段退出状态、失败原因、作业时间与费用估算。

所有组均保留。某些组仍为零方差不算接口失败，不会追加采样以追求好看的信号比例。即使有非零 loss，也没有发生模型训练。未来训练须按策略版本组织 rollout→update 并刷新数据，不能把本批固定 SFT 回答无限作为最新策略的 on-policy 数据复用。

## 云端一次执行

上传 phase5-signal-update-20261021.zip 到 /root/rivermind-data，使用当前已有完整产物和环境的 L20（若换卡，新批次生成/评分必须在同型号上完成）：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase5-signal-update-20261021.zip .
cd AgenticPostTrainingLab
python scripts/collect_phase5_signal.py --hourly-price 2.18
```

时价若已变化，替换为实际数字。依照用户后续“预算不是问题”的说明，不再要求累计已花金额，也不再执行旧 200 元累计限制。本轮仍限制约 30 分钟、按作业耗时估算上限 10 元，清理预留 10 秒；不包含开机/空闲/存储/计费取整，不能当作账户级费用控制。失败或超时终止进程组，保留产物，尽可能打包，不自动重试。

每保存 64 条打印采样进度，每评分 12 组打印评分进度。输出目录或 ZIP 已存在则拒绝重用，不需要重新安装依赖或重新上传基线。

最终回传包位于项目根目录，便于 JupyterLab 查找：

```text
/root/rivermind-data/AgenticPostTrainingLab/phase5-rollout-review-20261021.zip
```

下载后关机，上传 ZIP 复核。若失败，先提供现有包或终端错误，不整批重跑。

## 本地验证

新增 20 项测试通过，全套 python -m pytest tests -q 为 289 passed，无 skip，1 条已有 tiny PEFT 提示。测试包括真实训练 split 的完整选择与类别平衡、1152 个 seed 唯一、8 回答保留重复/EOS、length 奖励、原生记录异常拒绝、同批前4/全8对比、预算计算，以及合成 1152 条产物完整审计和 mask 篡改拒收。

真实冻结基线的 74 份快照与 8 份 cache 校验通过，真实 tokenizer 编码全部 144 题，最长 prompt 为 64，带回答预算共 128，低于上下文 512。预检未加载模型、未生成真实新回答；不伪称本地通过 Linux cloud bundle 路径清单。记录位于 artifacts/evaluations/phase5-signal-preflight/local_preflight.json。入口语法编译通过。

## 代码阅读顺序

1. configs/phase5_rollout_signal.json：全部 144 个输入、组大小 8、新 seed、分块/时间上限和上一阶段身份。
2. src/agentic_post_training/rollouts/phase5_signal.py → select_cases/sample_from_output/validate_records：训练 split、原生 token/q、解码、组内 8 条及 seed/奖励重审。
3. scripts/collect_phase5_signal.py → generate：固定采样、18 次分块、保存与进度；score：原始 FP32 同路径 old/new，不反向。
4. phase5_signal.py → signal_summary：全对/全错/混合和前4/全8的意义；collect_phase5_signal.py → audit：张量、reward、分数和摘要的完整性。
5. 同入口 run/main：分进程、失败/超时、预算放宽后的单作业边界及项目根目录 ZIP。
6. tests/unit/test_phase5_signal.py 与 test_phase5_signal_audit.py：先看选择全部144题、前4/全8，再看合成1152条产物与拒收篡改。

下一步先取回该批次并分析有效组与错误类型，再冻结有限的真实 GRPO 更新/监控/评测方案，不直接长时间训练或声称有效提升。
