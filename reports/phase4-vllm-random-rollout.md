# Phase 4：GPU 4×4 随机 rollout 与概率、奖励审计（2026-10-20）

## 本步范围与执行状态

用户读完单序列完整前向重算代码后，要求生成随机 rollout。本步使用已有 L40S 和固定 SFT v2，在原来的 4 个训练输入上各生成 4 条随机回答，并逐条完整前向重算、分组打包及应用已冻结的 arithmetic-strict-v1 reward。无优化器、无参数反向更新；仍为 Phase 4，不是 GRPO trainer。

当前交付的是本地实现与云端执行包。只有收到云端运行结果后才能声称生成了这批新回答。本地测试不调用 vLLM、不下载模型，也不能替代 GPU 执行。

## 固定采样与概率契约

- 沿用 configs/rollout_cpu_smoke.json 的 4 个训练 ID，通过偏好数据 manifest 校验并只从 train 取题；前三题共享数字 family，是接口诊断，不是新开发评测。
- 每题 4 条，每请求 n=1。种子为 42+group_index*1000+sample_index，共 16 个独立请求种子。
- temperature=1、top_p=1、top_k=-1（vLLM 的禁用值），min_tokens=0、repetition_penalty=1、presence/frequency_penalty=0。无 stop 字符串、无 JSON 约束、无预测修复；使用 tokenizer EOS 作为正常停止 token，最多 64 个新 token。
- 显式 generation_config=vllm，避免继承模型的隐含采样默认参数。关闭 prefix cache，开启 chunked prefill，与此前实际 V1 引擎行为一致。eager、float16、最大上下文 512、最大并发序列 4、每步 token 预算 512、显存比例 0.25、max_lora_rank=8（实际冻结 adapter 仍 rank 4）。记录引擎生效配置，不只保存请求值。
- 引擎返回 raw_logprobs；以上采样契约不改变分布，因此 behavior=policy。两字段均保存，但 behavior 值是在这个受限契约下由 raw policy 分数复制，不是独立提取的 processed 分数。新增任何温度/过滤/惩罚都必须拒绝使用当前适配器，另行定义分布契约。
- 同引擎额外复现首条一次，共 17 条生成，只有原 16 条进入 rollout。复现不要求与旧 CPU 或不同 batching 配置逐 token 相同。

原生多请求接口和生效配置读取依据 [vLLM 0.11.0 LLM](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/llm.py) 与 [该版 LLMEngine](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/engine/llm_engine.py)。本次适配器使用原始 prompt token 输入，原始 response token 输出，不先 decode 再 encode。

## 三段执行流程

`scripts/collect_vllm_rollout.py` 默认 stage=run，用独立子进程顺序执行 generate → replay → audit。

1. **generate**：验证原包、模型/数据和版本，检查单序列 replay 前置结果并从保存分数复算其误差结论；建立新的 run 目录与源码快照。调用 vLLM 批量采样 16 条，保存每条原生 ID、policy/behavior logp、种子、EOS/length、文本、评分。重复和错误回答都保留。再复现首条，显式关闭 engine_core，清理当前进程已初始化的 distributed process group。
2. **replay**：在采样进程结束后，用新 HF/PEFT 进程加载同一冻结 SFT，单条拼接 prompt+response，use_cache=False，逐条计算回答 raw logp。FP16 模型、FP32 log_softmax，逐 token 最大绝对误差限 0.001，沿用单序列筛查标准。评分模型无梯度，核对参数 version counter 未变；这不是 GPU 内存所有权重的前后密码学哈希证明。vLLM 阶段通过固定只读推理接口执行，未额外声称逐参数 digest 已核对。
3. **audit**：不加载模型权重，重新检查模型/数据/源码身份、token 解码、编号和种子、分组一致性、EOS/截断、mask/张量、规则评分、reward/advantage、概率误差计算与文件哈希。完整模型前向发生在 replay 阶段，无模型审计只复核已保存的重算结果。

采样进程与评分进程不同时常驻 GPU，避免 KV cache 和 HF 模型抢显存。清理调用不等于已证明平台停止计费；平台实例仍需按使用安排关机。

## 验收、失败与停止规则

16 条记录必须齐全。EOS 属于有效回答，mask 按实际长度建立，pad 位置不参与训练目标。错误 JSON、算术错误、重复回答及长度截断不丢弃，截断按已冻结 reward 得 0。问候组只作诊断，不进入算术奖励训练范围。

全部 16 条概率重算、首条复现与冻结评分检查通过，才报告 rollout_interface_checks_passed=true；training_admitted 始终 false，因为尚未实现 old/new/reference、ratio/KL 和真实 GRPO 更新契约。即使 reward 全部零方差，也可以完成接口验收，但 has_reward_learning_signal=false，不能据此声称有效 RL 更新。

FP16 随机采样包含低概率 token，误差可能高于之前高置信度贪心样本。数值或复现门失败时，保存原始 rollout、重算、reward 和 summary，并退出 2；不自动提高阈值、不重采样、不换题或改奖励。结构/身份错误直接报错，已落盘的原始记录保留。只要原记录完整且采样阶段已完成，可在诊断后单独执行 replay；不能覆盖已存在的 replay 产物。

默认输出目录已存在时拒绝重新生成，避免误覆盖或无意追加费用。默认 scope 固定 4×4，首轮只运行一次，沿用累计总预算 200 元及本次平台定时关机规则。L40S 当前报价和账单仍待用户确认，不推算实际累计费用。

## 产物

默认目录 artifacts/trajectories/rollout-vllm-phase4-20261020-01/：

| 文件 | 内容 |
| --- | --- |
| run_manifest.json、source/ | 配置、模型/adapter/数据/源码身份及源码快照 |
| prerequisite_replay.json | 先前已通过的单序列 GPU 重算产物副本 |
| rollouts.jsonl | 16 条完整原始采样记录及规则评分 |
| generation_checks.json | 生效引擎配置、采样参数、额外复现探针、关闭调用记录 |
| replay_checks.json | 16 条完整前向分数、逐 token 误差、冻结检查 |
| response_batch.pt | [4,4,T] token、长度 mask、policy/behavior 分数 |
| reward_audit.json | 契约、逐条奖励、组内 advantage 与零方差统计 |
| summary.json | 每组多样性/成绩/reward、概率失败数、mask、耗时和文件身份 |

generation_call_seconds 是 engine.generate 的墙钟耗时，不是纯解码吞吐；replay_seconds 是评分前向与分数提取耗时。首条复现不计入生成 16 条的耗时，不用该小实验声称并发性能收益。

## 云端运行步骤

上传 phase4-vllm-rollout-update-20261020.zip 到 /root/rivermind-data。新增路径不会覆盖原迁移清单中的源文件，也不需要重新安装依赖或上传权重。

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase4-vllm-rollout-update-20261020.zip .
cd AgenticPostTrainingLab
python -u scripts/collect_vllm_rollout.py
```

成功时最后打印 artifact_audit_passed=true 和 rollout_interface_checks_passed=true。如果数值检查退出 2，最后 summary 会包含 replay_failed_samples 和 max_replay_abs_error；保留文件并提供终端摘要，不重跑采样。单独无模型重审：

```bash
python scripts/collect_vllm_rollout.py --stage audit
```

无模型审计可以确认一个数值未达标实验的产物一致性：artifact_audit_passed=true 与 rollout_interface_checks_passed=false 可以同时成立。

## 代码阅读顺序

1. `configs/rollout_vllm_smoke.json`：固定采样规模、种子、温度、精度、误差限和引擎资源。
2. `src/agentic_post_training/rollouts/vllm_records.py` → `validate_sampling`、`sample_from_output`：原生 token/分数转换、EOS/截断，以及何时允许 behavior=policy。
3. `scripts/collect_vllm_rollout.py` → `main`、`context`、`inventory`、`check_prior_replay`：分进程编排、只取训练输入、模型/数据/源码身份与前置校准结果。
4. 同文件 `generate`、`effective_config`：请求种子、16 请求、结果落盘、复现探针和引擎关闭。
5. 同文件 `replay` → 已读的 `replay_cloud_smoke.response_logps/compare_logps`：原生完整前向、逐样本误差门和失败保存。
6. 同文件 `load_and_validate`、`summarize`、`audit`：解码和 mask、reward/advantage、文件重审，区分记录一致、数值通过与奖励信号。
7. `tests/unit/test_vllm_rollout.py`：优先看 EOS=PAD、分布修改拒绝，以及 mask/奖励/编号/源码损坏拒收；再看“数值失败仍能完成产物一致性审计”。测试使用合成记录，不依赖被 Git 忽略的历史实验目录。

## 本地验证

测试覆盖原生结果数量/身份/概率对齐、非法分数、EOS与截断、采样契约变更、固定组编号/种子、behavior、mask、reward、重算映射、前置文件/源码/summary 篡改及模型无关完整产物重审。测试和实际云端运行结果在完成后补记。

2026-10-20 本地实际结果：新增 30 项测试通过，全套 211 项通过，无 skip，1 条已有 tiny PEFT 配置提示。真实 vLLM 初始化、GPU 采样及重算仍待用户执行，不能将上述测试数量当作本批 16 条采样已完成的证据。
