# Phase 4：云端单序列完整前向概率重算（2026-10-20）

> 云端执行更新：用户已在 L40S 完成重算，终端摘要显示 single_sequence_probability_check_passed=true。两后端与完整前向重算的最大逐 token 误差均低于本步阈值 0.001。以下“待执行”保留为交付时状态，实际结果见文末；训练准入仍为 false。

## 此前命令分别验证了什么

| 命令/步骤 | 已验证的范围 |
| --- | --- |
| nvidia-smi | 实例分配了 GPU，系统驱动能识别型号、显存与进程 |
| python/conda --version、df -h | Python、环境工具版本和磁盘挂载，帮助决定安装位置 |
| pip check、import 各库 | 包声明的依赖无冲突、Python 模块可导入；不等于模型可运行 |
| CUDA 张量运算 | PyTorch 能在该 GPU 上实际计算 |
| verify_cloud_bundle.py | 原迁移文件哈希、冻结快照和基础模型文件一致 |
| cloud_model_smoke.py --backend hf | Transformers+PEFT 能加载固定 base+adapter，以 CUDA FP16 生成一条回答 |
| cloud_model_smoke.py --backend vllm | vLLM 原生 LoRA 接口能加载并生成，保存实际 token 和 raw policy logp |
| 取回 hf.json/vllm.json 后比较 | 单题输入 54 token、回答 28 token 一致，文本/评分复算一致；raw logp 有数值差异 |

以上已证明最小 GPU 链路能运行，但还没有完整前向重算、随机多回答 rollout、吞吐实测或 GRPO 更新。尤其“相同答案”不能代替概率对齐，后续训练使用的是实际 token 的概率。

## 本步问题与停止条件

不重新采样。取已生成的同一条 prompt_ids+response_ids，加载同一冻结 SFT，用 Transformers 完整 teacher-forced 前向，分别比较 HF 增量生成和 vLLM 生成时保存的 raw logp。两个输入的 ID 相同，因此只做一次模型前向，得到 28 个回答 token 的分数。

新入口 scripts/replay_cloud_smoke.py；配置 configs/cloud_replay_smoke.json 绑定两份原始输入 SHA256、模型精度 float16、log_softmax 精度 float32。仍用 L40S、torch 2.8.0+cu128、transformers 4.57.6、PEFT 0.17.1，并检查记录中的 vLLM 版本 0.11.0。使用 eager attention、use_cache=False、eval/inference_mode，所有模型参数冻结，无 optimizer、无 backward、无 generate。

本次新完整前向运行前，固定每个回答 token 的绝对 logp 误差上限为 0.001，两条比较均满足才通过“单序列检查”。此容差是在已观察 HF/vLLM 生成分数差异后，为后续尚未执行的重算选定的工程筛查标准；不是未见数据上预注册的泛化标准，也不回溯宣布旧比较通过。其含义是单 token 概率比约在 exp(±0.001) 范围，即约 ±0.1%；不代表整条序列的概率比误差同样只有 0.1%。不使用接近零 logp 的相对误差，也不直接套用 CPU float32 的 1e-4。

失败也保存报告并退出 2；出现模型身份/记录结构错误直接报错。不能为了通过而自动放宽阈值。若失败，先检查对齐、dtype/adapter、增量和完整前向的数值路径，必要时另建 FP32 诊断。本步骤即使通过也保持 training_admitted=false：只是一条高置信度贪心回答，不覆盖低概率 token、随机采样、多输入/长序列或新权重更新。

## 数据与数学路径

输入目录：artifacts/trajectories/cloud-model-smoke/，文件 hf.json、vllm.json，要求与已取回的两份字节哈希一致。脚本核验原迁移包、模型/adapter/生成源码哈希，重新核对 tokenizer 编码/解码、详细评分、EOS 和分数数量。

完整序列为 54+28=82 个 token。logits 第 t 位预测第 t+1 个 token；causal_token_log_probs 先 shift 一次，回答切片从 prompt_length-1=53 开始，得到 28 个回答分数，包含真实 EOS。始终使用原始 ID，绝不将解码文本重新 tokenize 作为评分序列。

输出 artifacts/evaluations/cloud-replay-smoke/replay.json，记录输入/源码哈希、配置、实际重算向量、逐 token 误差、最大/平均误差、序列总分差、单序列检查结论。原 hf.json/vllm.json 不修改。目标文件已存在则拒绝覆盖。计时只是本次前向加分数提取，不能称 rollout 吞吐。

## 云端执行

上传新增小增量包到 /root/rivermind-data，解压后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase4-cloud-replay-update-20261020.zip .
cd AgenticPostTrainingLab
python scripts/replay_cloud_smoke.py
```

执行后取回终端摘要和 replay.json；出现退出 2 时保留该文件用于诊断，不重新跑生成命令。检查本次定时关机设置，不默认延长运行或扩大实验。

## 代码阅读顺序

1. `configs/cloud_replay_smoke.json`：先看被绑定的两份文件哈希、精度和本步数值容差。
2. `scripts/replay_cloud_smoke.py` → `main` 前半：输入、版本、设备、模型/源码身份、tokenizer 与 EOS 的复核。
3. 同文件模型加载与前向段：冻结 PEFT、拼原生 ID、use_cache=False 与 inference_mode，理解为何不调用 generate。
4. 同文件 `response_logps` → `src/agentic_post_training/algorithms/losses.py` 的 `causal_token_log_probs`：看 causal shift 与 prompt_length-1 切片，包含 EOS。
5. `compare_logps` 与输出段：逐 token 误差、序列总分差、阈值和退出码，区分单序列通过与训练准入。
6. `tests/unit/test_cloud_replay.py`：重点看第一个回答/EOS 的预测位置、FP32 log_softmax，以及“总分相同但 token 分数错位”仍应失败的测试。

## 当前执行状态

本报告交付重算代码与方案，尚未在云端执行完整前向，不能声称数值校准通过。本地仅运行适合 CPU 的逻辑和回归测试；测试结果在完成后补记。

2026-10-20 本地实际验证：本步新增 13 项测试通过，全套 181 项通过，无 skip，1 条已有 tiny PEFT 配置提示。未在本地重跑历史采样、训练或正式模型前向；GPU 重算待用户执行。

## L40S 云端重算实际结果

证据来自用户提供的完整终端摘要，本地未重新运行 GPU。原迁移清单与历史基线检查通过，重算单序列检查通过。完整 replay.json 仍保存在云端，当前尚未取回逐 token 数组、实际配置和文件身份供独立产物重审，因此不声称已经完成整份 replay.json 的离线审计。

| 生成分数与完整前向重算比较 | HF 增量生成 | vLLM 生成 |
| --- | ---: | ---: |
| 最大逐 token 绝对误差 | 0.00030709803104400635 | 0.00016791652888059616 |
| 平均逐 token 绝对误差 | 0.000026271324031768017 | 0.000012900437582175073 |
| 生成时序列 logp 总和 | -0.03623845550964688 | -0.03662333364627557 |
| 重算序列 logp 总和 | -0.03674495342647788 | -0.03674495342647788 |
| 序列 logp 总和绝对差 | 0.0005064979168309947 | 0.0001216197802023089 |
| 固定 0.001 逐 token 误差门 | 通过 | 通过 |

两列重算总分相同是因为两后端生成了相同原生 ID，只重算一次并分别比较；不是两次独立前向重复实验。结果支持当前固定模型、精度和这条序列的生成分数与训练侧完整前向对齐；不证明所有输入或采样分布均已对齐。vLLM 在本条序列上的最大误差较小，不说明它在一般情况下更精确。

输出文件为 artifacts/evaluations/cloud-replay-smoke/replay.json。training_admitted=false 为预设范围标记，不是本次检查失败：当前只覆盖一题高置信度贪心回答，还未验证随机多回答、低概率 token、不同长度/EOS 边界及组内数据打包。下一实现子步是受限的 GPU 4 题×4 回答随机 rollout，保存生效引擎配置、原生 ID 与逐 token 分数，再做训练侧重算和既有 reward/零方差审计；仍属于 Phase 4，不开始 GRPO 更新，不修改 reward 来追求非零方差。

代码阅读顺序沿用上节：重点回看 response_logps 的 causal shift/切片与 compare_logps 的逐 token 误差门。本次仅记录实际结果，未改代码或重复执行已通过的测试。
