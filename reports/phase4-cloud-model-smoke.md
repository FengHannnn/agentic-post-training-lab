# Phase 4：云端固定 SFT 最小模型加载（2026-10-19）

> 2026-10-20 执行更新：用户已在克隆后的 L40S 实例完成 HF 和 vLLM 两后端最小生成。以下“尚待执行”描述保留为最初交付状态；实际结果和限制见文末，不代表概率校准已完成。

## 前置状态与本步范围

用户云终端截图已确认迁移包 245 个文件校验通过，74 个冻结 snapshot 和 8 个基础缓存文件校验通过，包内源码提交为 d14c133。GPU 依赖导入和 CUDA 张量运算已通过。本报告交付最小模型脚本，云端实际运行仍待完成，不声称已经加载或校准模型。

新增 scripts/cloud_model_smoke.py，使用两个独立进程分别执行 hf 与 vllm 后端，固定首个已有训练输入，每后端仅一次贪心生成、最多 64 个新 token；不训练、不改 reward、不扩大采样。不将此步骤称为 4×4 随机 rollout、完整概率校准或吞吐实验。

## 实现约束

两后端均先验证原迁移清单和历史基线，再从固定基础模型目录加载权重，使用冻结 adapter 和 checkpoint tokenizer；输入来自已保存 rollout 的首个训练目标，并核对配置中的首题 ID。按原生 prompt token 调用后端，不经文本往返编码。模型身份使用历史文件哈希；云端 dtype 为 float16，与历史 CPU float32 不同。

hf 后端使用 Transformers eager attention、冻结 PEFT adapter 和显式贪心 GenerationConfig；保存生成 logits 对实际 token 的 raw logp。vllm 后端使用原生 LoRARequest、max_lora_rank=8（容量上限，实际 adapter rank 仍为 4）、单序列、512 上下文、eager、显存比例 0.25；脚本请求禁用 prefix cache/chunked prefill，但实际日志显示 prefix cache 禁用、chunked prefill 被引擎启用，因此不能将请求参数等同于生效配置。显式 raw_logprobs，不启用结构约束解码或输出修复；禁用模型 generation 默认配置。依据 [vLLM 0.11.0 LoRA 说明](https://docs.vllm.ai/en/v0.11.0/features/lora.html) 和 [该版 raw_logprobs 定义](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/config/model.py)。

记录原始 ID、结束原因和 raw policy logp。greedy 的实际行为分布不是 raw policy softmax，因此本步骤不填造 behavior logp，也不当作训练轨迹。检查非空回答、概率长度/有限值、EOS 只能在末尾及 EOS/length 一致性；概率 teacher-forced replay 留到下一步，training_admitted=false。严格评分只作观察，不将题目答错视为加载失败，也不以单题答对宣称引擎概率正确。

输出 artifacts/trajectories/cloud-model-smoke/hf.json 或 vllm.json，已存在则拒绝覆盖。原迁移清单没有新脚本，运行结果另记录新脚本 SHA256；不改写原迁移清单。

## 操作步骤

新增脚本和本报告以小 ZIP 增量上传，不重新传 280MB 权重，不覆盖原迁移包中有哈希的文件。解压到 /root/rivermind-data 后，在项目目录分步执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data/AgenticPostTrainingLab
python scripts/cloud_model_smoke.py --backend hf
python scripts/cloud_model_smoke.py --backend vllm
```

先检查 hf 成功再运行 vllm；失败保留报错，不自动尝试新版本或回退到合并权重。注意平台定时关机时间；不因接近截止时间默认延长付费运行。

## 代码阅读顺序

1. `scripts/cloud_model_smoke.py` → `main` 前半：离线模式、原包核验、固定版本检查、base/adapter/tokenizer 路径与 prompt ID 构造。
2. 同文件 hf 分支：冻结 PEFT、CUDA 半精度、显式贪心生成、对实际 token 提取 raw logp。
3. 同文件 vllm 分支：LLM 参数、LoRARequest、原生 token 输入、引擎结束原因与逐 token logp 提取；注意 top_k=-1 是该引擎禁用值。
4. 同文件 `validate_response` 与输出段：结束边界和概率结构检查，文件身份与“尚未 replay”的状态。
5. `tests/unit/test_cloud_smoke_contract.py`：真实 EOS/长度结束，以及空回答、错位概率、非有限值、非法正 logp、EOS 后 token 和结束原因矛盾的拒绝。

本地 CPU 只能检查纯记录逻辑和语法，不验证云端模型/引擎 API 的实际运行。云端执行后须据真实输出补记结果，随后再做随机 rollout 与训练侧概率重算。

本地实际验证：上述 8 项记录边界测试通过。此次未重跑历史训练、生成或全套测试；云端 hf/vllm 最小运行尚待用户执行。

## 2026-10-20 L40S 实际执行结果

用户截图确认克隆后的 GPU 为 NVIDIA L40S，46068MiB 可见显存，驱动 580.119.02；torch 2.8.0+cu128 CUDA 运算通过，迁移包/基线核验通过。L40S 小时单价、累计实际消费及旧实例是否释放尚未获得确认。

HF 运行截图与 vLLM 完整文本日志均显示：回答为合法 calculator 请求，表达式 12*(47+20)，finish_reason=eos，response_tokens=28，strict_pass=true。两者可见文本相同、长度相同；尚未取回两份 JSON，不能据此声称 token ID 逐项一致或概率一致。

vLLM 日志时间为 17:34:51 至 17:35:41，识别 LlamaForCausalLM，V1 引擎、float16、Flash Attention、PunicaWrapperGPU，基础权重加载约 0.2621GiB，报告可用 KV 缓存内存 10.79GiB。这些启动日志不是生成吞吐基准，KV cache 容量估算也不是实测并发能力。max_lora_rank=8 为引擎容量，不能误称 adapter rank 变为 8。

日志提示：未安装 FlashInfer，top-p/top-k 回退到 PyTorch 实现，本次仍生成成功；eager 下未启用 CUDA graph，符合请求；torch_dtype/LoRA 词表或 tokenizer 的弃用提示未阻止运行，tokenizer 路径已显式指定为 checkpoint/tokenizer。程序退出出现未调用 destroy_process_group 的 NCCL 提示：不影响已写出的生成摘要，但进程资源清理仍待后续采集器完善；不能声称所有告警已消除，不为此重装依赖。

生效配置的偏差：脚本 enable_chunked_prefill=False，而调度器日志和引擎最终配置均报告 chunked_prefill_enabled=True。后续采集器应保存生效配置并核实引擎调整逻辑，不伪报已禁用或静默修改既有实验身份。

原始日志归档于 artifacts/evaluations/cloud-model-smoke-20261020/vllm-console.txt（来自用户粘贴文本，不是本地重跑）。HF/vllm JSON 当前在云端 artifacts/trajectories/cloud-model-smoke/ 下；待取回核对。下一步先比对两文件的身份/原生 ID/逐 token 分数，再实现完整前向概率重算；尚未进行随机 4×4 rollout、GRPO 或基线晋升。
