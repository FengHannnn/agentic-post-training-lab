# Phase 4：智川云资源与接入方案（2026-10-19）

## 已确定的约束与当前状态

用户选择智川云，租机总预算不超过 200 元；本方案按人民币、包含准备/调试/运行及相关存储费用的累计总额控制。预算不是每次会话重新获得 200 元。当前尚未创建实例、充值或消费，控制台实时库存、小时单价和可选镜像尚未获取。

当前仍处于 Phase 4。CPU rollout 和 reward 离线审计完成，真实 GPU 接口、vLLM 校准和吞吐实验尚未执行。本步只完成资源方案与本地配置核查，无新增运行代码，无模型采样，无训练，无需重跑之前已通过的 160 项测试。

## 资源选择

建议比较单张 RTX 3090 24GB 和 RTX 4090 24GB 的按量报价，优先选择价格合理且镜像/驱动匹配的实例。此为依据当前 135M 模型与小规模采样作出的工程建议，不是已测显存结论；首轮不需要多卡或大显存高价实例。智川云官方选型文档将 3090/4090 列为推理候选，实时可用性仍以控制台为准。[官方 GPU 选型](https://docs.gpuhome.cc/docs/gpu/selection.html)

资源目标：单 NVIDIA GPU，Linux x86_64，Python 3.11 优先，建议至少 4 个 CPU 核与 16GB 系统内存。内存和 CPU 是准备环境及前向评分的预留建议，不是已测最低要求。先用默认磁盘容量，检查实际空闲空间后再决定是否扩容。不要仅根据 CUDA 名称选镜像；需同时记录宿主驱动、PyTorch/CUDA 和引擎版本。

## 费用和停止规则

建议分配如下，属于计划上限，不是保证会花费这些金额，也不承诺这些金额足以完成所有后续阶段。

| 用途 | 预算上限 |
| --- | ---: |
| 首轮环境、加载和最小校准 | 30 元 |
| Phase 4 批量 rollout、评分与测量 | 50 元 |
| 留待 Phase 5 最小训练闭环 | 100 元 |
| 存储、无卡准备与异常预留 | 20 元 |
| 合计 | 200 元 |

首轮 GPU 会话设为最多 2 小时，且本轮全部消费目标不超过 30 元，先到任一界限就收尾。设 GPU 单价为 p 元/小时、已发生准备费用为 c、此次其他预计费用为 s，则 GPU 可用小时数不超过 min(2, (30-c-s)/p, (200-累计消费-s)/p)，余量非正则不开机。该公式是人工预算约束，不是已部署的自动扣费拦截器。每次开始和结束记录控制台消费，读取/讲解代码期间关机。

按量 GPU 从开机到关机按秒计费；无卡启动按 CPU 模式收费；数据盘扩容、超免费额度文件存储和自定义镜像可能产生额外费用，关 GPU 不代表所有存储费用都消失。公开文档未给出本实例的实时 GPU 价格，本方案不使用示例价格或赠券推算可用时长。[官方计费](https://docs.gpuhome.cc/docs/billing/pricing.html)

开启 GPU 前在平台控制台设置截止时间，并确认实际生效；不以 SSH 断开或 Python 进程退出作为停止计费的证据。平台提供控制台定时关机功能。[官方定时关机](https://docs.gpuhome.cc/docs/instance/scheduled-shutdown.html)

若 30 分钟内仍无法完成 CUDA/引擎导入，或 60 分钟内仍无法完成 base+adapter 的最小生成，保存诊断并关机，在本地分析；不在付费实例上反复源码编译或无限尝试版本。完成后导出记录、核对本地文件再决定是否释放实例和清理收费存储。

## 接入前的依赖与身份检查

本地实际 config 的 architecture 为 LlamaForCausalLM。冻结 adapter 为 rank 4、alpha 8、q_proj/v_proj、dropout 0，base revision 为 12fd25f77366fa6b3b4b768ec3050bf629380bac。vLLM 当前官方表列出 LlamaForCausalLM 及 LoRA 支持，这是选择它进行验证的依据，不能替代本项目模型/adapter 的实际校准。[模型支持表](https://docs.vllm.ai/en/stable/models/supported_models/)；[LoRA 接口](https://docs.vllm.ai/en/stable/features/lora/)

本地 pyproject.toml 要求 torch>=2.14,<2.15，训练 extras 要求 transformers<5。vLLM 安装具有其自身的 PyTorch/CUDA 兼容要求。因此不要直接复制 Windows .venv，不要在云端盲目 pip install -e .[train] 后再覆盖依赖。待确认镜像后先选择可满足的固定版本组合；必要时引擎和训练侧评分分成独立环境，以原生 token 记录交换数据。若两个环境不共享同一 torch 版本，不要声称已经复现本地依赖；需分别冻结版本并记录数值差异。暂不发布未验证的安装命令。[vLLM GPU 安装说明](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

上传材料：代码/配置/训练数据、冻结基线及 adapter/tokenizer、固定 revision 的完整基础模型文件、用于比较的旧 rollout。HF 缓存需保留真实文件内容，避免只上传失效链接；不上传 Windows 虚拟环境。到云端后重新核验 manifest 和 adapter 文件哈希，再加载本地固定模型。不得使用同名模型的最新 revision 代替。

## 首轮验收范围

1. 保存 GPU 型号/显存、驱动、Python、torch、CUDA、transformers、PEFT、vLLM 和安装依赖清单。
2. 用原生 LoRA 接口加载固定 base+adapter；不能默认为合并权重。若需要改为合并方案，单独记录并验证差异。
3. 沿用 4 个训练输入、每题 4 个回答、最多 64 个新 token，temperature=1、top_p=1、无 top-k 截断，固定种子、chat template 和终止 ID。引擎 top-k 禁用值按实际 API 设置，不直接照抄 HF 数值。
4. 保存实际 prompt/response token、EOS、finish_reason、文本和逐 token 概率。核对引擎的 logprobs 语义，不能只凭字段名映射 policy/behavior。
5. 同一固定模型在训练侧完整前向重算实际生成 token 的概率。先确认 token/shift/长度对齐，再根据选定精度明确容差；不能照搬 CPU float32 的 1e-4，也不能根据观察误差无限放宽阈值。
6. 运行既有 reward 审计，报告有效组、零方差比例、错误/重复/截断；不要求这 4 个输入必须产生非零优势。吞吐扩大实验等上述检查通过后再做，并分别记录加载、生成与评分耗时。

跨引擎不要求相同 seed 生成相同 token；验收重点是模型身份、记录完整、概率可对齐及费用可核对。首轮没有真实 GRPO 参数更新。

## 代码阅读顺序

本次没有新增代码。按需要回看以下已有入口，路径相对项目根目录；无需再次逐行阅读已掌握部分。

1. `pyproject.toml`：重点看 Python、torch 和训练 extras 的范围，理解为何云端引擎环境要单独核对。
2. `configs/rollout_cpu_smoke.json`：回看 4 个输入、组大小、采样参数和长度上限，作为引擎迁移的校准基准。
3. `src/agentic_post_training/rollouts/records.py`：`RolloutSample`、`pack_rollout_groups`，确认新引擎输出需要适配的 token/logp/EOS 契约。
4. `src/agentic_post_training/rollouts/hf_sampling.py`：`sample_response`、`replay_policy_log_probs`，回看 policy/behavior 分数和训练侧重算边界。
5. `scripts/collect_rollout_smoke.py`：`main`，关注模型/数据身份和产物清单；未来 GPU 采集器须保留等价检查。
6. `scripts/audit_rewards.py`：`main`，明确云端新记录如何进入当前离线奖励审计。

## 紧接下一步

需要从智川云控制台取得候选卡型、显存、小时价格，以及可选镜像名称与版本（含 CUDA/PyTorch，如有）。据此锁定具体环境组合，准备上传材料和最小引擎脚本，再开始首轮付费运行。当前不需要账号密码、密钥或充值凭据。
