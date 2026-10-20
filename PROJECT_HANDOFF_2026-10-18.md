# AgenticPostTrainingLab 完整交接：截至 2026-10-18

> 2026-10-20 当前进展：已实现训练侧 HF FP32 概率契约与无更新 ratio 检查，新增 14 项 CPU 测试通过，原始 FP32 SFT 参数身份与 LoRA 可训练参数数目核验一致。使用全部已有 16 条 token，分开记录 vLLM sampler behavior 与 trainer old/new；不生成、不反向、不更新。云端检查尚待执行，training_admitted=false；旧跨引擎门失败与 reward 零方差状态保留。最新入口：[训练侧概率契约报告](reports/phase4-trainer-probability-contract.md)。下方为历史更新。

> 2026-10-20 数值诊断已执行：四条 HF FP16 full 重复评分与旧值完全一致；HF 自身 full/cached 最大差 0.0207724571；将同一 FP16 舍入权重转成 FP32 计算后，与旧 vLLM 分数仍不全部对齐，最大 0.0287253857。diagnostic.json 已取回，身份和 24 组误差统计复核一致。本轮有限诊断收尾，不自动提高阈值/追加精度搜索；下一步先明确 sampler behavior 与 trainer old/new 评分契约及无更新 ratio≈1，尚未实现该契约或 GRPO 更新。见 [诊断实际结果与后续边界](reports/phase4-rollout-numeric-diagnosis.md)。

> 2026-10-20 最新执行：L40S 的 4×4 随机 rollout 已生成并取回完整 ZIP，记录/评分/mask/reward/源码身份重审一致，但 10/16 条在原 0.001 概率误差门上失败，最大 0.023653745651245117（已纠正此前截图误读）。3 个算术组仍零方差。已准备对 4 条已有序列做 FP16 完整/缓存强制评分与 FP32 计算对照，不重新采样、不调高阈值、不扩大到 192 条。全套 218 项测试通过；云端有限诊断尚待执行。最新入口为 [数值差异诊断报告](reports/phase4-rollout-numeric-diagnosis.md)，仍处 Phase 4，无训练更新。

> 2026-10-20 后续进展：克隆到 L40S 后，HF/vLLM 固定 SFT 最小生成均成功，54 prompt/28 response token 完全一致，单序列完整前向概率重算在 0.001 逐 token 绝对误差限内通过。用户已读完重算代码并要求进入随机 rollout；已交付固定 4×4 vLLM 采样、分进程重算和无模型奖励/产物审计代码，真实随机运行仍待云端执行。本地全套 211 项测试通过。当前仍是 Phase 4，无 GRPO 参数更新；后续以 [GPU 随机 rollout 报告](reports/phase4-vllm-random-rollout.md) 为最新入口。

> 2026-10-18 后续更新：当前阶段仍是 Phase 4。用户选择先在 CPU 完成 reward 契约与离线审计，再推进云 GPU 资源与引擎接入。奖励接口是对后续 Phase 5 的准备，不表示已进入 Phase 5。新增 23 项测试，全套 160 项通过；现有 3 个算术组均为零方差，真实 GRPO 更新尚未开始。详见 [Phase 4：CPU reward 契约与离线审计](reports/phase4-cpu-reward-contract.md)。下文保留交接整理当时的历史状态。

本文可单独交给另一个对话阅读。它根据原 PROJECT_HANDOFF.md、本对话的推进与代码讲解、当前仓库源码和阶段报告、已保存的训练/评测/rollout 清单重新整理。本文统一描述当前状态，旧报告的“下一步”仅代表当时计划。

本次整理日期为北京时间 2026-10-18；最近一次模型实验与全套测试记录来自 2026-10-17。本次只整理文档并检查文件身份，不重新训练、生成或声称重新执行过全套测试。

## 1. 接手摘要：先读这一节

项目是一个面向工具调用的后训练学习实验：从手写张量损失开始，接真实模型的 SFT＋LoRA、DPO，再推进随机 rollout、可验证奖励、GRPO 和完整工具 Agent。已有工程闭环，不是已经完成的生产级 Agent 或完整 RL 框架。

当前状态：

1. Phase 0/1 的基础环境和 SFT/DPO/GRPO 张量目标已实现或讨论。
2. Phase 2 已完成真实 tokenizer、CPU SFT＋LoRA、保存重载、工具请求执行评测与括号/措辞诊断。
3. Phase 3 最初要求的四步全部完成：冻结 SFT v2 → 小规模偏好数据 → DPO 训练与保存重载 → 逐项生成比较。随后按用户要求追加规模对照、退化诊断、监督锚定修复、真实错误采集与有限同题三组对照，均已完成。
4. Phase 3 最终没有证明 DPO 在共同新开发集上优于继续 SFT；仍有结构退步，任何候选均未替换冻结基线。按预先冻结的有限方案收尾，不因 DPO 没有胜出而继续无边界调参。
5. Phase 4 已完成 CPU 随机 rollout 接口验证：4 题×4 回答，原始 token、两种 log probability、EOS/padding mask、逐条前向重算、一次复现与产物重审均已验证。
6. GPU、vLLM、SGLang、TRL 尚未接入；GRPO 尚未进行真实模型更新；完整多轮工具 Agent 尚未实现。
7. 截至本轮最后一次测试记录：137 项通过、无 skip、1 条已有 tiny PEFT 配置提示。这是 2026-10-17 的历史验证结果。

接手后首先继续用户的代码阅读节奏。用户已读并询问 rollout 的记录结构、分组打包、概率重算、随机采样及 collect_rollout_smoke.py 的总体作用。尚未明确要求跳过剩余阅读并启动 GPU 工作。本次请求只是整理交接材料。

当前没有运行中的训练任务，没有已批准的云 GPU 预算，没有待执行的付费操作。

## 2. 路径、环境、Git 与协作约定

### 2.1 本地位置

```text
项目根目录：
C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab

本项目 Python：
C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab\.venv\Scripts\python.exe

本对话工作目录（不是项目根目录）：
C:\Users\app\Documents\Codex\2026-10-08\x20

便于阅读/带走的导出目录：
C:\Users\app\Documents\Codex\2026-10-08\x20\outputs
```

后文代码模块使用包名或文件名定位；附录给完整绝对路径。执行命令先切到项目根目录，不依赖终端默认 cwd 或虚拟环境自动激活。曾出现 PowerShell 激活失败、裸 python 指向 3.8 的情况，直接用项目解释器最稳妥。

### 2.2 环境快照

| 项目 | 已记录环境 |
| --- | --- |
| 操作系统 | Windows 10，build 19045 |
| Python | 3.11.5 |
| torch | 2.14.0+cpu |
| transformers | 4.57.6 |
| peft | 0.17.1 |
| accelerate | 1.15.0 |
| pytest | 9.1.1 |
| 实际模型计算 | CPU，float32，eager attention，通常 2 线程 |
| 模型缓存 | 项目内 .cache/huggingface |

没有本地 NVIDIA GPU 的既有使用条件；当前环境是 CPU 版 torch。不要直接向这个虚拟环境安装 CUDA/vLLM 栈。未来 GPU 使用独立环境并重新验证依赖组合。pyproject.toml 是版本范围约束，不是完整锁文件。

包名 agentic-post-training-lab，导入名 agentic_post_training，src 布局，Python 范围 >=3.11,<3.13。正式模型已缓存在本地，当前实验入口使用离线模式和 local_files_only；不是把训练权重默认上传 Hugging Face。Hugging Face 提供基础模型来源，本项目产生的 adapter 存在本机。

### 2.3 Git 和文件保存

2026-10-08 已初始化 Git。核对时分支 main，文档整理前工作区干净，无远程配置。最近实现提交 b7ec600（Phase 4 CPU rollout）；本交接文档会作为后续本地文档提交保存。

用户此前要求初始化 Git、整理忽略规则并提交，后续每个已完成的逻辑步骤延续本地提交。没有远程推送授权，也没有创建远程仓库。无需每解释一个函数就提交；完成可独立审阅的实现/报告步骤后提交即可。

忽略规则排除 .venv、.cache、checkpoint、evaluation、trajectory、日志、完整基线快照等；冻结基线的 manifest.json 单独跟踪。文本采用 LF。代码和清单进 Git 不代表模型与逐条实验产物也已备份；仅克隆 Git 无法复现离线模型加载。

### 2.4 用户背景与节奏

用户之前做过 MiniServe，偏推理基础设施与工具 Agent；该项目是用户背景，不是本仓库已经实现的能力。MiniServe 的 KV/prefix/paged cache、Agent loop 不能直接算成本项目成果，也不复用它的虚拟环境。

用户希望通过代码理解后训练，通常逐段提问，再说“代码已读完，进入下一步”。解释时讲清变量、数学、数据流和边界；明确说“简要”时压缩。所有新增 Markdown 用中文。

用户于 2026-10-18 明确要求：以后每次 report 都必须写入“代码阅读顺序”，按推荐顺序列出文件、关键函数及阅读重点；不能只在聊天中提供。没有新增代码的报告应注明本次需要回看的已有入口，或明确说明本次无代码阅读内容。

2026-10-19 资源更新：用户选择智川云，租机累计总预算不超过 200 元。已准备 [Phase 4 智川云资源与接入方案](reports/phase4-zhichuan-cloud-plan.md)，建议首轮不超过 30 元且最多 2 小时；这是支出上限而非消费目标。尚未获取控制台实时卡价/镜像，未创建实例、充值或发生云端费用；下一步先锁定候选规格和环境兼容组合。

用户已提出 Phase 3 花时过长。后续应推进小而完整的步骤，明确验收与停止条件，避免把“学习闭环完成”变成“算法必须赢才准进入下一阶段”。任何外部费用、云实例或明显扩大的实验须先确定资源、预算和停止规则；以前的“进入下一步”不等于批准租机。

## 3. 主体架构与三条数据流

### 3.1 已实现的 SFT 流程

```text
JSONL: id/system/prompt/response
  → 字段、标签、split/保护集隔离审计
  → 真实 tokenizer + chat template
  → encode_sft：检查前缀、回答边界、长度、终止符
  → collate_sft：按长度补齐，labels/loss_mask 区分回答与 padding
  → Instruct base + LoRA
  → response-only causal CE → backward → clip → AdamW
  → adapter/tokenizer/log/summary 保存
  → 新加载相同 base + adapter，检查 logits
  → 贪心输出 → 严格 JSON/schema/route → calculator → 逐题指标
```

### 3.2 已实现的偏好训练流程

```text
同一输入 + chosen/rejected + EvalCase 目标 + provenance
  → chosen 严格正确，rejected 语义错误，family/文件/长度审计
  → 同一冻结 SFT 初始化 policy/reference
  → reference 冻结评分并缓存
  → policy chosen/rejected 序列 logp
  → 纯 DPO 或明确标注的 DPO + chosen NLL
  → 仅更新 LoRA，校验 base/reference 不变
  → 保存重载 → 固定题目真实生成 → 逐项改善/退步与质量门
```

### 3.3 已实现的 rollout 与尚未接上的 RL

```text
冻结 SFT + 训练输入 + 独立采样种子
  → 原生 prompt/response token IDs
  → 原始 policy logp + 实际 behavior logp
  → 完整序列前向重算 policy logp
  → 保存回答与现有严格/数值评分
  → [prompt, generation, response_token] 张量和 mask
  → 分组观察、可复现探针、无模型产物审计

尚未接上：冻结 reward 契约 → advantage → old/new/reference 对齐
          → GRPO 模型反向更新 → 新 rollout → 训练/行为评估
```

当前工具闭环结束于“输出请求、解析、执行计算器、评分”。没有把工具结果回传模型继续生成最终答案，也没有多轮任务状态机。

## 4. Phase 总安排与完成标准

| Phase | 主线 | 当前状态 | 仍需完成或明确的边界 |
| --- | --- | --- | --- |
| 0 | 后训练概念、独立环境 | 已讨论并建立环境 | 不把讨论过当作正式掌握度测验 |
| 1 | 手写 SFT/DPO/GRPO 张量损失 | 实现与测试完成 | 是张量函数，非完整 trainer |
| 2 | tokenizer、SFT＋LoRA、工具请求评测 | 最小真实闭环完成，v1/v2 诊断完成 | 鲁棒性和自然语言运算顺序仍差；没有最终盲测 |
| 3 | 冻结基线、偏好审计、DPO、生成对照 | 四步及有限扩展全部完成，按方案收尾 | 没有 DPO 额外新开发收益，结构回归未解决，基线未晋升 |
| 4 | GPU 环境与 rollout 引擎 | CPU 数据接口前置验证完成 | GPU 资源、引擎、adapter 支持、训练侧分数对齐和吞吐/成本待做 |
| 5 | verifier/reward 与 GRPO | 只有早期张量损失；真实闭环未开始 | reward 审计、零方差组、ratio/KL 契约、更新与评测 |
| 6 | 完整工具 Agent/轨迹评测 | 未开始 | observation 回传、final、停止与恢复、完整轨迹 |
| 7 | 总结与可选架构消融 | 未开始，可选 | 可复现总结；按明确问题做 rank/target/模型规模实验 |

Phase 4—7 是项目路线，可在用户明确选择后调整顺序，并非每个阶段都必须接入所有候选库。vLLM/SGLang 是采样推理引擎，不直接代替 trainer；先掌握一个即可。TRL 是否接入需要单独决定和核验。

Phase 5 验收不仅是 loss.backward 能运行，还要确认 reward 不误奖、组内信号真实、参数身份清楚，监测 reward、零方差比例、KL、clip fraction、长度与真实执行通过率。Phase 6 要区分模型原始请求能力、系统防护效果、最终轨迹成功，不能把自动修复当成模型已学会。

## 5. 固定模型、基线和数据版本

### 5.1 模型身份

```text
基础模型：HuggingFaceTB/SmolLM2-135M-Instruct
固定 revision：12fd25f77366fa6b3b4b768ec3050bf629380bac
冻结基线：sft-v2-phase3
原 SFT v2 run：sft-lora-20261008-202445-a3d10354
baseline manifest SHA256：
a40ef7a505c3f76bbdaf485106a588679bda02d8bab852e0d1a91c8aa9d70f0f
冻结 adapter safetensors SHA256：
0efeb95765ece648571bb20c73fb7818eab3e6e3cd357812551275bd1823f094
同一冻结模型全参数 digest（匹配实验/rollout 所用序列化规则）：
9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb
```

文件 SHA 与 parameter_digest 不是同一东西：后者逐参数累计名字、shape、dtype 和 CPU 连续字节，不能拿它直接替代 safetensors 文件哈希。

冻结快照包含 74 个文件：adapter、tokenizer、历史训练/评测记录、冻结当时源码/配置/数据等；清单另记录 8 个 base 缓存文件。基础模型权重未重复复制到快照。verify 定义于 freeze_sft_baseline.py，核验冻结文件/缓存的身份；不能将“当前工作树后来新增文件”误判为基线被改写。

manifest_path 指清单文件路径；baseline_path 指冻结基线目录。两者本身都是路径，不等于读取权重的动作。真正模型构造是加载固定 base，再加载 checkpoint/adapter；tokenizer 从 checkpoint/tokenizer 读取。reference 是完整的 base＋冻结 SFT v2 adapter，不是裸 base，也不是另一个大模型。

### 5.2 数据版本

| 数据 | train | development | 说明 |
| --- | --- | --- | --- |
| 中文教学 examples | 3 | 2 | 只用于编码教学 |
| 英语最小 examples | 4 | 2 | 3/20 步真实闭环 |
| SFT expanded_v1 | 120（80 tool＋40 final） | 24（16＋8） | 确定性模板，格式/路由/基本算术 |
| SFT expanded_v2 | 144（104 tool＋40 final） | 32（24＋8） | v1 加括号乘法，正式冻结 SFT 起点 |
| 偏好 dpo_v1 | 28（24 math＋4 greeting） | 10（8＋2） | 真实 SFT 错误与合成负例混合 |
| 偏好 dpo_v2 大集 | 168（144＋24） | 56（48＋8） | 形式×问法平衡，全部合成错误负例 |
| 偏好 dpo_v2 小集 | 28（24＋4） | 同一 56 | 大集中的完整 family 子集 |
| 偏好 dpo_v3_matched_errors | 38 个同题 chosen，真实/合成两套 rejected | 原 56 字节一致 | 三臂有限对照，不再追求全因子平衡 |

SFT v2 与偏好 v2 是两套不同的数据，不要因为同叫 v2 混为一谈。SFT v2 从 base＋新 LoRA 开始训练，不是在 SFT v1 上续训；Phase 3 各分支则从冻结 SFT v2 adapter 继续。

dpo_v1 的真实错误/合成负例：train 为 13/15，dev 为 3/7；最多完整 90 token，无截断。偏好来源与题型混杂，不能把分组分数直接归因于来源。

dpo_v2 数学 family 为同一 (a,b,c) 的 4 种结构×3 种问法，每族 12 对：a*(b+c)、a*b+c、(b+c)*a、b+c*a；问法 symbolic/natural/sequence。train 12 个数学 family，dev 4 个；问候 train 6 族、dev 2 族，每族 4 变体。小集使用 seed42 抽 2 个数学族和 1 个问候族，整族保留，28 条精确嵌套于 168 条。

训练数字 a=12..23、b=47..91、c=20..53；开发 a=30..33、b=103..121、c=61..76，存在数字范围/位数外推。共享模板、相关 family、单次小子集抽样，不是独立同分布大量样本。

### 5.3 评测集身份

| 组别 | 规模 | 使用限制 |
| --- | --- | --- |
| 原工具题 | 20＝16 tool＋4 final | 已反复查看并指导扩充，ID 的 holdout 字样不代表现在是盲测 |
| 括号主诊断 | 48＝3 数字组×4 问法×4 结构 | 含已见训练表达式探针；24 对括号/无括号题 |
| 空格控制 | 12 | Calculate 前缀下紧凑表达式，隔离空格/前缀影响 |
| 旧偏好开发 | 10 | 已查看开发题 |
| 新偏好开发 | 56＝48 math＋8 greeting | 24 对数学对照；已查看、模板共享、数字外推 |

五组合计每模型 146 条记录，但只有 145 个唯一输入，且样本相关；不要合并后称为 146 道独立最终测试。没有新的最终未查看测试集。

## 6. Phase 2 已完成实验

| SFT 分支 | 步数 | dev loss 前→后 | 保存重载 |
| --- | --- | --- | --- |
| 4/2 最小 smoke | 3 | 2.163145→1.902697 | logits 最大差 0 |
| 4/2 增加步数 | 20 | 2.163145→0.251388 | 同上 |
| expanded_v1 | 240 | 2.286667→0.036035 | 同上 |
| expanded_v2 | 288 | 2.193550→0.029906 | 同上 |

正式 SFT LoRA：q_proj/v_proj，rank4、alpha8、dropout0、bias none；230400 个可训练参数/134745408 总参数，约 0.171%。CPU float32/eager、batch1、lr0.001、AdamW weight_decay0、clip1、seed42、2 线程；扩充数据各训练两遍，原 SFT 训练顺序固定循环。

原 20 题严格请求执行通过：base 0/20、20-step 9/20、v1 19/20、v2 19/20。base 常输出自然语言，0 分只表示不满足该严格 JSON 协议，不代表无数学能力。

v2 在 `Calculate 2 * (6 + 5).` 输出 `2*6+5` 得 17，目标 22；改为紧凑 `Calculate 2*(6+5).` 可以正确。括号主诊断 v1→v2：严格28→33/48，数值30→33/48，成对9→11/24；顺序问法数值8→4/12退步，自然语言3→6/12改善，空格控制9→12/12。整体提升不等于稳定理解运算顺序。

v1/v2 同时改变数据、步数、开发集，不能把变化单独归因于样本数量或括号覆盖。v2 基线保存的是一个清楚可追踪的起点，不是已解决任务的模型。

## 7. Phase 3 全部实验链与结果

### 7.1 首轮 DPO 与 chosen-only 控制

首轮 28 对、28 步、lr1e-4、beta0.1，同一 SFT v2 初始化 policy/reference。独立 reference 冻结、缓存 detach、仅 LoRA 更新、base 不变、保存重载最大差0均通过。

train/dev DPO loss 从0.693147降到0.465978/0.461117；相对 preference accuracy 为25/28、9/10。初始相同模型的 margin 为0，采用严格 >0，初始 accuracy 记0，不是50%。chosen 平均序列 logp 也下降，不能用偏好指标替代生成。

| 严格通过 | 冻结 SFT | 首轮 DPO | 同28题继续 SFT |
| --- | --- | --- | --- |
| 工具20 | 19 | 20 | 19 |
| 括号48 | 33 | 33 | 33 |
| 空格12 | 12 | 11 | 12 |
| 旧偏好10 | 7 | 8 | 8 |

DPO 括号3改善3退步；继续 SFT 括号数值34、成对12/24，比其严格33不同，但新增1条坏JSON。首轮没有全面赢家。

chosen-only 控制从同一 SFT adapter 开始，核对28个 chosen 的曝光顺序、LR与更新次数；它不是在 DPO 上继续训练。同曝光不等于同 token 计算量，也不等于相同梯度尺度。旧 train_sft_control.py 仍是旧 v1/28 步专用；新38题实验使用新入口。

### 7.2 大小集与步数对照 A/B/C

用户怀疑样本偏少。为区分规模与重复更新，冻结 dpo_v2：

| 分支 | 训练对数 | 步数 | 含义 |
| --- | --- | --- | --- |
| A | 小集28 | 28 | 小集单遍 |
| B | 小集28 | 168 | 小集重复6遍 |
| C | 大集168 | 168 | 大集单遍 |

均 lr1e-4、beta0.1、seed42、纯DPO、同冻结 SFT 起点。“原C”指该完整实验分支及其训练后模型，不是训练集名字。

| 严格通过 | SFT | A | B | C |
| --- | --- | --- | --- | --- |
| 新56 | 36 | 33 | 7 | 4 |
| 工具20 | 19 | 19 | 4 | 2 |
| 括号48 | 33 | 34 | 4 | 1 |
| 空格12 | 12 | 12 | 3 | 0 |
| 旧偏好10 | 7 | 5 | 1 | 1 |

开发相对偏好改善 A45/56、B53/56、C55/56，和生成严重背离。C 开发 chosen 平均logp从−3.643747到−17.491135；rejected从−9.124099到−41.487884，二者都下降但 rejected 降得更多。

B/C 各单独加载 adapter 的两题复现与主评测一致；冻结、shift、mask、保存重载等未发现明显实现错误。这是有限核验，不是数学证明代码没有任何缺陷。严格与数值同时退化，放宽 AST 指标不能解决。

A/B 对照固定小集改变曝光；B/C 同步数但改变独特输入数及重复次数；不能将所有区别简化为“仅数据量”。168条同类合成偏好没有在该配置下救回质量，也不能推断更多真实多样数据无价值。

### 7.3 相对目标诊断与四轮修复

诊断脚本对固定 chosen/rejected 评分，计算 delta_chosen、delta_rejected、相对 margin，按 category、rejected_source、expected_type 汇总，并先复现训练保存的 after 指标。它不重新训练，不把训练对评分当泛化；开发实际行为来自独立生成记录。

纯DPO中，chosen/rejected 的共同回答前缀在对数差里抵消，缺少该前缀的直接偏好梯度；后续 token 的梯度仍可通过共享参数影响格式。144条数学对平均共享约18个回答token，包括JSON/calculator前缀。这是机制线索，不是已量化的唯一原因。

引入明确标注的混合目标：

```text
L_total = L_DPO + alpha * (-log pi(chosen | prompt) / chosen_response_tokens)
```

四轮均168题/168步、同起点/顺序、beta0.1：

| alpha / LR | 新56严格 | 新56数值/路由 | 工具20 | 括号48 | 空格12 | 旧偏好10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / 1e-4 | 31 | 31 | 19 | 29 | 12 | 5 |
| 5 / 1e-4 | 36 | 36 | 19 | 34 | 12 | 6 |
| 5 / 2e-5 | 35 | 42 | 20 | 34 | 12 | 7 |
| 5 / 1e-5 | 36 | 44 | 20 | 34 | 12 | 7 |

相邻第二到第三、第三到第四轮只改 LR；最后显著恢复，但新56仍未超过SFT，且存在旧题结构退步。不能把混合目标叫“纯DPO成功”，也没有同168题/168步的纯SFT控制证明混合收益来自DPO部分。

alpha=0 分支保持纯DPO损失及梯度。DPO用序列总logp，chosen NLL按回答token平均，包含EOS不含prompt/pad；alpha=1不意味着两项尺度相等。训练日志 loss 是总目标，而 before/after 评估 loss 仍是纯DPO诊断，应区分。

第二轮保存 reference 分数；第三/四轮在身份、源码、数据、张量检查及4次实际前向探针后复用。reference_log_probs.pt 是分数缓存，不是模型权重。前两轮精确历史源码保存在评测 source 目录，不能随意改旧源码后声称历史哈希仍对应当前入口。

### 7.4 逐题结构门为什么必须保留

旧题 `bracket-seen_large-sequence-left_plain`：`Multiply 12 by 3 first, then add that result to 41.`

SFT 输出合法JSON，但算式仅 `12*3`；修复候选输出含 `"expression":"12*3+"41"`，成为坏JSON。两者端到端都错，transition 是 both_fail，总成功数隐藏了结构恶化。

因此追加逐题 JSON/schema/route 新退化拒收门，并覆盖“总分不变仍有单题恶化”的测试。此门是诊断后补充，不改写原先五项验收条件的历史。恢复候选既未获得新开发净提升，也未通过该结构门，基线不晋升。

### 7.5 训练输入真实错误采集

对 dpo_v2 的168个训练输入，用冻结 SFT 贪心生成：严格108、语义130。排除22个数值正确但AST不同的回答，得到38个真实错误候选：36算式错误、2坏JSON；没有因空回答/长度上限/超长而过滤的实际条目。

覆盖12个数学family，每族2—4条，问法 natural29/symbolic7/sequence2，没有问候。它们不是完整12格平衡族，不能宣称仍满足 v2 全因子设计。chosen/rejected完整编码最大90/91，低于192。

与原56开发及444条保护记录的既定精确隔离检查通过。38个实际错误中仅12个与原同题合成负例的AST或结果匹配，其余26个不匹配；说明覆盖有缺口，不证明这是全部退化原因。采集仅用训练输入，开发答案不加入训练。

### 7.6 最终38同题三臂有限对照

冻结数据：train_chosen.jsonl、train_synthetic.jsonl、train_real.jsonl 各含相同38个输入与chosen，原56开发字节一致。真实/合成负例12条相同、26条不同。

计划 SHA256：a3520bf40a59e07e058b0cb0804571e4d999b197c3ca36de0b110b5601465405。

三臂均从冻结 SFT 出发，38步单遍、同ID/顺序、lr1e-5、seed42、clip1、AdamW无weight decay；DPO beta0.1、alpha0。固定三次训练，不追加超参数搜索。相同chosen曝光1108回答token；rejected分别0/1116/1043，计算量不同。

冻结/base/reference/LoRA更新/重载全部通过，最大重载差0。继续SFT chosen NLL从0.321789到0.285516。合成DPO loss0.693147→0.646207，chosen Δ+0.426050/rejected Δ−0.544117；真实DPO loss→0.648045，chosen Δ+0.879285/rejected Δ−0.050737。真实组相对改善37/38，但原始平均policy margin仍−7.563507。

| 指标 | 冻结SFT | chosen-only SFT | 合成负例DPO | 真实负例DPO |
| --- | --- | --- | --- | --- |
| 新56严格 | 36 | 36 | 36 | 36 |
| 新56语义 | 44 | 44 | 44 | 44 |
| 工具20严格 | 19 | 20 | 19 | 20 |
| 括号48严格 | 33 | 34 | 33 | 34 |
| 空格12严格 | 12 | 12 | 12 | 12 |
| 旧偏好10严格 | 7 | 7 | 7 | 7 |
| 全组JSON合法 | 146/146 | 145/146 | 145/146 | 145/146 |

新开发数学数值36/48、成对16/24、问候8/8四模型相同；natural数值4/16，symbolic和sequence各16/16。各模型无生成上限命中。

真实DPO与继续SFT的新56文本完全相同，全部146记录仅1条文本不同且都失败：题目 `Add 4 to the product of 3 and 7.`，目标3*7+4，SFT为3*(7)/2，真实DPO为3*(7)。所有组严格/语义的二者改善与退步均0。

三个新模型都在7.4所述同题输出相同坏JSON。该结构回归也出现在继续SFT，不能全部归因于DPO。恢复质量门失败；两种DPO额外收益门也失败。保留SFT v2基线，按方案结束Phase 3。

实际435次候选新生成＋3次SFT重现探针＝438次。每模型一个重复输入复用缓存，SFT146条使用核验后的历史预测；总584条保存记录重算，六组对照、30份changes、20份预测文件身份一致。不要说584次全新生成或584道独立题。

`--audit-only --require-admission` 的历史实际退出码为2：artifact_audit_passed=true，但 any_dpo_added_value=false。审计一致与质量达标是两个不同结论。

冻结数据清单仍写 training_started=false / runner_adaptation_required=true，表示冻结当时状态；实际训练已完成并记录在后来的run/summary/Git。不要修改冻结文件来“纠正”历史字段，否则破坏计划哈希。

## 8. 如何客观理解 Phase 3

已证明的事实：工程链条能训练、冻结参考、保存恢复并受控评测；相对偏好改善可与实际生成退化并存；该配置中大集纯DPO严重退化；锚定和保守更新能缓解；最终匹配对照未证明DPO对新开发集有额外收益。

| 可能因素 | 证据强度与边界 |
| --- | --- |
| 相对目标不保证chosen绝对概率上升 | 原C实际观测明确；不能单凭此断言所有退化只由该因素导致 |
| 更新幅度/曝光 | B/C坏、修复中降低LR恢复；最终小LR也未提升；未做完整纯DPO单变量矩阵 |
| 固定负例覆盖真实错误不足 | 38训练错误仅12匹配原负例；最终真实替换仍无新开发收益 |
| 样本少且高度相关 | 数据规模、模板、family事实明确；未证明“增加数量就能解决” |
| 数字范围外推、自然语言分布窄 | 数据和分组失败支持假设，未独立消融 |
| 135M模型/rank4 q-v LoRA容量限制 | 合理假设，尚未做模型/rank/target对照 |
| 最后38步/LR1e-5可能更新不足 | margin仍负；不能据此擅自追加训练或保证加步数有益 |

不能得出的结论：DPO普遍无效、纯DPO必然崩溃、样本数量是唯一原因、真实负例一定胜合成、混合目标的收益必来自DPO、实现检查通过等于模型质量成功。

Phase 3 的学习与工程验收完成，模型泛化改进目标未完全达到；有限负结果本身需要如实保存，不需要把报告包装成正结果。

## 9. Phase 4 当前实现、结果与阅读位置

### 9.1 本步固定范围

冻结 SFT v2；从偏好 dpo_v2 的 train 选择4题：12/47/20的 symbolic-right_grouped、natural-right_plain、sequence-left_plain，另 greeting-0-0。前三题共享数字family，人为选择且包含已知错误，不是独立开发评测。

每题4次采样，seed=42+group_index*1000+sample_index。temperature1、top_p1、top_k0、repetition_penalty1、max_new_tokens64、单beam、每调用一个回答、CPU2线程。显式 use_model_defaults=False。另复现首条一次，共17次生成。

### 9.2 各文件职责

| 文件/对象 | 职责 |
| --- | --- |
| RolloutSample（records.py） | group/case/index/seed，原生输入/回答ID，两类逐token logp，EOS、结束原因、text/raw_text |
| pack_rollout_groups | 检查同组同输入和唯一编号，按实际长度打包四个三维张量 |
| replay_policy_log_probs（hf_sampling.py） | 拼原始ID完整前向，causal shift后取回答，核验原始policy logp |
| sample_response | 显式随机配置，独立CPU种子，抽取raw logits与processed scores，保存并重算 |
| collect_rollout_smoke.py | 基线/数据/源码身份，选择训练题，采样与评分，复现/冻结检查，落盘和audit-only |
| rollout_cpu_smoke.json | 本次固定4题、组大小、种子、温度与长度配置 |
| test_rollout_records.py | EOS/PAD、非法记录、组内一致性与编号 |
| test_rollout_sampling.py | 随机tiny Llama实际采样、概率重算、温度分布差、RNG与权重不变 |

RolloutSample检查token非空且为非负int、logp数量匹配/有限/不明显大于0、EOS只能出现在末尾，末尾EOS与finish_reason一致。frozen=True仅阻止字段重新赋值，不深度冻结内部list。它不是所有运行不变量的唯一检查层；长度上限、解码一致、实际概率重算由采样器和采集器补充。

### 9.3 两种概率、温度与复核

policy logp来自未处理logits的log_softmax，是模型π对实际token的条件概率；behavior logp来自采样处理后scores，是抽样分布q。温度T通过softmax(logits/T)改变分布：T<1更集中、T>1更平缓，不修改模型权重。当前T1且无过滤，两种分数实际差0；更换温度/top-p后通常不同。

不要未经目标推导就把π和q任意一个塞进未来GRPO的old_log_probs。未来必须明确采样分布、更新目标及是否需要分布校正；本步只是分别记录，不声称已完成这项RL设计。

生成时使用KV cache，重算时不使用cache。重算直接拼prompt_ids+response_ids，不先decode再tokenize；后者可能改变序列。causal_token_log_probs已shift一次，回答切片从len(prompt_ids)-1开始；输出已与response对齐，打包后不能再shift。

核验意义：概率值数量正确不代表对应token正确，错一位也可能静默产生坏训练信号。逐步生成与完整teacher-forced前向两条路径在同模型/同上下文下应接近；允许浮点误差，不要求bitwise一致。

fork_rng(devices=[])保存并恢复CPU随机状态，块内manual_seed控制该样本，不消耗调用者CPU RNG流；当前CPU限定不能外推为已实现CUDA RNG隔离。首条重复只说明同环境该样本可复现，不保证跨引擎/设备/版本同seed必同token。

### 9.4 mask与EOS

张量形状[问题数,每题回答数,最长回答token数]。tokens预填PAD；mask默认False；各回答前n个位置写入实际ID/logp并置True。EOS=PAD=0时，真实EOS仍有效：

```text
token: [15, 26, EOS=0, PAD=0, PAD=0]
mask:  [ 1,  1,     1,     0,     0]
```

不能用tokens!=pad_token_id建立mask。填充位置logp的0只是占位，后续必须mask掉。response_batch不是完整HF输入batch，prompt仍在记录内。

### 9.5 实际观察

| 组 | 不同文本数 | 严格通过/4 | 语义通过/4 |
| --- | --- | --- | --- |
| symbolic | 1 | 4 | 4 |
| natural | 3 | 0 | 0 |
| sequence | 3 | 0 | 1 |
| greeting | 4 | 4 | 4 |

16条严格8、语义9、坏JSON4，全部EOS结束，无截断。400个有效回答token含16EOS；batch [4,4,41]，656总位置，256 padding排除。模型无梯度/无更新，首样本复现。policy完整前向最大差4.1961669921875e-05，通过atol=rtol=1e-4。

16条采样＋重算约57.75秒，不包括模型加载和额外复现探针，不是纯解码吞吐。不能拿这个4题接口结果与Phase 3贪心通过率直接比较。

四组严格0/1结果都恒定：若直接作为reward，组内优势全0，没有该reward带来的相对策略梯度。文本不同不等于奖励不同。语义仅sequence组混合1/4，实际20*12+47与目标47+20*12数值相同但AST不同。不能只为制造非零方差而随意改reward；尚未冻结训练reward。

### 9.6 产物和audit-only边界

六个产物：run_manifest.json、rollouts.jsonl、run_checks.json、response_batch.pt、summary.json、verification.json。

run_manifest固定模型、adapter、数据、源码、依赖与采样语义；rollouts逐条保存sample/case/score/semantic/replay误差；run_checks保存初末权重与复现样本。错误/重复/空/截断回答均应保留，本次没有后两类。

历史audit-only已通过：无模型权重加载，重算原ID解码、规则评分、分组、mask/张量并核验清单。policy概率完整前向验证发生在实际采集时；无模型审计不重新计算这些概率。

## 10. 关键实现契约与容易混淆的概念

### 10.1 数据/模型/目标

- PreferenceExample.target()生成EvalCase标准目标对象，含输入、预期路由、表达式/结果；不是返回chosen字符串，也不是把模型预测标准化。
- load_preferences逐行读取JSONL，定位文件:行号错误，拒绝空集/重复ID。validate_preferences审查family、ID/input/prompt/精确表达式AST隔离，chosen严格通过、rejected语义不通过，并检查token长度与统计。
- build_balanced_preferences负责构造大集/小集/开发集；audit_balanced_design核对完整形式×问法、路由数量、整族子集；脚本还负责写出/重建审计。它不只是读取已有数据“调用一下”。
- load_preference_splits只允许manifest固定的训练文件，不允许development；核验所有清单文件，即使这次没选大的train也要检查它；拒绝路径越出数据目录。保护的是完整版本身份。

### 10.2 SFT、mask与shift

encode_sft要求prompt token是完整序列的真实前缀，超max_length报错而非截断JSON。prompt/回答/padding分别处理：attention为1/1/0，labels为−100/token/−100，loss_mask为0/1/0。真实EOS属于回答目标。

attention_mask并不是loss_mask；prompt虽然没有直接目标loss，仍影响回答上下文和梯度。HF内部shift一次，手写causal函数也只shift一次；手写gather输入是input_ids，不能用含−100的HF labels。

LoRA是训练哪些参数的方法，与SFT/DPO目标正交，不是“先SFT再单独做LoRA”。保存adapter不含完整base，也通常不含optimizer状态，不能宣称支持任意步续训。

### 10.3 DPO数学与日志

```text
m_policy = log pi(chosen|x) - log pi(rejected|x)
m_ref    = log ref(chosen|x) - log ref(rejected|x)
relative_margin（当前日志）= beta * (m_policy - m_ref)
L_DPO = -log sigmoid(relative_margin)
```

代码字段relative_margin已经乘beta，不要再次乘或误报为未缩放差。positive只说明相对reference更偏好chosen，不代表m_policy>0，不代表chosen绝对概率上升，不代表实际贪心生成chosen。

learning_rate控制优化器参数更新尺度。beta在DPO中缩放上述对数比差，源于KL正则化偏好优化推导；当前实现没有额外计算一个显式KL项再相加。不要把beta粗略当作学习率替代品，或断言增大beta必然使参数更新单调更强/更弱。

cache_reference需要完全冻结且无梯度的reference，eval/no_grad评分并detach/clone。objective也detach reference。底层张量dpo_loss不是自动冻结任意传入reference的trainer，冻结职责要看调用层。

train_dpo检查仅LoRA可训练，AdamW、按seed洗牌、每步一对、清梯度→前向→反向→裁剪→更新。记录loss/logp/margin来自该步更新前前向；grad_norm是裁剪前返回范数。训练后evaluate才是更新后的完整评分。

### 10.4 评分与逐项比较

工具schema严格只接受calculator请求或final答复；拒绝多余键、重复JSON键、非标准常量、围栏和前后解释文本，不修复预测。calculator为受限AST解释器，支持有限数字与+−*/及一元正负，不用eval；长度、节点、数值范围/除零均有限制。

严格参数匹配使用含具体数字的AST，忽略空白/冗余括号但不忽略操作数顺序。semantic_pass只查合法结构/路由与当前数值结果，容差1e-9；不是符号等价证明，常数或巧合结果可能过单题。问候只检查final非空与路由，不判断回复内容质量。

paired汇总用于明确设计的成对算术：括号和无括号必须同时正确，不能用单题总分代替；普通任务没有这些配对元信息。新v2 category不是旧brackets的编码结构，使用对应scaling逻辑。

compare_records按ID对齐且核对输入/目标，输出both_pass/both_fail/improved/regressed，并独立记录prediction_changed。文本改变未必得分改变；总分持平可能有多题相反变化；both_fail可能从算术错恶化为坏JSON。

### 10.5 当前GRPO函数的边界

group_relative_advantages对[B,G]奖励组内标准化，std使用unbiased=False，std<=eps组置0。grpo_clipped_loss接[B,G,T]逐token概率，ratio=exp(new−old)，取min(ratio*A,clip(ratio)*A)，可选KL估计exp(ref−new)−(ref−new)−1。

当前按整个batch有效token总数归一化，不是每条completion先平均；是教学变体，不应宣称逐项复现某库默认GRPO。reference/old的detach、版本管理与behavior分布校正仍需未来trainer明确。beta>0缺reference时底层函数不会自动报错，trainer需建立更严格配置契约。

clip_fraction是ratio越界比例，不等于所有越界token都处于平坦裁剪区，需看优势符号。全零advantage表示reward策略项无信号，不代表任何配置下KL项也必然无梯度。

## 11. 下一步建议与未完成清单

### 11.1 紧接当前对话

用户最近已逐段读过records.py、hf_sampling.py，并询问为什么要重算概率；已得到collect_rollout_smoke.py简要说明。可以继续讲采集器主流程、产物审计和测试；不需要重新采样来解释代码。

若用户明确说进入下一实现步骤，先做Phase 4资源与引擎方案：确认是否已有可用Linux/NVIDIA GPU；若需云端，准备具体可审阅配置、预计使用范围、预算上限与释放规则后获得用户授权。现有CPU文档不是租机授权。

### 11.2 GPU/引擎接入顺序

1. 确定资源、成本边界和独立Linux/CUDA环境，不改变当前可复现CPU环境。
2. 在接入当时核对官方版本支持，包括SmolLM2架构、LoRA q/v adapter、tokenizer和chat template；不能预先承诺支持。
3. 固定base revision、adapter哈希、EOS/PAD、截断/停止、采样配置，查清引擎logp是raw还是processed。
4. 少量输入校准原生token、长度、结束原因、训练侧完整前向评分；不能要求跨引擎同seed必然逐token一致。
5. 通过后扩大多prompt×多generation批量，分别测解码、评分、显存、吞吐和成本。
6. 再推进reward审计及真实GRPO更新；若用户选择先做CPU离线reward审计，明确记录路线调整。

### 11.3 尚未完成的实质工作

- 真正未查看的最终测试、稳健自然语言/数字泛化、多种子与系统单变量消融。
- 结构回归问题；现有候选未达到晋升门，SFT v2也仍有算术错误。
- GPU/vLLM或SGLang实际集成、模型/adapter支持验证、吞吐与成本实验。
- 明确可验证reward定义与反例审计、零方差监控、old/new/reference与behavior契约。
- 真实GRPO训练、checkpoint/恢复、行为比较；不能拿张量单测代替。
- 工具结果回传、最终答复、多步Agent、安全停止/恢复及轨迹级评测。
- 可迁移环境锁定与完整产物备份；Git当前不包含大部分运行产物。

不自动重启Phase 3，不把开发错误答案转入训练，不追加LR/beta/alpha搜索，也不覆盖历史数据、adapter、manifest与原评测。新问题需要新的明确实验问题与方案。

## 12. 安全的接手核验命令与判断方式

以下命令用于需要核验时，不要求为了读文档全部执行。

```powershell
Set-Location "C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab"
git status --short
git log -6 --oneline
& ".\.venv\Scripts\python.exe" -c "import sys; print(sys.executable); print(sys.version)"
& ".\.venv\Scripts\python.exe" scripts\freeze_sft_baseline.py --verify artifacts\baselines\sft-v2-phase3
& ".\.venv\Scripts\python.exe" scripts\collect_rollout_smoke.py --output artifacts\trajectories\rollout-smoke-phase4-20261017-01 --audit-only
```

需要代码回归检查时：

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -q
```

匹配实验质量门重审：

```powershell
& ".\.venv\Scripts\python.exe" scripts\compare_matched_control.py --training-run artifacts\checkpoints\matched-control-v3-20261017-01 --output artifacts\evaluations\matched-control-v3-20261017-01 --audit-only --require-admission
```

历史结果预期是退出2，表示质量拒收；先看verification的artifact_audit_passed和any_dpo_added_value。去掉require-admission退出0仅说明记录一致。

这些audit入口仍可能更新verification等派生产物，不等于纯读取；但不会启动训练或模型生成。默认collect命令会重新采样，新训练入口会产生新run，因此不要为了恢复上下文调用默认训练/生成命令。

当前137测试覆盖张量、编码、EOS/PAD、数据隔离、评分器、真实tiny PEFT训练/冻结/重载、DPO/anchor、曝光和质量门、rollout记录与实际采样。tiny集成不下载正式135M模型；有些旧集成在缺依赖时会skip，迁移后要检查是否实际执行。唯一已有PEFT警告来自随机tiny模型没有原始config路径，不能误称正式训练失败。

## 13. 文档、源码和产物索引

下面附录由本次检查到的文件生成，提供完整绝对路径与身份摘要，便于新对话直接打开。训练与评测目录名按实际保存名保留，不根据当前日期重命名。

阅读优先级：本文 → Phase 4 rollout报告与源码 → Phase 3匹配对照结果 → 需要时追溯repair/scaling报告 → 旧handoff。旧handoff保留原始历史细节，但其中“尚无Phase4”“待适配38题”等阶段性措辞已被后续工作覆盖。

## 14. 给新对话的启动提示

> 请先完整阅读 C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab\PROJECT_HANDOFF_2026-10-18.md，并核对当前Git和必要产物。项目已完成CPU SFT＋LoRA、Phase 3 DPO全闭环及有限扩展。最终38同题三组未证明DPO有额外新开发收益，仍有结构回归，已按方案收尾，不要重新扫超参数或替换冻结SFT基线。Phase 4已完成4题×4回答的CPU随机rollout、原生token与两类概率、EOS/mask、前向重算和审计；vLLM/GPU及真实GRPO尚未接入。用户按代码阅读后逐步推进，最近读到records.py、hf_sampling.py及collect_rollout_smoke.py总体作用。先说明当前进度，再按用户本次请求继续；不要默认租GPU、扩大实验或重跑历史训练。使用本项目虚拟环境，新增Markdown中文。所有当前评测均为已查看开发诊断，不是最终盲测。

如果新对话没有本地文件权限，请上传本文件；需要具体代码或核验时再附相关源码/JSON。Markdown里的本地路径不会自动把文件、权重或缓存传给对方。

## 附录 A：核心源码入口（已核对存在）

| 入口 | 职责 |
| --- | --- |
| [src/agentic_post_training/algorithms/losses.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/algorithms/losses.py) | causal shift、masked CE、序列logp、DPO、组内advantage与GRPO教学损失 |
| [src/agentic_post_training/data/sft.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/data/sft.py) | SFT记录、JSONL、编码与按长度padding |
| [src/agentic_post_training/data/hf_tokenizer.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/data/hf_tokenizer.py) | HF tokenizer适配和特殊token处理 |
| [src/agentic_post_training/data/validation.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/data/validation.py) | SFT标签、目标和跨集精确隔离 |
| [src/agentic_post_training/data/preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/data/preferences.py) | 偏好记录、目标、训练文件选择、哈希/语义/隔离审计 |
| [src/agentic_post_training/data/balanced_preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/data/balanced_preferences.py) | v2形式×问法构造、整族小集和设计审计 |
| [src/agentic_post_training/trainers/sft_lora.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/trainers/sft_lora.py) | LoRA配置、训练/开发CE、训练循环 |
| [src/agentic_post_training/trainers/dpo.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/trainers/dpo.py) | 配对编码/logp、reference缓存、DPO/anchor目标及更新 |
| [src/agentic_post_training/environments/calculator.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/environments/calculator.py) | 有界AST计算器 |
| [src/agentic_post_training/evaluation/tool_calls.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/tool_calls.py) | EvalCase、严格解析/schema/route/执行评分 |
| [src/agentic_post_training/evaluation/generation.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/generation.py) | 历史评测共用CPU贪心生成 |
| [src/agentic_post_training/evaluation/brackets.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/brackets.py) | 旧括号诊断、数值与成对汇总 |
| [src/agentic_post_training/evaluation/comparison.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/comparison.py) | 同ID输入核对、逐项transition、汇总 |
| [src/agentic_post_training/evaluation/scaling.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/scaling.py) | v2开发结构、曝光与规模对照汇总 |
| [src/agentic_post_training/evaluation/preference_diagnostics.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/evaluation/preference_diagnostics.py) | chosen/rejected绝对变化及分组统计 |
| [src/agentic_post_training/rollouts/records.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/rollouts/records.py) | RolloutSample与三维分组张量 |
| [src/agentic_post_training/rollouts/hf_sampling.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/src/agentic_post_training/rollouts/hf_sampling.py) | CPU随机采样、两类logp和完整前向重算 |
| [scripts/inspect_sft_data.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/inspect_sft_data.py) | 教学tokenizer编码检查 |
| [scripts/inspect_hf_tokenizer.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/inspect_hf_tokenizer.py) | 真实模板/回答/EOS检查 |
| [scripts/validate_sft_dataset.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/validate_sft_dataset.py) | SFT扩充数据独立完整审计入口 |
| [scripts/train_sft_lora.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/train_sft_lora.py) | 原始SFT训练与保存重载 |
| [scripts/evaluate_tool_calls.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/evaluate_tool_calls.py) | 原20题工具请求比较 |
| [scripts/diagnose_brackets.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/diagnose_brackets.py) | v1/v2括号与控制题生成诊断 |
| [scripts/freeze_sft_baseline.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/freeze_sft_baseline.py) | 冻结快照与verify；--verify需要目录参数 |
| [scripts/build_dpo_preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/build_dpo_preferences.py) | 首轮偏好数据构造/审计 |
| [scripts/train_dpo.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/train_dpo.py) | 历史DPO/anchor真实模型主入口 |
| [scripts/compare_sft_dpo.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/compare_sft_dpo.py) | 首轮SFT与DPO行为对照；对话中曾误写compare_str_dpo.py |
| [scripts/diagnose_dpo_preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/diagnose_dpo_preferences.py) | 固定偏好概率重评分；真实文件名是复数preferences |
| [scripts/train_sft_control.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/train_sft_control.py) | 旧v1/28步chosen-only控制，不是新38题通用入口 |
| [scripts/compare_sft_control.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/compare_sft_control.py) | 旧chosen-only与历史SFT/DPO三分支行为对照 |
| [scripts/build_balanced_preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/build_balanced_preferences.py) | v2平衡偏好数据写出/审计 |
| [scripts/summarize_dpo_scaling.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/summarize_dpo_scaling.py) | A/B/C训练指标、样本曝光与身份汇总；文件名不是summary_dpo_scaling.py |
| [scripts/compare_dpo_scaling.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/compare_dpo_scaling.py) | A/B/C共同实际生成和新旧分组比较 |
| [scripts/compare_dpo_repair.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/compare_dpo_repair.py) | anchor/LR修复候选、控制核验、行为质量门 |
| [scripts/audit_dpo_repair.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/audit_dpo_repair.py) | 修复产物无模型重算审计与质量拒收 |
| [scripts/collect_sft_training_errors.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/collect_sft_training_errors.py) | 168训练输入SFT生成，筛38真实错误与重审 |
| [scripts/build_matched_preferences.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/build_matched_preferences.py) | 冻结38同题三套训练文件及有限实验计划 |
| [scripts/matched_control_common.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/matched_control_common.py) | 冻结计划、数据、曝光与收益/结构门共用逻辑 |
| [scripts/train_matched_control.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/train_matched_control.py) | 38题chosen-only/合成DPO/真实DPO三臂执行 |
| [scripts/compare_matched_control.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/compare_matched_control.py) | 三臂与SFT共同生成、六对照、无模型重审 |
| [scripts/collect_rollout_smoke.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/scripts/collect_rollout_smoke.py) | 当前4×4采样、概率/模型核验、落盘及重审 |
| [configs/rollout_cpu_smoke.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/configs/rollout_cpu_smoke.json) | 当前固定采样配置 |

包内rewards、benchmarks及docker目录目前没有已实现功能；名字存在不等于已接入。测试具体文件见tests/unit与tests/integration，所有正式执行入口以scripts下实际名字为准。

## 附录 B：阶段报告（历史叙述按日期解读）

- [Phase 1：后训练目标函数基础](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase1-foundations.md)
- [Phase 2：乘法与括号组合数据 v2](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-bracket-data.md)
- [Phase 2：括号与文字问法诊断](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-bracket-diagnostic.md)
- [Phase 2：扩充格式与路由数据](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-expanded-data.md)
- [Phase 2：LoRA 最小训练闭环](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-lora-training.md)
- [Phase 2：接入真实 tokenizer](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-real-tokenizer.md)
- [Phase 2：SFT 数据流水线与 LoRA 训练](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-sft-data.md)
- [Phase 2：独立工具调用评测](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase2-tool-evaluation.md)
- [Phase 3：均衡偏好数据 v2 与规模对照设计](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-balanced-preferences.md)
- [Phase 3 第 4 步：SFT-only 与 SFT＋DPO 逐项行为比较](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-behavior-comparison.md)
- [Phase 3 第 1 步：冻结 v2 SFT 基线](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-dpo-baseline.md)
- [Phase 3：DPO退化定位与稳定性恢复](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-dpo-repair.md)
- [Phase 3：v2 DPO规模对照的实际生成评测](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-dpo-scaling-behavior.md)
- [Phase 3：v2 DPO 规模对照训练](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-dpo-scaling-training.md)
- [Phase 3 第 3 步：DPO 训练、保存与重载](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-dpo-training.md)
- [Phase 3 结果回顾与38同题对照冻结](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-matched-control-plan.md)
- [Phase 3：38同题三组对照执行与结果](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-matched-control-results.md)
- [Phase 3 第 2 步：小规模偏好数据](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-preference-data.md)
- [Phase 3 后续：偏好概率分组诊断与受控对照设计](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-preference-diagnostic.md)
- [Phase 3：chosen-only 继续 SFT 对照](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-sft-control.md)
- [Phase 3 后续：采集冻结 SFT 的训练错误](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase3-sft-training-errors.md)
- [Phase 4 第一步：CPU 随机采样与 rollout 数据接口](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/reports/phase4-rollout-contract.md)

## 附录 C：关键运行产物（本地目录已核对存在）

目录内summary.json是机器可读汇总，report.md是中文解释，*_changes.json是逐题变化。pair_scores.jsonl位于偏好概率诊断目录；rollouts.jsonl位于trajectory目录。adapter位于训练run内，不在这些评测目录内。

| 实验 | 本地目录 |
| --- | --- |
| 冻结SFT基线 | [sft-v2-phase3](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/baselines/sft-v2-phase3) |
| SFT 3步 | [sft-lora-20261008-002302-e5bfa695](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/sft-lora-20261008-002302-e5bfa695) |
| SFT 20步 | [sft-lora-20261008-005426-9d8f2036](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/sft-lora-20261008-005426-9d8f2036) |
| SFT v1 | [sft-lora-20261008-180828-05ced0f0](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/sft-lora-20261008-180828-05ced0f0) |
| SFT v2 | [sft-lora-20261008-202445-a3d10354](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/sft-lora-20261008-202445-a3d10354) |
| 原20题v2评测 | [tool-calls-20261008-203842-cc578a24](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/tool-calls-20261008-203842-cc578a24) |
| 括号主诊断 | [bracket-diagnostic-20261008-204902-76e6b53b](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/bracket-diagnostic-20261008-204902-76e6b53b) |
| 空格控制 | [bracket-diagnostic-20261008-210057-380ef02a](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/bracket-diagnostic-20261008-210057-380ef02a) |
| 首轮偏好构造日志 | [dpo-preferences-v1](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-preferences-v1) |
| 首轮DPO训练 | [dpo-lora-20261009-205045-21dd3596](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261009-205045-21dd3596) |
| 首轮DPO行为 | [sft-dpo-20261010-150134-fc7e5e9a](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/sft-dpo-20261010-150134-fc7e5e9a) |
| 偏好概率诊断（含pair_scores.jsonl） | [dpo-preference-diagnostic-20261010-181453-22baa0cd](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-preference-diagnostic-20261010-181453-22baa0cd) |
| 旧chosen-only训练 | [continued-sft-20261011-181322-52e2bc03](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/continued-sft-20261011-181322-52e2bc03) |
| 旧chosen-only行为 | [sft-control-20261011-181518-0ef09401](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/sft-control-20261011-181518-0ef09401) |
| A 小集28步 | [dpo-lora-20261016-141339-249a77f0](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-141339-249a77f0) |
| B 小集168步 | [dpo-lora-20261016-142317-5d529ace](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-142317-5d529ace) |
| C 大集168步 | [dpo-lora-20261016-145728-bc1fc34c](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-145728-bc1fc34c) |
| A/B/C训练汇总 | [dpo-v2-scaling-training-20261016-152209-97915019](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-v2-scaling-training-20261016-152209-97915019) |
| A/B/C行为汇总 | [dpo-v2-scaling-behavior-20261016-163551-cb6e9477](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-v2-scaling-behavior-20261016-163551-cb6e9477) |
| 修复1训练 | [dpo-lora-20261016-180833-e336ae0b](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-180833-e336ae0b) |
| 修复1行为 | [dpo-anchor-20261016-182453-d989d3f1](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-anchor-20261016-182453-d989d3f1) |
| 修复2训练 | [dpo-lora-20261016-184519-79d1335a](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-184519-79d1335a) |
| 修复2行为 | [dpo-anchor-20261016-190110-56ef6fae](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-anchor-20261016-190110-56ef6fae) |
| 修复3训练 | [dpo-lora-20261016-191348-bfddf667](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-191348-bfddf667) |
| 修复3行为 | [dpo-anchor-20261016-192955-24ca2895](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-anchor-20261016-192955-24ca2895) |
| 修复4训练 | [dpo-lora-20261016-194142-df9d5829](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/dpo-lora-20261016-194142-df9d5829) |
| 修复4行为 | [dpo-anchor-20261016-195656-ae7cb3a7](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-anchor-20261016-195656-ae7cb3a7) |
| 四轮修复总汇 | [dpo-repair-summary-20261016-201033-4f337b8d](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/dpo-repair-summary-20261016-201033-4f337b8d) |
| 168题真实错误采集 | [sft-training-errors-20261017-200708-66935487](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/sft-training-errors-20261017-200708-66935487) |
| 38同题冻结数据和方案 | [dpo_v3_matched_errors](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/data/preferences/dpo_v3_matched_errors) |
| 38同题三臂训练 | [matched-control-v3-20261017-01](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/matched-control-v3-20261017-01) |
| 38同题共同生成/审计 | [matched-control-v3-20261017-01](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/matched-control-v3-20261017-01) |
| Phase4 CPU rollout | [rollout-smoke-phase4-20261017-01](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/trajectories/rollout-smoke-phase4-20261017-01) |

## 附录 D：本次整理的证据身份与迁移要求

2026-10-18实际读取并核对以下文件，未重跑模型；哈希用于确认本文对应的已保存证据版本。

| 文件 | SHA256 |
| --- | --- |
| [artifacts/baselines/sft-v2-phase3/manifest.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/baselines/sft-v2-phase3/manifest.json) | `a40ef7a505c3f76bbdaf485106a588679bda02d8bab852e0d1a91c8aa9d70f0f` |
| [data/preferences/dpo_v3_matched_errors/experiment_plan.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/data/preferences/dpo_v3_matched_errors/experiment_plan.json) | `a3520bf40a59e07e058b0cb0804571e4d999b197c3ca36de0b110b5601465405` |
| [artifacts/checkpoints/matched-control-v3-20261017-01/summary.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/checkpoints/matched-control-v3-20261017-01/summary.json) | `b4d0c0e5d2d578855fc9569c1949525b9e7cf828ce1275f7fc358011847207a4` |
| [artifacts/evaluations/matched-control-v3-20261017-01/summary.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/matched-control-v3-20261017-01/summary.json) | `f405d5762e72b280c426e1253565d4d3c5f4d46c4dd8155c5af3b6d6ffe68522` |
| [artifacts/evaluations/matched-control-v3-20261017-01/verification.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/evaluations/matched-control-v3-20261017-01/verification.json) | `2da12a2c9c687cb004409bf16a5a6d6a7f0983223e2e8aa9ef6a4a3cc7e2d4c5` |
| [artifacts/trajectories/rollout-smoke-phase4-20261017-01/summary.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/trajectories/rollout-smoke-phase4-20261017-01/summary.json) | `0f2b3dc9409b8d3052953fa5d6278d86d75b0c5db13c17f50593cf3a41bbc500` |
| [artifacts/trajectories/rollout-smoke-phase4-20261017-01/verification.json](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/artifacts/trajectories/rollout-smoke-phase4-20261017-01/verification.json) | `ffa8a7cb6f2268669f27943d5b109ae58acc024d5ba73fbb6d2e7ed14baea6fa` |

关键本地提交：

| 提交 | 内容 |
| --- | --- |
| b7ec600 | Phase4 CPU rollout实现、12项新测试、137全套记录 |
| 8227414 | 38同题负结果报告和Phase3收尾 |
| 0bf89a7 | 38同题三臂训练/评测实现 |
| d15236a | 同题数据与有限方案冻结 |
| 7857262 | SFT训练真实错误采集 |
| 1c30598 | 四轮修复总汇与逐题结构拒收 |

迁移时至少保留：Git代码/配置/数据；冻结基线完整目录；固定revision的HF缓存（含其真实文件/链接目标）；要复核的训练adapter/tokenizer/summary/log；对应evaluation/trajectory原始记录、源码快照及清单。只复制adapter不能独立加载完整模型；只复制report不能执行审计。

本对话outputs是阅读导出副本，原始实验身份仍以项目内目录为准。本文和原handoff可以一并上传给无本地权限的新对话；若只上传一份，优先本文。
