# AgenticPostTrainingLab 完整交接：截至 2026-10-27

本文面向接手本项目的新对话，可单独阅读。依据最初的 `PROJECT_HANDOFF.md`、用户此前提供的 `PROJECT_HANDOFF_2026-10-18.md`、当前源码/配置、阶段报告和已回传产物整理。它统一描述当前状态；旧报告中的“当前/下一步/尚未开始”只代表各自写作时间，不能覆盖本文的最新结论。

本次任务仅整理交接，不启动新训练、不租机、不改变实验门限。整理前 Git HEAD 为 `1637ef3b8a9df5f4d760e4132dcf8ce8f3b290f0`。最近全套测试记录为 2026-10-27 的 **327 passed、1 条已有 tiny PEFT 提示**；本次未重跑测试或模型。本文所列最新关键文件哈希与存在性已重新核对。

## 1. 接手摘要：先读这十条

1. 项目定位：面向算术工具调用的后训练学习实验，从手写损失到真实 SFT＋LoRA、DPO、vLLM rollout、规则 reward 和有限 GRPO 更新。不是已经完成的通用 RL 框架或完整多轮 Agent。
2. 固定基础模型为 **HuggingFaceTB/SmolLM2-135M-Instruct**。135M 是已经实际训练的模型；0.5B 扩展仅讨论过，未实现、未运行。
3. Phase 2 真实 SFT 与 Phase 3 DPO/有限对照已完成。DPO 没有证明对共同新开发集有额外收益；保留负结果，不能为了“必须赢”重新开启无界调参。
4. **Phase 4 已完成工程验收**：云 GPU、vLLM 原生 LoRA、随机 rollout、严格 reward、同路径训练概率、TIS 近似校正、审计与有限吞吐测试。旧跨引擎数值门失败仍保留，工程收尾不是宣称所有概率路径一致。
5. **当前在 Phase 5，尚未全部完成。** 已完成 144 题×8 回答的信号准备，以及从冻结 SFT 出发的首次真实 GRPO 更新。
6. 首轮真实更新使用 62 个混合奖励组、496 条回答，候选轨迹 16 个 optimizer 步。保存恢复、冻结 base/reference、最终重载通过；开发严格成功 **36/56→36/56**，没有证明严格收益，没有晋升基线。
7. 尚未从更新后的候选重新采样；没有多次刷新 rollout 的在线策略循环。下一条主线是候选版本治理、新 rollout/new old-policy、有限后续更新与相同开发集比较。
8. 微批并行实现与 L20 对照已完成：微批 8 比串行快约 **6.46 倍**，但 token logp 差超过预设 `1e-5`；其余损失/梯度/参数/Adam 门通过。按原契约仍推荐串行 1，未自动用于正式训练。
9. 完整工具 Agent 尚未实现：当前到“模型输出请求→解析→计算器执行→评分”为止，没有 observation 回传和模型最终答复的多轮循环。SGLang、TRL、分布式训练也未接入。
10. 用户刚读到并理解了 `batched_scores.py`、`accumulate_microbatches`、`guarded_microbatch_step`；本次只要求交接。若之后说“进入下一步”，优先续 Phase 5 主线，不重跑旧实验，也不默认必须先解决并行误差。

## 2. 路径、环境与协作约定

### 2.1 路径

```text
本地项目根目录：
C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab

本地项目 Python：
C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab\.venv\Scripts\python.exe

当前对话工作目录（不是项目目录）：
C:\Users\app\Documents\Codex\2026-10-18\new-chat

当前交付目录：
C:\Users\app\Documents\Codex\2026-10-18\new-chat\outputs

此前用户提供的 handoff：
C:\Users\app\Documents\Codex\2026-10-08\x20\outputs\PROJECT_HANDOFF_2026-10-18.md

云端项目：/root/rivermind-data/AgenticPostTrainingLab
云端 conda 环境：/root/rivermind-data/envs/apt-vllm
云端上传目录：/root/rivermind-data
```

本文后续相对路径均相对本地或云端项目根目录。新对话的默认 cwd 可能不同，执行命令必须显式指定项目目录。

### 2.2 两套独立环境

| 项目 | 本地既有环境 | 云端已验证环境 |
| --- | --- | --- |
| 平台 | Windows，CPU | 智川云 Linux/Ubuntu22 镜像 |
| Python | 3.11.5 | 独立 conda Python 3.11；镜像 base 原为 3.10 |
| torch | 2.14.0+cpu | 2.8.0+cu128 |
| transformers | 4.57.6 | 4.57.6 |
| peft | 0.17.1 | 0.17.1 |
| accelerate | 1.15.0 | 安装日志记录 1.15.0 |
| vLLM | 未要求本地安装 | 0.11.0 |
| GPU | 无本地 NVIDIA GPU | 最近实验 NVIDIA L20，约 48GB 标称显存 |

历史迁移顺序是 RTX4090D→L40S→L20；最近训练与微批对照均为 L20，历史时价 2.18 元/小时。云端驱动记录为 580.119.02，`nvidia-smi` 显示 CUDA 13.0；这不等于 PyTorch wheel 使用 CUDA13，实装为 cu128。

镜像是 Miniconda `26.3.2-CUDA12.6-Devel-Py3.10-Ubuntu22`。后续在数据盘创建 `apt-vllm` 环境，不是直接使用镜像 base。换卡/克隆后先验证 GPU、环境和产物，不能假设不同 GPU 的数值记录仍能满足同路径门。现有部分入口明确绑定相同 GPU 型号及依赖版本。

**不要在云端直接执行 README 中的 `pip install -e .`。** `pyproject.toml` 的本地基础依赖是 torch>=2.14,<2.15，会与已验证的 vLLM0.11/torch2.8 云环境冲突。云端脚本已通过项目 `src` 路径导入。不要为了消除本地 Pylance 的 vLLM 导入提示，把 CUDA/vLLM 栈装进 CPU 环境。

本地曾有 PowerShell 激活失败、裸 python 指向3.8的情况，直接使用项目 `.venv\Scripts\python.exe` 最稳妥。不复用用户另一个 MiniServe 项目的环境。

### 2.3 用户偏好与授权状态

- 全部新 Markdown 用中文。用户通常逐段读代码、问原理，再说“进入下一步”；说“简要”时压缩解释。
- **每次 report 都要写代码阅读顺序、关键函数和阅读重点。** 不只列文件名。
- 用户曾要求一口气完成 Phase 4，也希望避免无限审计阻塞主线；有限负结果允许收尾，不要求算法必须胜出。
- 已允许本地代码实现、测试、报告与逻辑步骤的 Git 提交。当前 `main`，无远程；不自动推送或创建远程仓库。
- 用户已自行租云 GPU 并运行命令。最初总预算200元，后来明确“预算不是问题”；新 runner 已取消累计200元阻断，但保留单作业约30分钟/估算10元边界、无自动重试和失败保留。不要把旧200元约束误当当前硬上限，也不要将预算放宽理解为允许无界实验。
- 本对话没有可用的直接云端执行会话。实际协作方式是本地生成增量 ZIP→用户上传并在 JupyterLab Terminal 执行→下载结果 ZIP→本地复核。不要声称本地已经替用户跑过云 GPU。
- 最新实例是否开机、是否仍有库存、账户余额、剩余保留时间，本次没有实时核验。平台历史提示关机后7天释放数据、克隆限同算力区域；接手时以平台当时显示为准。关机不保 GPU，之后可能排队或开机失败。
- 用户以前的 MiniServe 推理/Agent 成果是背景，不是本仓库已经实现的功能。

## 3. 主体框架与数据流

```text
固定 base + tokenizer/chat template
  ├─ SFT 文本 → response-only CE → LoRA → 保存重载 → 严格工具请求评测
  ├─ chosen/rejected → 冻结 SFT reference → DPO/监督锚定 → 同条件生成对照
  └─ 冻结或版本化策略 → vLLM LoRA 随机采样
       → 原生 token IDs + q_sampler logp + finish_reason
       → JSON/schema/route/AST/执行结果 verifier → 二元 reward
       → 同 prompt 组内 advantage
       → HF FP32 trainer_old / trainer_new 与固定 SFT reference
       → token TIS + PPO clipped policy surrogate + sampled k3
       → LoRA 梯度累积/停止检查/AdamW → checkpoint → greedy 行为评估

尚未闭合：更新后的策略 → 新版本 rollout → 多轮有限训练循环
尚未实现：tool_call → 执行 → observation 回传 → 模型 final 的完整 Agent
```

核心目录：

| 路径 | 职责 |
| --- | --- |
| `src/agentic_post_training/algorithms/losses.py` | causal token logp、response CE、DPO、组 advantage、早期张量 GRPO |
| `src/agentic_post_training/algorithms/rollout_correction.py` | 实际使用的 detached token TIS 与校正策略项 |
| `src/agentic_post_training/data/` | tokenizer、SFT编码/打包、偏好/平衡数据、隔离审计 |
| `src/agentic_post_training/environments/calculator.py` | 有界 AST 算术解释与表达式结构比较；不使用 Python eval |
| `src/agentic_post_training/evaluation/` | 严格工具评分、贪心生成、括号/问法/偏好诊断和逐题比较 |
| `src/agentic_post_training/rewards/contract.py`、`audit.py` | 版本化严格 reward、重新评分、组 advantage 与零方差审计 |
| `src/agentic_post_training/rollouts/records.py` | RolloutSample 与分组打包；原生ID、EOS、mask、分数对齐 |
| `rollouts/hf_sampling.py`、`vllm_records.py`（同包内） | CPU采样与 vLLM 输出转统一记录 |
| `rollouts/trainer_scores.py` | HF原始FP32逐条评分、old/new无更新ratio检查、评分张量打包 |
| `rollouts/closeout.py`、`phase5_signal.py` | Phase4/5固定批次的记录验证与信号汇总 |
| `rollouts/batched_scores.py` | 多条原始回答一次完整前向评分，右padding与有效切片 |
| `trainers/sft_lora.py`、`dpo.py` | 真实 SFT/DPO 训练支持、模型参数身份 |
| `trainers/grpo.py` | 首次真实GRPO目标、串行梯度累积、LoRA检查、更新、checkpoint恢复 |
| `trainers/grpo_microbatch.py` | 新增并行微批实现；尚未获准替代正式串行路径 |
| `scripts/` | 可执行的训练、生成、审计、有限实验编排入口 |
| `configs/` | 版本化模型/采样/训练计划、固定case IDs、门限 |
| `data/` | 本项目 SFT/偏好/评测数据，不等同包内 data 模块 |
| `reports/` | 中文方案、实际结果、局限和代码阅读顺序 |
| `artifacts/` | baseline、adapter/optimizer、轨迹、预测、源码快照、摘要与审计 |

实际云端 trainer 是项目手写代码加 PyTorch/Transformers/PEFT，不是 TRL trainer；vLLM 负责生成，不负责反向训练。名称为 GRPO 的实现是明确声明的 token 平均 PPO/GRPO 风格近似目标，不能冒充其他框架默认目标的逐项复现。

SFT编码会核验chat template前缀、回答边界、终止符与长度；prompt不参与response CE，padding不参与损失，label的因果shift只做一次。正式SFT最大长度192；后续采样context512、最大回答64，不要混用两者。LoRA是参数高效更新方式，与SFT/DPO/GRPO训练目标正交，不是SFT之后单独再做一道“LoRA训练”。

## 4. Phase 安排与当前验收边界

| Phase | 主线 | 当前状态 | 剩余/限制 |
| --- | --- | --- | --- |
| 0 | 概念、独立环境 | 已推进 | 不是正式掌握度测验 |
| 1 | 手写 SFT/DPO/GRPO 张量损失 | 已实现、测试 | 张量函数本身不等于真实 trainer |
| 2 | tokenizer、SFT＋LoRA、工具评测 | 最小真实闭环与诊断完成 | 自然语言/顺序理解不稳，没有最终盲测 |
| 3 | 冻结基线、偏好数据、DPO、有限控制 | 已按计划收尾 | 无额外新开发收益，存在结构退化，不晋升基线 |
| 4 | GPU/引擎接入、rollout与训练接口 | 工程验收完成 | 保留跨引擎及调度logp差异；不强求SGLang/TRL |
| 5 | reward信号、真实GRPO、监控/恢复/评测 | 首次真实更新完成；微批对照完成 | 更新后新rollout与多轮策略循环尚未实现；尚无严格行为提升证据 |
| 6 | 完整工具Agent、轨迹评测 | 未开始 | observation、final、多轮状态、停止/恢复与轨迹成功 |
| 7 | 可复现总结、可选消融/扩模型 | 未开始 | 0.5B、LoRA rank/target 等须围绕明确问题设计 |

Phase 4 中的 CPU reward 与概率/TIS 审计是 Phase 5 的接口准备，仍归 Phase 4。2026-10-21 的 144×8 信号扩展才是进入 Phase 5 数据准备。不要把每个 reward 相关文件都机械划为 Phase 5，也不要因为最后微批门没通过而回退宣称 Phase 4 未完成。

## 5. 固定模型、数据、评测身份

### 5.1 模型和参数

```text
模型：HuggingFaceTB/SmolLM2-135M-Instruct
revision：12fd25f77366fa6b3b4b768ec3050bf629380bac
冻结SFT基线：artifacts/baselines/sft-v2-phase3
原SFT v2 run：artifacts/checkpoints/sft-lora-20261008-202445-a3d10354
LoRA：q_proj/v_proj，rank4，alpha8，dropout0，bias none
可训练参数：230400；120个LoRA参数张量

基线 manifest SHA256：
a40ef7a505c3f76bbdaf485106a588679bda02d8bab852e0d1a91c8aa9d70f0f
基线 adapter safetensors SHA256：
0efeb95765ece648571bb20c73fb7818eab3e6e3cd357812551275bd1823f094
原始SFT完整FP32参数摘要：
9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb
冻结base参数摘要：
e72fa38da3ed620a2f0879d8f11b0a6fe908e965f058b95b52738417bae1a4b6
首次GRPO最终候选完整参数摘要：
853c4027c91b968faadba67d46e3e9f2f239f107ab7bf8e58f276b0a643fe55b
```

文件 SHA256 与 `parameter_digest` 是不同身份：后者累计参数名、shape、dtype、CPU连续字节。reference 是完整 base＋冻结 SFT adapter，不是裸 base，也不是更大模型。候选与基线始终分别保存。

冻结清单记录74个快照文件和8个base缓存文件。基础模型未重复拷入baseline，实际缓存路径由 `manifest['base_cache_files'][0]['source']` 的父目录确定。只复制adapter无法独立加载模型。

### 5.2 数据谱系

| 数据 | train | development | 用途 |
| --- | --- | --- | --- |
| 中文/英语教学 examples | 3 / 4 | 2 / 2 | 编码教学、最小训练 |
| SFT expanded_v1 | 120=80工具+40final | 24=16+8 | 格式/路由与基本算术 |
| SFT expanded_v2 | 144=104工具+40final | 32=24+8 | 增加括号乘法，冻结SFT来源 |
| dpo_v1 | 28=24数学+4问候 | 10=8+2 | 真实错误和合成负例混合 |
| dpo_v2 大集 | 168=144数学+24问候 | 56=48+8 | 形式/问法平衡偏好；后续rollout输入来源 |
| dpo_v2 小集 | 28=24数学+4问候 | 同一56 | 完整family嵌套子集 |
| dpo_v3_matched_errors | 38个同题chosen，合成/真实两套rejected | 原56字节不变 | 三臂有限匹配对照 |
| Phase5 rollout | dpo_v2中的144数学输入×8 | 不采开发 | 1152条生成回答，不是1152个新题 |

SFT v2 与 dpo_v2 是不同数据。SFT v2 从base＋新LoRA训练；Phase3各分支及首次GRPO从冻结SFT继续。dpo_v2的144算术输入来自12个数字family，每族4种表达式结构×3种问法：symbolic/natural/sequence。开发4个数学family，共48数学题，另8问候。共享模板、相关family、数字范围外推，不能视为大量IID独立任务。

62个混合组是144个训练输入中的子集；48个“训练诊断题”也是既有训练split中的固定评测子集，不是另有48个独立新增训练题。优化选择、训练内诊断和开发评估三个用途要分开。

### 5.3 评测不是盲测

原工具20题、括号48题、空格控制12题、旧偏好开发10题、新偏好开发56题均已查看。原ID中含holdout不代表现在是盲测；括号诊断含已见表达式。五组共146记录但只有145个唯一输入，且存在family相关性，不能称146道独立最终测试。暂无新的未查看最终测试集。

## 6. Phase 2/3 历史实验与结论

### 6.1 SFT

SFT smoke分别3步/20步；expanded_v1训练240步，v2训练288步，均通过保存重载。正式SFT用CPU FP32/eager、batch1、lr0.001、AdamW weight_decay0、clip1、seed42、2线程；v1/v2各两遍固定顺序。

原20题严格通过：base0、20步9、v1/v2均19。base0表示不满足严格JSON协议，不代表没有数学能力。v2括号诊断严格33/48，数值33/48、成对11/24；空格控制12/12。v2相对v1有整体改善，但顺序问法数值8→4/12退步，不能声称稳定理解运算顺序。v1/v2同时变更数据、步数和开发集，不能把收益单独归因于数据数量。

### 6.2 首轮 DPO 与同题继续 SFT

28对、28步、lr1e-4、DPO beta0.1，从同SFT初始化policy/reference。工程冻结/更新/重载通过。DPO train/dev loss从0.693147降至0.465978/0.461117，但chosen绝对序列logp也下降，偏好改善不等于生成改善。

实际DPO使用回答有效token的序列logp求和（非按长度平均），目标为 `-log sigmoid(beta × [(logπ(chosen)-logπ(rejected))-(logπ_ref(chosen)-logπ_ref(rejected))])`。相对margin可以通过rejected下降更多而改善；日志中的DPO隐式reward是相对reference的logp差，不是后来算术verifier的0/1 reward。初始policy/reference相同，margin=0；代码accuracy用严格大于0，所以初始记0，而不是50%。

| 严格通过 | SFT | DPO | 同28题继续SFT |
| --- | --- | --- | --- |
| 工具20 | 19 | 20 | 19 |
| 括号48 | 33 | 33 | 33 |
| 空格12 | 12 | 11 | 12 |
| 旧偏好10 | 7 | 8 | 8 |

没有全面赢家；chosen-only控制是从相同SFT独立训练，不是在DPO上续训。相同曝光不代表相同计算量/梯度尺度。

### 6.3 dpo_v2 规模/步数 A/B/C

A=28条28步，B=同28条168步，C=168条168步；同lr1e-4、beta0.1、seed42、纯DPO。

| 严格通过 | SFT | A | B | C |
| --- | --- | --- | --- | --- |
| 新56 | 36 | 33 | 7 | 4 |
| 工具20 | 19 | 19 | 4 | 2 |
| 括号48 | 33 | 34 | 4 | 1 |
| 空格12 | 12 | 12 | 3 | 0 |
| 旧偏好10 | 7 | 5 | 1 | 1 |

C开发chosen平均logp -3.643747→-17.491135，rejected -9.124099→-41.487884；相对偏好改善却生成严重退化。有限核验未发现明显shift/mask/冻结/重载错误，不是代码正确性的数学证明。

### 6.4 监督锚定与降低 LR 的四轮有限修复

明确使用混合目标 `L_DPO + alpha × chosen_response_mean_NLL`，不能把收益都归给纯DPO。四轮同168题/168步、beta0.1：

| alpha / LR | 新56严格 | 新56语义/路由 | 工具20 | 括号48 | 空格12 | 旧偏好10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / 1e-4 | 31 | 31 | 19 | 29 | 12 | 5 |
| 5 / 1e-4 | 36 | 36 | 19 | 34 | 12 | 6 |
| 5 / 2e-5 | 35 | 42 | 20 | 34 | 12 | 7 |
| 5 / 1e-5 | 36 | 44 | 20 | 34 | 12 | 7 |

严重退化缓解，但没获得新56严格净收益，且有逐题结构退步。必须逐题比较JSON/schema/route：总分持平、同题前后都错，也可能从合法请求变为坏JSON。修复候选没有替换基线。

### 6.5 真实错误与38同题三臂收尾

冻结SFT在168训练输入贪心生成：严格108、语义130。排除22条数值对但AST不同的回答，得到38真实错误（36算式、2坏JSON），natural29/symbolic7/sequence2；仅12条匹配原同题合成负例，26条不匹配。错误采集未使用开发输入。

冻结38同题chosen-only SFT、合成rejected DPO、真实rejected DPO，均38步、lr1e-5、同曝光顺序/seed，DPO beta0.1、无监督锚定。三臂工程通过；新56四模型均严格36/语义44。工具/括号：SFT19/33、继续SFT20/34、合成DPO19/33、真实DPO20/34。三个新模型均新增同题坏JSON，结构门失败；DPO没有相对继续SFT的额外新开发收益。

实际438次新生成含3条基线探针，584份保存记录重评分；不要称584次新生成/独立测试。按预先冻结条件结束Phase3，不追加调参、不晋升基线。不能推导DPO普遍无效，也不能把容量、数据量或负例覆盖某一项当作已证明的唯一原因。

## 7. Phase 4：reward、云端接入和概率契约的演进

### 7.1 CPU rollout 与 reward准备

从SFT对3道数学训练题+1道问候各采4条，另1条复现探针。16条严格8/语义9，全部EOS；三个算术组均严格零方差。CPU policy前向重算最大差约4.1962e-5，在当时CPU门内通过；不是GPU跨引擎验证。

随后固定 `arithmetic-strict-v1`，新增 `score_reward`、`audit_records` 和测试。reward准备仍是Phase4；文档中旧“reward尚未实现”已过时。

### 7.2 云端迁移与模型 smoke

最初云包 `phase4-cloud-bundle-20261019.zip`，`cloud_bundle_manifest.json`记录245个文件与source commit `d14c1331147924de61e41d6a29cfaf02e5d2e7b8`。`verify_cloud_bundle.py`只核验清单哈希、路径和baseline，不加载模型、不证明GPU推理成功。当前Windows项目根没有这份云端manifest，不能直接宣称本地重演了Linux整包检查。

HF/vLLM原生LoRA最小生成成功：同一54 prompt token、28 response token、严格通过，token完全相同。单序列完整前向重算门0.001通过：HF最大差0.000307098，vLLM0.000167917。只能说明这一条序列；不是所有随机回答通过，也不是训练准入。

### 7.3 4×4 vLLM失败门与有限诊断

L40S随机4×4：16回答、378 response token，严格8/语义10，3个算术组仍零方差。原0.001逐token门有10/16条失败，最大差 **0.023653745651245117**，不要使用早先截图误读值。

有限数值诊断选择4条已有回答，不重新生成。HF FP16 full重复评分与原值完全相同；HF自身full/cached最大差0.0207724571。将已经FP16舍入的权重转FP32计算后，与vLLM仍最大差0.0287253857；这不等于重新加载原始FP32 checkpoint，也没有修复全部差异。可认为精度/执行路径相关，未逐项隔离attention、LoRA kernel等原因。

### 7.4 sampler 与 trainer 分开

新训练契约用原始FP32 checkpoint、HF eager、use_cache=False、eval/dropout0、关闭TF32，基于原生ID完整teacher-forced评分。old冻结、new保留图，未更新时相同路径要求logp差≤1e-5；L40S实际old/new差0、ratio=1。sampler/trainer差仍0.0288867950。

此后定义token TIS，不将vLLM q直接当trainer old分母；也不将“重算old”误称为样本已变成trainer分布。旧失败门未改成通过，新契约允许在明确有偏近似与监控下推进工程。

### 7.5 L20 的48×4收尾

48训练数学输入×4=192主回答，另16+16调度对照，共224请求。192均EOS，5400 response token；严格97、语义123、坏JSON5；19全对、19全错、10混合，40条非零advantage回答。old/new差0，sampler/trainer最大差0.0760447979；TIS权重0.9580–1.0790，无cap触发。

同一vLLM串行/批量调度单次16条对照：14.15秒对3.94秒，约3.59倍。16对token全部相同，但raw logp原门仅3/16通过，最大差约0.02992344。不能称跨调度概率稳定复现，也不能把该速度当vLLM对HF的比较。

作业约109.18秒，按2.18元/时估计0.0661元。本地完整产物重审通过。Phase4据新契约完成工程验收，保留上述限制；没有在该阶段进行真实GRPO更新。

## 8. 当前 reward、概率、目标函数的精确定义

### 8.1 严格奖励

`score_output`先检查严格JSON（拒绝重复key、非标准常量），再检查精确字段schema、calculator路由、合法算术AST、执行结果和目标表达式结构。tool_call必须只有type/name/arguments，arguments仅expression；final必须只有type/answer且answer非空。

算术严格通过需要目标AST与执行数值都正确。空白/冗余括号不改变AST，可通过；交换运算顺序等数学等价但AST不同可能严格失败。数值比较容差1e-9。`semantic_pass`仅作诊断，不能用它替代训练reward。

reward=1 iff `expected_type==tool_call` 且 `finish_reason==eos` 且 `end_to_end_pass`；否则0。问候final不参与算术RL，eligible=false，即使严格评测通过，训练reward仍0。长度截断不得奖励。`strict_pass`字段本身来自评分，不检查EOS；训练reward和最新行为严格计数另外要求EOS。

没有输出修复、抽取JSON片段、部分分、judge模型或学习reward模型。旧rollout的score是多项评测字典，reward是依版本/eligible/结束原因映射出的训练标量，两者不能直接混同。

`audit_records`重建case/sample身份、重算评分并与保存值比较，校验组内目标、编号、EOS/mask，随后计算reward/advantage。重复回答保留，不能去重改变经验分布。CPU audit-only不重新解码验证/前向概率时会明确标识；更完整的采集器审计还会重解码/重建张量。审计通过只说明对应范围内的一致性，不等于模型有效。

### 8.2 原生token与四种分数

每条RolloutSample保留prompt/response原生ID、seed、group/case/index、policy/behavior logp、finish_reason、text/raw_text。禁止decode再tokenize当作原始轨迹。causal shift在`causal_token_log_probs`中做一次，response切片从prompt长度减1开始。

EOS是回答的有效token，PAD不是；按真实长度构造mask，不能用token_id!=pad判有效，因为EOS可以等于PAD。打包形状为[group,sample,response_token]，prompt另存；已对齐分数不能再次shift。

| 符号 | 含义 | 梯度 |
| --- | --- | --- |
| q | vLLM采样行为条件概率；本配置T=1无过滤时等于该引擎raw policy概率 | 固定 |
| p_old | 本轮采样策略对应的HF训练评分路径，更新前存储 | 固定 |
| p_new | 当前待优化policy、同训练路径、相同原生回答的概率 | 保留 |
| p_ref | 原始冻结SFT reference，跨策略迭代仍固定 | 固定 |

T=1、top_p=1、vLLM top_k=-1、无重复/存在/频率惩罚，max_new_tokens64、context512。历史CPU top_k=0与vLLM -1分别表示其接口的禁用值，不能混抄。vLLM FP16、max_num_seqs4、max_num_batched_tokens512、memory_utilization .25、eager、prefix cache关、chunked prefill开、raw_logprobs、原生LoRA。HF trainer是原始FP32，不是将FP16权重简单float()后的模型。

### 8.3 实际GRPO计算

同题G条回答，二元奖励R，population标准差（unbiased=False）：

\[
A_i=\begin{cases}(R_i-\bar R)/\sigma_R,&\sigma_R>10^{-8}\\0,&\text{否则}\end{cases}
\]

A是整条回答的优势，广播到回答所有有效token；没有逐token奖励定位。组内全对或全错都使该组相对策略项信号为0，不代表二者质量一样，也不代表额外reference惩罚必定为0。

对同一历史上的每个回答token：

\[
w_{i,t}=\operatorname{stopgrad}\left[\min\left(2,\exp(\log p_{old,i,t}-\log q_{i,t})\right)\right],
\quad r_{i,t}=\exp(\log p_{new,i,t}-\log p_{old,i,t})
\]

\[
L_{policy}=-\frac1N\sum_{i,t}w_{i,t}\min\big(r_{i,t}A_i,\operatorname{clip}(r_{i,t},0.8,1.2)A_i\big)
\]

\[
d_{i,t}=\log p_{ref,i,t}-\log p_{new,i,t},\quad
k_{3,i,t}=\exp(d_{i,t})-d_{i,t}-1,
\quad L=L_{policy}+0.02\frac1N\sum_{i,t}w_{i,t}k_{3,i,t}.
\]

N是当前有效batch的response token总数，含EOS、不含prompt/padding。实际k3使用`expm1(d)-d`，权重在log空间截上限后exp防溢出。old/q/ref/A均固定，梯度只经new到LoRA。

TIS只重新加权损失贡献，不重新采样；token权重不校正完整prefix分布与组联合分布，cap也引入偏差。sampled k3是在存储历史上计算的近似惩罚，不宣称是当前完整策略KL的无偏估计。早期`grpo_clipped_loss`不是当前实际trainer完整目标；实际走`rollout_correction.py`加`trainers/grpo.py`。

PPO clip约束目标激励，不把真实ratio硬限制在[0.8,1.2]。`ratio_clip_fraction`统计ratio超区间比例，不等于所有这些token均实际走到裁剪的最小分支。静态rollout reward和非零/负loss都不能证明训练发生或行为提升。

## 9. Phase 5 已完成：信号扩展与真实更新

### 9.1 144×8 信号批次

`collect_phase5_signal.py`在L20对全部144算术训练输入各采8条，保存全部1152条；不使用开发题，不按结果挑回答。新seed、64请求分块，不要把请求分块64与引擎并发4或组大小8混淆。

结果：32296 response token、全部EOS；严格572/1152、语义731、坏JSON46；31全对组、51全错组、62混合组；496条非零advantage回答。自然/顺序/符号严格分别108/384、159/384、305/384。语义比严格多159条，不因此修改reward。

同一批每组前4条有47混合组，全部8条62组，多15组。这是同批截取比较；不能把此前48×4与现在144×8的全部变化仅归因于组大小。old/new差0，sampler/trainer最大差0.1388685703，权重0.908509–1.148973，无cap触发。该采集本身没有反向/更新。

作业358.47秒，估算0.2171元；本地完整token/评分/张量重审通过。此批足以启动有边界真实实验，不继续盲目扩量。

### 9.2 首次真实更新计划

入口`train_grpo_first.py`，计划`configs/grpo_first_update.json`。绑定上述summary，选全部62混合组并披露选择规则；82个零方差组保留在原始数据，但不参与此次优化，不能把优化子集表现当原分布表现。

每有效batch最多4组×8条=32回答，最后2组=16回答，共16批，单遍496回答。正式更新逐回答前向/反向（microbatch1），按回答长度/有效batch总token加权梯度后只更新一次。LoRA eval/dropout0仍允许autograd；`eval()`不等于no_grad。

AdamW lr1e-5、weight_decay0、beta(reference系数).02、PPO clip.2、TIS cap2、grad clip1、seed20261027。先验证原始policy/reference与存储old分数≤1e-5，初始实际差0。每步若sampled k3>.05或ratio超区间比例>.5则清空梯度、不更新；非有限值或零梯度报错，不自动调参重试。检查是更新前，不是更新后的回滚。

### 9.3 checkpoint与实际结果

第1步保存adapter、完整Adam状态、步数、参数名顺序、CPU/CUDA RNG和配置/输入联合digest。先执行不中断第2步作为探针，再丢弃该分支，从第1步重载并重做第2步，比较参数/动量/损失，继续候选到第16步。因此：

- 候选逻辑步16；实际optimizer调用17，额外1次是丢弃探针。
- 候选496回答；实际backward528=496+32。
- 120/120 LoRA张量变化，最大元素变化0.0001601102；base冻结不变。
- Adam恢复下一步参数/状态误差均0；最终新加载adapter后8条logp误差0、8个greedy token检查全一致。
- 无提前停止；梯度范数0.18446–0.72163；sampled k3最高0.00147061；ratio超区间比例最高约1.75%。

完整作业309.83秒，估算0.1876元。结果不是任意断点/分布式作业通用恢复系统，只验证当前明确保存点和下一步重放。

### 9.4 行为结果与本地证据

固定FP32/eager、相同模板、greedy、最多64token，前后各48训练诊断+56开发，共208预测记录：

| 指标 | 更新前 | 更新后 |
| --- | --- | --- |
| 训练诊断严格 | 28/48 | 28/48 |
| 训练诊断语义 | 35/48 | 36/48 |
| 开发严格 | 36/56 | 36/56 |
| 开发语义 | 44/56 | 44/56 |
| 新增JSON/schema/route退化 | — | 0 |
| length截断 | 0 | 0 |

工程检查true、行为安全门true、baseline_promoted=false。安全门通过不等于收益达标；只多一条训练内语义正确，不足以宣称泛化提升。

本地复核46产物/26源码哈希，208预测重新解码评分、16步history及Adam状态。CPU只加载原base和原/step1/final adapter，独立重建完整/冻结参数摘要，确认LoRA变化与base不变；没有重跑GPU前向。云端reference运行状态、恢复探针数值依赖已绑定源码与日志，不能称本地独立重演GPU。

## 10. 最近工作：微批并行探索与结果

新增`batched_scores.py`与`grpo_microbatch.py`，保留原串行trainer。一次前向多条prompt+response，右padding、按长度attention mask；各回答损失按T_i/N加权，在微批内求和，只反向一次。外层先zero_grad，所有微批后停止检查、裁剪、一次Adam更新。

`benchmark_grpo_microbatch.py`固定首次GRPO第16步候选与Adam状态、原计划第一batch的32回答/915token，每个探针都恢复同起点。每模式一次正确性探针兼预热，三次轮换计时；总12个临时更新全部丢弃，新发布候选0、新采样0。

| 微批 | 前向/反向次数 | 耗时中位数 | 加速 | 峰值已分配显存 |
| --- | --- | --- | --- | --- |
| 1 | 32 | 3.4549秒 | 1× | 1.223GiB |
| 4 | 8 | 0.9473秒 | 3.65× | 1.647GiB |
| 8 | 4 | 0.5352秒 | 6.46× | 2.223GiB |

| 数值比较（对串行） | 门限 | 微批4 | 微批8 |
| --- | --- | --- | --- |
| 最大token logp差 | 1e-5 | 4.1485e-5，失败 | 5.7220e-5，失败 |
| loss差 | 1e-6 | 7.7649e-8 | 5.9084e-8 |
| 梯度最大绝对差 | 1e-6 | 2.2026e-7 | 2.5332e-7 |
| 梯度整体相对L2差 | 1e-4 | 2.0867e-5 | 2.5133e-5 |
| 更新后adapter最大差 | 1e-6 | 7.4506e-9 | 7.4506e-9 |
| Adam状态最大差 | 1e-7 | 2.1886e-8 | 2.5146e-8 |

唯一失败项是logp门，推荐仍为1。原阈值不追溯修改；串行不是绝对数学真值，超过该工程门不自动证明并行代码bug。批矩阵运算的分块/求和顺序可能带来浮点差异，但本实验没有定位到具体算子，不能把可能原因写成定论。PyTorch官方也说明批处理与逐条计算不保证逐位相同：<https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html>。

计时仅一次固定32回答的更新工作负载，包含目标/反向/clip/Adam，排除恢复与CPU快照；不是整个GRPO流程的6.46倍。显存包含常驻policy/reference，不等于nvidia-smi整卡占用。完整作业37.317秒，估算0.0226元。

本地核验45输出/30源码哈希、各门/计时/推荐/费用重算一致。原始梯度与临时权重未导出，本地未GPU重演；恢复原候选、reference/base不变是云端运行检查记录。当前保留可用并行代码，但尚未作为正式训练路径启用。

## 11. 尚未完成与下一步建议

### 11.1 紧接当前的主线

如果用户要求继续，设计并实现有限的下一策略迭代，先冻结范围、输入ID、采样次数/seed、更新步数、停止界、评测方案，然后交付云端增量包：

1. 从`grpo-first-20261027-01/final-checkpoint/adapter`加载当前候选，明确模型/adapter版本和参数摘要，不再默认把冻结SFT当rollout policy。
2. 用该候选生成新原生token与q，保存新run目录/manifest；不能把旧1152条继续称作新策略的on-policy样本。
3. 在实际训练路径重新冻结本轮p_old，验证未更新old/new ratio≈1；p_ref继续使用原始SFT，二者此时身份不同。
4. 按预先公开的混合组选择和有限计划更新，记录采样/优化/评估计数、损失/梯度/ratio/长度/结构错误，版本化checkpoint和输入绑定。
5. 比较原SFT、上一候选、新候选在同一开发集的真实生成；收益未出现可以收尾，不无界追加LR/步数。

现有`collect_phase5_signal.py`/`train_grpo_first.py`绑定原SFT和旧run，不是只改路径就能安全复用的通用多轮runner。下一步必须处理策略版本、old/ref分离、恢复起点和新输入digest；是否沿用Adam动量也应在新计划里明确。以上尚未实现，本文没有新建执行包或启动作业。

### 11.2 可选并行后续，不阻塞主线

可继续现有串行路径完成主实验。若专门推进并行，预先建立统一批形状/顺序/mask的old/new评分契约，并对不同长度、不同batch检查无更新ratio、梯度与恢复。必要时比较无padding/有padding、各层输出定位误差。不要简单把这次单batch结果视为可自动放宽门限的理由；也不要说旧1e-5是普适正确性标准。

### 11.3 其他明确未完成

- 未验证GRPO严格收益、多seed稳定性、最终盲测或广泛泛化；没有大规模数据/长训练。
- 完整工具Agent：observation/final、多轮状态、预算/停止、异常恢复、逐轨迹评测。
- 0.5B及更大模型、rank/target modules消融；只能称可选方向，不预设某个新模型已选定。
- SGLang、TRL、分布式训练、在线权重同步/热更新、服务化调度；不是当前验收必选依赖。
- token TIS不是完整序列/组分布校正；sampled k3不是完整当前策略KL的无偏估计。
- 当前checkpoint恢复验证不等于任意失败点自动续跑系统。

## 12. 关键产物与证据索引

路径均已按项目根核对。`artifacts`大部分被Git忽略，目录存在不代表远端备份已完成。

| 用途 | 项目内路径 |
| --- | --- |
| 原始冻结SFT | `artifacts/baselines/sft-v2-phase3` |
| Phase3最终三臂训练 | `artifacts/checkpoints/matched-control-v3-20261017-01` |
| Phase3最终三臂行为/审计 | `artifacts/evaluations/matched-control-v3-20261017-01` |
| CPU rollout | `artifacts/trajectories/rollout-smoke-phase4-20261017-01` |
| HF/vLLM smoke本地归档 | `artifacts/evaluations/cloud-model-smoke-20261020` |
| 随机vLLM4×4 | `artifacts/trajectories/rollout-vllm-phase4-20261020-01` |
| 有限数值路径诊断 | `artifacts/evaluations/rollout-numeric-diagnostic` |
| 训练侧无更新概率契约 | `artifacts/evaluations/trainer-probability-contract-v1` |
| TIS策略项审计 | `artifacts/evaluations/rollout-correction-contract-v1` |
| Phase4最终48×4 | `artifacts/trajectories/phase4-closeout-20261021-01` |
| Phase4回传本地审计 | `artifacts/evaluations/phase4-closeout-review` |
| Phase5全144×8 | `artifacts/trajectories/phase5-signal-20261021-01` |
| Phase5信号本地审计 | `artifacts/evaluations/phase5-signal-review` |
| 首次真实GRPO完整run | `artifacts/checkpoints/grpo-first-20261027-01` |
| 首次GRPO候选adapter | `artifacts/checkpoints/grpo-first-20261027-01/final-checkpoint/adapter` |
| 首次GRPO最终Adam/RNG | `artifacts/checkpoints/grpo-first-20261027-01/final-checkpoint/optimizer.pt` |
| 首次GRPO本地审计/权重重建 | `artifacts/evaluations/grpo-first-review` |
| 微批对照完整结果 | `artifacts/evaluations/grpo-microbatch-20261027-01` |
| 微批结果本地审计 | `artifacts/evaluations/grpo-microbatch-review/local_audit.json` |

常见文件：run_manifest绑定配置、模型/数据/源码/设备身份；rollouts.jsonl保存原始轨迹与case；trainer_scores.jsonl保存old/new/q及advantage等；response/trainer_score_batch.pt保存掩码和分数；summary汇总；history记录实际步；before/after_predictions保存行为证据；source目录留当时源码快照。取回ZIP不是只读summary就够，需复核关联文件。

四个重要summary文件SHA256：

```text
artifacts/trajectories/phase4-closeout-20261021-01/summary.json
a98feb977bb4bb6a1e57362fff01cda1882f098fa88f81066a5cb62736fb665a

artifacts/trajectories/phase5-signal-20261021-01/summary.json
272dc316970eca29fd9bbcd0beadfa196dbfa31a7eb257f29764b53004a42d52

artifacts/checkpoints/grpo-first-20261027-01/summary.json
1816932fd3df9a5aa6bb3e18c402cb4775378d50c78c2311c1910950194213ad

artifacts/evaluations/grpo-microbatch-20261027-01/summary.json
a0eb3f2dceb52e7ec7ad19057faa1885dd23645a17cf7dde9cc50decdfdebc19
```

以上包含当前微批结果，均为本次重新核对的文件身份。不要将summary哈希与ZIP哈希混用。

最新回传ZIP身份：

- `phase5-rollout-review-20261021.zip`：`c008077b529a70127268b449a9f8cc28950c20acd7aa1424255df8b45c12b071`。
- `grpo-first-review-20261027.zip`：`df068ba95ba4b03da76bd77e2bbff8bcbc32921c69f73e26e9ced389cf4e1e2f`。
- `grpo-microbatch-review-20261027.zip`：`38246d77d02756238658821369582c4e05c4ed10500fbabd0c112d0b007d7ff1`。

单序列replay最初只有终端摘要返回，后续4×4包携带前置replay资料；不要假定本地一定存在云端的原始`artifacts/evaluations/cloud-replay-smoke/replay.json`路径。本地trainer-probability最初仅取回summary，报告已注明未取回二进制batch；后续Phase4/5完整包的本地复核范围更完整。不同历史证据等级不能混写。

## 13. 运行、迁移与审计注意事项

### 13.1 本地只读检查

```powershell
Set-Location C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab
git status --short
git log -8 --oneline
& .\.venv\Scripts\python.exe scripts\freeze_sft_baseline.py --help
```

需要回归测试时使用 `python -m pytest tests -q`，不要裸跑pytest扫描整个仓库：baseline源码快照内也有历史tests，曾导致重复模块收集错误。文档修改无需重跑模型或全套测试；本次仅记录最近327项已通过结果。

### 13.2 云端通用工作流

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data/AgenticPostTrainingLab
nvidia-smi
python scripts/verify_cloud_bundle.py
```

这些检查不能代替模型加载/前向。真实实验入口需按新方案给定；旧运行默认输出目录已存在会拒绝覆盖，不应直接反复执行`train_grpo_first.py`或benchmark。

增量包统一包含顶层`AgenticPostTrainingLab/`，用户上传到数据盘根，再用`python -m zipfile -e 包名.zip .`解压。曾因旧ZIP未含新脚本出现找不到cloud_model_smoke.py，问题是包版本/目录，不是GPU故障。云端文件路径是Linux路径，本地`C:\...`不能直接供云端终端读取。

原始cloud manifest及后续run都绑定文件哈希。**既有云端实验源码尽量保留，后续入口/模块采用新文件与新配置**；若必须改变已有文件，应明确重建版本/清单和兼容规则，不能偷偷覆盖旧文件再改哈希假装同一实验。不要改冻结数据中training_started=false等历史状态字段，它们是冻结时刻记录。

输出ZIP放云端项目根，方便JupyterLab下载。文件夹通常没有下载按钮，实际.zip文件才可下载；可刷新文件浏览器并定位项目根。下载ZIP不等于运行脚本已经关机。平台自动关机时间需要用户按当时作业安排确认，脚本没有自动停机。

接收结果先验证ZIP路径/重复成员/完整性，再核验summary文件哈希、source快照/当前可信源码、数据/模型身份、原始记录与统计；读torch产物使用weights_only=True。明确区分规则重评分、读取保存概率、CPU权重重建与实际GPU前向/反向重演。

Git忽略`.venv`、`.cache`、checkpoint、轨迹、评测与baseline大部分内容，仅baseline manifest单独跟踪。新对话若在同机可读取上述目录；迁移到别机至少另带固定revision缓存真实文件、完整baseline、当前候选adapter/optimizer/tokenizer、源rollout及评分、源码快照/清单、预测/审计记录。只给这份MD不能运行或独立审计，只有Git也不够。

## 14. 代码阅读顺序与关键环节

用户已读过很多历史代码，不必从头重讲。若新对话需要建立全局理解，建议：

1. `evaluation/tool_calls.py` → `EvalCase`、`score_output`；再看`environments/calculator.py`：先理解任务协议、严格AST与数值通过区别。
2. `rewards/contract.py` → `score_reward`；`rewards/audit.py` → `audit_records`：eligible/EOS/reward、重算为何必要、零方差不等于全错。
3. `algorithms/losses.py` → `causal_token_log_probs`、`group_relative_advantages`：一次shift、response位置、population标准差；需要历史背景再读SFT/DPO目标。
4. `rollouts/records.py` → `RolloutSample`、`pack_rollout_groups`；`vllm_records.py`：原生ID、EOS、seed、mask及采样输出身份。
5. `rollouts/trainer_scores.py` → `score_response`、`no_update_metrics`、`pack_trainer_scores`：q与old/new分开，原始FP32同路径ratio检查。
6. `algorithms/rollout_correction.py` → `token_tis_weights`、`corrected_grpo_policy_loss`：两种比值、detach、cap、padding先清零、token分母与近似限制。
7. `configs/phase5_rollout_signal.json`、`scripts/collect_phase5_signal.py`和`rollouts/phase5_signal.py`：144×8、前4/全8统计、全部保留及生成/评分/审计阶段。
8. `configs/grpo_first_update.json`、`trainers/grpo.py` → `completion_objective`、`accumulate_minibatch`、`guarded_step`：62组、16步、reference惩罚、全batch token均值和真实更新。
9. `trainers/grpo.py` → checkpoint save/restore/compare；`scripts/train_grpo_first.py` → `context`、`train`、`compare_evaluations`：输入digest、恢复探针、工程/行为验收分开。
10. `rollouts/batched_scores.py` → `score_responses`；`trainers/grpo_microbatch.py` → `accumulate_microbatches`、`guarded_microbatch_step`：用户当前读到这里，重点是共享图一次反向和分母不变。
11. `configs/grpo_microbatch_benchmark.json`、`scripts/benchmark_grpo_microbatch.py` → `reset`、`numerical_gate`、`recommend`：同状态公平比较、只logp失败仍回退、临时更新不发布。
12. 对应测试：`tests/unit/test_reward_contract.py`、`test_trainer_probability_contract.py`、`test_rollout_correction.py`；`tests/integration/test_grpo_first_update.py`、`test_grpo_microbatch.py`；`tests/unit/test_grpo_first_comparison.py`、`test_grpo_microbatch_benchmark.py`。先看真实tiny LoRA更新/恢复、mask/不等长梯度，再看拒绝非法输入与回退。

上面包内简写路径均以`src/agentic_post_training/`为前缀。文件名实际为`audit_rewards.py`、`test_reward_contract.py`，用户早前聊天中的`audit_rawards.py`/`test_contract_reward.py`是称呼误差，不要去创建重复文件。

## 15. 报告与Git导航

优先阅读这些当前结果报告（均在`reports/`）：

| 文件 | 接手重点 |
| --- | --- |
| `phase3-matched-control-results.md` | Phase3为什么收尾，负结果、结构门与基线不晋升 |
| `phase4-cpu-reward-contract.md` | 严格reward版本与离线审计边界 |
| `phase4-rollout-numeric-diagnosis.md` | FP16 full/cached/FP32计算路径诊断的有限结论 |
| `phase4-trainer-probability-contract.md` | q与old/new分离、无更新ratio |
| `phase4-rollout-correction-contract.md` | TIS近似与旧跨引擎门保留 |
| `phase4-closeout-results.md` | 工程收尾、48×4与调度对照 |
| `phase5-rollout-signal-results.md` | 144×8信号与前4/全8比较 |
| `phase5-first-grpo-update.md` | 首轮真实更新计划/数学/恢复方案 |
| `phase5-first-grpo-results.md` | 实际16步、参数独立复核、行为不变 |
| `phase5-grpo-microbatch-preparation.md` | 微批实现、门限和计时设计 |
| `phase5-grpo-microbatch-results.md` | 6.46倍、唯一logp失败、继续串行 |

更早Phase2的tokenizer/LoRA/数据/工具/括号报告，Phase3的baseline/preferences/scaling/repair/matched报告均保留于同目录；原`PROJECT_HANDOFF_2026-10-18.md`有详细历史产物全索引，可按需追溯，不再作为当前待办列表。

| 提交 | 内容 |
| --- | --- |
| `b7ec600` | Phase4 CPU rollout |
| `8227414` | Phase3匹配对照负结果与收尾 |
| `305938a` | HF训练概率与无更新ratio |
| `9d0b965` | token TIS契约 |
| `1267b3f` / `4ceeb6f` | Phase4收尾实现 / 实际验收 |
| `34b194a` / `589ef05` | Phase5信号采集实现 / 实际验收 |
| `87ea30f` / `d0b6564` | 首次真实GRPO实现 / 实际结果复核 |
| `8b3f4a0` / `1637ef3` | 微批实现 / L20结果与失败门记录 |

## 16. 本次核对范围与给新对话的启动提示

本次重新读取原handoff、README、当前相关源码/配置及Phase3—5关键报告，Phase2历史结果结合旧handoff与冻结baseline清单整理；核验74个baseline快照和8个base缓存文件；再次核对Phase4收尾7产物/23源码、Phase5信号7/23、首次GRPO46/26、微批对照45/30的清单哈希与当前源码一致，并确认本文关键路径和身份。未重新训练、未生成、未重新计算GPU梯度；以前的行为审计与CPU权重重建记录按其原证据范围引用。

当前无新runner在本文整理过程中启动。用户云实例实时状态未知。所有作业费用是执行时长估计，不是完整账户账单。新对话不能把历史日志的success字段直接替代其应检查的原始证据。

可直接复制给新对话：

> 请先阅读本交接文件，项目根目录是 `C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab`。当前Phase4工程已完成，Phase5已完成一次SFT起点的真实GRPO有限更新；开发严格36/56不变、baseline未晋升。微批8加速6.46倍但logp门未通过，正式训练仍用串行。尚未从更新后的候选刷新rollout，下一主线是版本化新采样与有限下一策略迭代，reference继续固定原SFT。先检查现有代码和产物，不重跑旧目录、不改旧门、不把早期报告的待办当现状。新增报告使用中文，并写代码阅读顺序。云端通过增量ZIP和用户手动终端执行协作；本地无GPU，现有云环境不要pip install -e。继续遵循用户逐段读代码、再推进的节奏。

本文优先于旧handoff中关于当前状态的描述；原始实验文件、冻结清单与旧报告保持历史事实，不回写改造成“全部通过”。
