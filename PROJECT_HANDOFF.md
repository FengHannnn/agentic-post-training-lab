# AgenticPostTrainingLab：项目全景与对话交接文档

更新时间：2026-10-17。本文依据当前本地代码、配置、数据、训练 summary 和评测产物核对整理。面向接手本项目的另一个对话，尽量不依赖原聊天记录。

## 0. 接手前先读：当前状态与边界

项目根目录：`C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab`。

当前进度：Phase 0/1 基础与张量损失已推进；Phase 2 已跑通真实 tokenizer、CPU SFT＋LoRA、保存重载、工具请求评测和括号/问法诊断。Phase 3 的四步最小闭环已完成：冻结 v2 基线、偏好数据构造与审计、首轮 CPU DPO 训练与保存重载、SFT-only 与 SFT＋DPO 逐项行为比较。结果混合，有改善也有退步，尚未证明全面泛化或 DPO 优于继续 SFT。

Phase 3有限同题对照也已收尾，当前进入Phase 4：CPU随机rollout记录接口已验证，GPU环境与vLLM引擎待资源规划；GRPO仍没有真实训练更新。

一句话定位：这是从后训练目标函数出发，逐步走向工具调用模型的 SFT、偏好优化、可验证奖励强化学习与 Agent 评测的学习型实验项目；目前并不是完整的 RL 训练框架或生产级 Agent 系统。

必须避免的误读：

- DPO 已接入真实 CPU 训练，GRPO 仍只有张量损失与单元测试，没有真实更新闭环。
- 当前真实训练包含 SFT＋LoRA 和在 v2 adapter 上继续执行的 DPO＋LoRA。LoRA 是参数高效训练方式，不是 SFT 之后的第二道训练工序。
- vLLM、SGLang、TRL 尚未接入，不能说项目已经用它们完成 rollout 或训练。
- 工具评测闭环止于模型输出请求、解析、计算器执行与评分；尚未将工具结果传回模型生成最终答复。
- 20 条原工具评测题已被反复查看并用于指导数据扩充，虽然文件名/ID 有 holdout 含义，现在只能当开发诊断集。
- 48 条括号主诊断中含已见训练表达式探针，不是全部未见样本；12 条控制题也是诊断开发数据。
- 现有小样本高通过率、低开发 loss 不能证明全面泛化，也不能证明已学会稳定的自然语言运算顺序。
- 当前没有 NVIDIA GPU；真实运行均为本机 Windows CPU。大规模 GPU 实验需要后续另行规划云端环境和预算。

最新恢复结果：四轮监督/LR控制已完成，最后恢复新集严格36/数值44、工具20/20、括号34/48。严重退化显著缓解，但尚有结构退步，新增开发收益未达标，baseline未替换；见第9节及reports/phase3-dpo-repair.md。

最新数据检查点：2026-10-17已在v2的168条训练输入上采集冻结SFT，严格108、语义130，得到38对真实错误候选（36算式、2坏JSON），12个算术family；自然/符号/顺序候选29/7/2，问候0。22条数值正确但AST不同的回答已排除。随后已冻结data/preferences/dpo_v3_matched_errors的真实/合成负例与chosen-only同题数据及三组实验方案，新入口与执行均已完成，见下一段及reports/phase3-matched-control-results.md。原数据清单保留冻结当时状态。

最新执行：38同题三组已全部训练、生成与重审。新56严格36/语义44四模型相同；工具/括号为基线19/33、继续SFT20/34、合成DPO19/33、真实DPO20/34，空格12、旧偏好7均相同。三个新模型均在同题新增坏JSON，不通过结构门；两组DPO无额外新集收益。438次新生成含3基线探针，584记录重算、六组对照/30份changes一致；125项测试通过，冻结/更新/重载通过且误差0。训练artifacts/checkpoints/matched-control-v3-20261017-01，生成artifacts/evaluations/matched-control-v3-20261017-01；代码0bf89a7，详见reports/phase3-matched-control-results.md。按冻结停止条件收尾本轮，不重跑、不追加调参、不替换baseline。

Phase 4最新执行：已从冻结SFT对4个偏好训练输入各随机采样4次，另1次复现探针；原生token、policy/behavior logp、长度mask与EOS已核验，完整前向最大差4.1962e-5。16条严格8/语义9，4组严格结果均恒定、只有1组语义结果有差异；这是接口检查，不是性能提升结论。权重不变、无梯度、无训练、无开发采样；137项测试通过。产物artifacts/trajectories/rollout-smoke-phase4-20261017-01，详见reports/phase4-rollout-contract.md。vLLM/GPU仍未接入，未启用付费资源。

最新生成结果：v2规模对照A/B/C已全部完成，新56题严格通过36/33/7/4（SFT/A/B/C）；C单独加载的两题复现一致，未发现切换adapter造成的定点偏差。详见第9节及reports/phase3-dpo-scaling-behavior.md。

最新验证：2026-10-17 重跑 `pytest tests -q`，137 项通过、1 条已有 PEFT 提示；偏好数据审计及冻结基线校验通过，真实 DPO 的参数冻结、更新与保存重载检查通过。行为比较重新生成 180 次，原三组 80 条 SFT prediction 与历史文本全部一致；后续偏好分组诊断重现了训练后的汇总指标。chosen-only继续SFT已完成28步、冻结/更新/保存重载检查，新生成90题三分支对照完成。

## 1. 用户背景、目标与协作偏好

### 1.1 前一个项目 MiniServe

用户此前完成 MiniServe：<https://github.com/FengHannnn/Miniserve>。以下为用户提供的背景，本文没有重新审计 MiniServe 仓库。

原项目目录：`C:\Users\app\Documents\Codex\2026-09-07\ai\MiniServe`。

其基础技术栈为 Qwen2.5-0.5B-Instruct、PyTorch、Transformers、FastAPI，包含手写 prefill/decode、KV Cache、Prefix Cache、Paged KV 内存管理原型，以及 Tool Registry、calculator、BM25、带防护的单 Agent Loop、10 条 Agent 评测与 35 项测试。

用户报告的结果：手写 KV 相比无缓存平均延迟降低约 73.7%；215-token Prefix Cache hit 相比 miss 有 2.33× p50 加速；Paged KV 分配实验减少 70.31% 逻辑 KV 空间；原始 Agent 路由 90%，防护后系统路由/端到端通过 100%。限制为 CPU 本机结果、Paged KV 未接 attention kernel、Prefix Cache 只做精确 token 前缀、Agent 评测只有 10 条。

新项目目的是从偏推理 Infra 的 MiniServe 向后训练、Agent 与模型机制扩展，而不是继续重写同一套服务。用户担忧 Infra 岗位机会，对大模型架构、后训练、Agent 都感兴趣；本文不对岗位数量或就业行情作未经核实的判断。

用户希望技术栈尽量不重复。目前仍复用 PyTorch/Transformers 作为通用底座，但已更换模型家族并引入 PEFT/LoRA、训练目标、数据审计、可验证评测；后续考虑 TRL 与 rollout 引擎。不是完全零重合，也不应为了换技术栈而丢掉必要基础。

### 1.2 用户的学习方式

- 用户按 Phase 推进，通常先读代码、逐段提问、运行测试、贴出日志，再进入下一步。
- 所有新增 Markdown 文档应使用中文；代码标识符、库名称和模型训练数据可保留英语。
- 只有用户明确要求“简洁/简要”时才特别压缩解释；平时应解释变量、流程、数学含义和边界。
- 避免一口气替用户跨越多个学习阶段；创建代码时提供阅读顺序和运行方式。
- 用户无本地 NVIDIA GPU，不要直接安排本机 CUDA/bitsandbytes/vLLM GPU 命令。
- 不复用、不修改 MiniServe 的虚拟环境，不把新依赖装进旧项目。
- 后续外部费用、云实例、付费服务或较大实验需要先与用户确认，不应把“规划阶段”当成已获准租机。

## 2. 运行环境与隔离约定

### 2.1 路径与解释器

本项目解释器绝对路径：

```text
C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab\.venv\Scripts\python.exe
```

实际核对为 Python 3.11.5。创建该环境所用的本机 Python 路径为：

```text
C:\Users\app\AppData\Local\Programs\Python\Python311\python.exe
```

Windows 虚拟环境的 `Scripts` 保存 python、pip 与激活脚本。两个项目即使相对路径都叫 `.venv\Scripts\python.exe`，绝对路径不同，就是不同环境。

用户此前执行 Activate.ps1 遇到 PowerShell 执行策略限制，随后裸 `python --version` 显示 3.8.0；这代表当时没有激活成功，不代表 .venv 内不是 3.11。最稳妥做法是直接调用上面的项目解释器，无须改系统执行策略。

`.vscode/settings.json` 已设置：本项目 `.venv` 为默认解释器，终端尝试自动激活，Pylance extraPaths 包含 `src`。VS Code 若仍打开 MiniServe 文件夹，不能假定它使用新项目环境，应打开 AgenticPostTrainingLab 或显式选择本项目解释器。

当前 Codex 对话的默认工作目录可能是另一个 MiniServe 检出目录，而不是本项目。接手对话的文件操作和命令必须显式指定本项目路径。

### 2.2 已验证依赖快照

| 组件 | 本机已安装/实验记录版本 |
| --- | --- |
| Python | 3.11.5 |
| PyTorch | 2.14.0+cpu |
| Transformers | 4.57.6 |
| PEFT | 0.17.1 |
| Accelerate | 1.15.0 |
| pytest | 9.1.1 |

这是本机快照，不是对云端 CUDA 兼容组合的承诺。`pyproject.toml` 使用范围约束，不是完整依赖锁文件；换机器需要再次验证。

项目包名为 `agentic-post-training-lab`，导入名为 `agentic_post_training`，采用 `src` 布局；Python 范围 `>=3.11,<3.13`。基础依赖是 torch，`[tokenizer]` 增加 Transformers，`[train]` 增加 Transformers/PEFT/Accelerate。pytest 已单独安装，目前没有专门的测试依赖 extra。

构建后端是 hatchling；它是构建依赖，未必常驻运行时虚拟环境。当前包已可导入；若出现 `ModuleNotFoundError: agentic_post_training`，先核对解释器和 editable install，不要把它误判成测试代码错误。

缓存：项目 `.cache/huggingface`，已有 tokenizer 和基础模型权重；现有脚本可通过 `--local-files-only` 离线运行。先前安装训练依赖时禁用 pip 缓存以避免受限环境默认缓存写入问题，这与 HF 模型缓存不是同一回事。

2026-10-08 已初始化 Git，分支为 `main`。现有 Phase 1/Phase 2 与 Phase 3 基线冻结分为两次本地提交，可用 `git log --oneline` 查看；尚未配置远程仓库或推送。首次提交是在实验完成后整理当前文件，不是对历史每次实验的逐步提交记录。

`.gitignore` 排除了 `.venv`、`.cache`、checkpoint、evaluation、日志与部分原始数据目录，以及完整基线快照；仅允许跟踪 `artifacts/baselines/sft-v2-phase3/manifest.json`，记录文件哈希、条件与指标。adapter 和完整逐题产物未纳入 Git，发布/迁移必须另行备份或选择性保存。只分享此 Markdown 或克隆仓库不会携带权重。`.gitattributes` 统一文本文件使用 LF，避免 Windows 换行转换影响后续校验。

## 3. 主体框架与真实数据流

### 3.1 当前已经实现的流程

```text
SFT JSONL（system/prompt/response 文本）
  → load_examples：字段校验、重复 ID 检查
  → 数据审计：标签合法、参考结果一致、跨集隔离
  → HF chat template + tokenizer
  → encode_sft：完整序列、回答标签、掩码、长度/前缀检查
  → collate_sft：按真实长度右侧 padding
  → 预训练 Instruct 基础模型 + 新初始化 LoRA
  → HF response-only causal CE
  → backward + 梯度裁剪 + AdamW 更新 LoRA
  → 开发 loss / 单题生成 / adapter 保存
  → 新加载同版本 base + adapter，比较 logits
  → 独立题目生成 JSON
  → 严格解析/schema/route 校验
  → 受限 calculator 执行
  → AST/数值/成对诊断指标与逐题日志
```

已增加偏好样本构造 → 审计 → 固定 reference 分数 → DPO 更新 → 保存重载 → 同条件逐题行为比较。另已实现冻结SFT同prompt多回答的CPU随机rollout、原生token/概率记录与规则评分观察。尚未形成reward定义 → advantage → GRPO真实更新，也未接GPU引擎或工具结果回传 → 多步Agent。

### 3.2 目录与职责

下表路径均相对上述项目根目录，属于定位索引，不代表代码全部完成。

| 路径 | 职责/状态 |
| --- | --- |
| `src/agentic_post_training/algorithms/losses.py` | 已实现 SFT、DPO、GRPO 张量级目标函数 |
| `data/sft.py`（包内） | SFTExample、JSONL 读取、encode、collate |
| `data/demo_tokenizer.py`（包内） | 字符级教学 tokenizer，仅用于离线教学/测试 |
| `data/hf_tokenizer.py`（包内） | 真实 HF tokenizer 加载与接口适配 |
| `data/validation.py`（包内） | 标签目标一致性与精确数据隔离审计 |
| `trainers/sft_lora.py`（包内） | LoRA 挂载、split 检查、训练/开发 loss |
| `environments/calculator.py`（包内） | 有界算术 AST 解释执行，不用 Python eval |
| `evaluation/tool_calls.py`（包内） | EvalCase、严格 JSON/schema、路由和执行评分 |
| `evaluation/generation.py`（包内） | 评测共用的 CPU 贪心生成，保留原始/评分输出 |
| `evaluation/brackets.py`（包内） | 按问法/数字/结构与成对题汇总诊断 |
| `scripts/inspect_sft_data.py` | 教学 tokenizer 下的数据检查 |
| `scripts/inspect_hf_tokenizer.py` | 真实模板、回答边界、EOS、padding 检查 |
| `scripts/train_sft_lora.py` | CPU SFT 训练、保存、重载验证主入口 |
| `scripts/validate_sft_dataset.py` | 扩充数据审计入口 |
| `scripts/evaluate_tool_calls.py` | 同一 base 与指定 adapter 的 20 题比较入口 |
| `scripts/diagnose_brackets.py` | v1/v2 adapter 的括号与问法对照入口 |
| `configs/` | 20/240/288 步 CPU 训练配置 |
| 根目录 `data/examples/` | 中文教学数据与最初英语训练/开发示例 |
| 根目录 `data/training/` | v1/v2 SFT 数据、参考 targets、manifest |
| 根目录 `data/eval/` | 20 题工具诊断、48 题括号诊断、12 题控制 |
| `tests/unit/`、`tests/integration/` | 单元不变量与随机 tiny Llama 的 LoRA 集成测试 |
| `reports/` | 中文阶段说明及诊断解读 |
| `artifacts/checkpoints/` | 本地真实训练产物 |
| `artifacts/evaluations/` | 本地逐题预测、summary、中文报告 |
| 包内 `rollouts/records.py`、`rollouts/hf_sampling.py` | CPU随机采样、原始ID/两类logp、EOS长度mask与前向重算；不是vLLM |
| 包内 `rewards/`、`benchmarks/`；`docker/` | 当前为空目录，不能视作已实现功能 |

这里“包内 data”是 `src/agentic_post_training/data`，根目录 `data` 是数据文件，两者不要混淆。

## 4. Phase 安排：完成事实与未来建议分开

现存阶段文档明确覆盖 Phase 1、Phase 2。Phase 0 的概念学习主要在原对话中进行，没有独立完整 Phase 0 文档。以下 Phase 3 及之后是为接手续接整理的建议路线，不声称原来已经有完整文件或已实施；用户可调整次序和范围。

### Phase 0：概念与实验环境基础——已推进

已讨论 SFT/DPO/PPO/GRPO、group-relative advantage、为什么 GRPO 不需要 value model、verifier 与 reward、vLLM 的作用；已建立独立 Python 环境。

完成标准应是能把监督标签、偏好对、采样 reward 区分开，理解 policy/reference/old policy 的职责，以及训练与推理/rollout 的区别。只是已读材料/对话，不是掌握程度的正式考试。

### Phase 1：手写后训练损失——已实现并验证

学习和实现因果 shift、token/sequence log probability、response-only CE、sigmoid DPO、组内标准化 advantage、clipped GRPO 与可选 KL。`test_losses.py` 的 10 项测试已通过；使用 unittest.TestCase，但由 pytest 运行，没必要为形式而重写。

交付边界：张量函数，不含模型、采样、reward pipeline 或完整 trainer。

### Phase 2：真实 SFT＋LoRA 与行为评测——主要闭环完成，泛化改进仍进行中

已完成数据预处理、教学/真实 tokenizer 检查、真实 CPU 模型训练、adapter 保存重载、独立工具请求评分、v1/v2 数据扩充和问法诊断。

尚未收尾：稳定的措辞/空格/顺序鲁棒性，真正未查看的最终测试集，单变量消融和多随机种子证据。不要无限追求原 20 题 100%，也不要把阶段内诊断开发集当最终成绩。

### Phase 3：DPO 最小真实训练闭环——四步完成，结果与限制已记录

2026-10-08 用户已选择依次执行：冻结 v2 基线 → 小规模偏好数据构造与审计 → DPO 训练/保存/重载 → SFT-only 与 SFT＋DPO 逐项行为对比。第 1 步的快照目录为 `artifacts/baselines/sft-v2-phase3`，说明见 `reports/phase3-dpo-baseline.md`，创建/校验入口为 `scripts/freeze_sft_baseline.py`。快照复制 adapter、tokenizer、历史训练和评测记录、数据与相关实现，使用 SHA256 校验；基础模型权重只校验本地缓存、不重复复制。policy/reference 的初始化都固定为 base＋v2 SFT adapter。

第 2 步已完成 `data/preferences/dpo_v1/`：28 train（24 算术＋4 问候）、10 development（8 算术＋2 问候）；真实 v2 错误 rejected 分别为 13/3，合成负例分别为 15/7。chosen 为人工复核的模板标签。family 分组，现有 SFT/eval 的 ID、prompt、精确 AST 隔离通过；真实 tokenizer 的完整长度最大 90，未截断。train/dev 共用措辞模板，仍是开发数据，没有最终未查看测试。入口 `scripts/build_dpo_preferences.py --audit-only`，包内审计 `data/preferences.py`，说明 `reports/phase3-preference-data.md`。完整生成日志在 `artifacts/evaluations/dpo-preferences-v1/`，偏好 JSONL 自身也保留逐题 provenance。

第 3 步于 2026-10-09 完成：`trainers/dpo.py` 接入 Phase 1 的序列 log probability 与 sigmoid DPO，`scripts/train_dpo.py` 在独立 CPU policy/reference 上运行；reference 分数预先缓存并 detach，policy 仅更新 LoRA。配置 `configs/dpo_cpu.json`，28 步、lr=1e-4、beta=0.1。产物 `artifacts/checkpoints/dpo-lora-20261009-205045-21dd3596`，train/dev loss 0.693147→0.465978/0.461117，相对偏好 accuracy 25/28、9/10；初始平局按严格 >0 记为 0。reference 参数不变/无梯度、policy 冻结参数不变/可训练参数有更新、保存重载 logits_match=true、最大差0.0。chosen 平均 log probability 也下降，不能把相对偏好改善当生成正确率。说明 `reports/phase3-dpo-training.md`。首轮 PEFT 保存尝试远程 config 后回退成功；当前脚本已显式禁用 embedding 保存和设置离线模式，未重跑首轮训练。

第 4 步于 2026-10-10 完成：`scripts/compare_sft_dpo.py` 和 `evaluation/comparison.py` 在同一 base、两个命名 adapter、同一 tokenizer/模板、CPU 贪心 64-token 条件下各生成 90 题。结果目录 `artifacts/evaluations/sft-dpo-20261010-150134-fc7e5e9a`；说明 `reports/phase3-behavior-comparison.md`。严格通过 SFT→DPO：原工具题19/20→20/20；括号33/48→33/48（3改善、3退步），成对11/24→10/24；空格控制12/12→11/12；偏好开发生成7/10→8/10。数值变化与严格变化一致，两模型 JSON/schema/route 均合法，无生成上限命中。原三组 80 条 SFT 文本完全复现。原失败题 2*(6+5) 改善，但 3*41+12 在不同问法下退步；natural 主诊断仍6/12。仍是开发诊断，含已见 SFT 表达式探针、没有最终未查看测试，该轮当时尚无继续 SFT 对照；后续对照见第9节。该轮当时测试75项通过。

学什么：chosen/rejected 数据质量、序列 log probability、冻结参考模型、beta、隐式 reward 与偏好 margin，长度偏差与训练/评测隔离。

做什么：从模型真实错误和正确请求构造偏好对；先审计偏好对，再接真实模型或 TRL 的 DPO 流程；比较同一 SFT 起点的 SFT-only 与 SFT→DPO；保存、重载、复用行为评测。

必要防护：chosen 不能只是格式更好但运算错误；不能把最终测试题答案写进偏好训练；reference 不参与更新；不以 preference_accuracy 代替真实执行效果。

验收：训练可运行、参考冻结、策略有更新、保存恢复正确，有至少一组受控行为对照。TRL 的具体 API/版本在接入当时查官方文档并验证，当前未安装或锁定。

### Phase 4：GPU 环境与 rollout 引擎——CPU接口前置验证完成，GPU/引擎待推进

2026-10-17已完成不产生外部费用的前置步骤：冻结SFT的4题×4次随机采样，保存原生token、模型/采样两类logp，完整前向重算、长度mask、一次复现与无模型审计。没有训练，没有vLLM/GPU或付费资源。全套137测试通过；见reports/phase4-rollout-contract.md。下一子步需资源与环境方案，不能把CPU接口验证视作本阶段引擎验收完成。

学什么：vLLM 的批量采样/调度与训练侧 log probability 的接口；推理吞吐、并发、采样参数、权重/adapter 版本对齐。SGLang 可作为可选对照，先掌握一个引擎即可，不要求两套同时实现。

做什么：经用户确认后另建云端 Linux/CUDA 环境，选定资源与预算；先跑推理 smoke test，再从多 prompt、多 generation 收集 rollout，保存原始输出、采样参数、token 数和模型版本。

验收：能复现采样并追踪成本；确认与训练 tokenizer/模板/终止条件一致。vLLM 是推理与采样引擎，不直接替代 DPO/GRPO trainer。暂不承诺具体 GPU 型号、CUDA 版本、价格或本机 Windows 支持组合。

### Phase 5：可验证奖励与 GRPO 实验——未开始

学什么：每 prompt 的 generation group、rule-based verifier、reward 设计、零方差组、old/new/reference、importance ratio、clipping、KL、token mask 与长度归一化。

做什么：为严格 JSON、路由、可执行参数和正确结果设计明确 reward；先离线审计 reward，再形成采样→评分→advantage→更新闭环；保留 SFT、DPO、GRPO 可比较基线。

验收：reward 没有把坏结构/错误请求误记为成功；能监测零方差组比例、reward、KL、clip fraction、长度、实际执行通过率；不靠修复输出掩盖原始策略行为。

特别注意：Phase 1 的 grpo_clipped_loss 是教学变体，不能未经对齐就声称复现原始 GRPO 或某个库的默认实现。真正在线采样需组内多样性，现有贪心诊断生成不等于 GRPO rollout。

### Phase 6：完整工具 Agent 与轨迹评测——未开始

学什么：多轮 messages、tool observation、状态推进、停止条件、最大步数、错误恢复、轨迹级 reward 与工具安全边界。

做什么：工具执行后将结果回传模型并生成 final；加入失败工具请求和恢复场景；区分 raw model、系统防护后成功和全轨迹成功；再考虑第二种工具。

验收：保存完整可回放轨迹；能分别统计请求成功、最终答复正确、步数/延迟/成本，不将系统修复能力算成模型能力。

这是新项目尚未完成的部分，不能拿 MiniServe 已有 Agent Loop 直接充当本项目实现。

### Phase 7：总结与可选模型架构实验——未开始，可选分支

将实验、消融、限制和失败样本整理为可复现仓库/报告。若用户继续对架构感兴趣，再单列 attention/MLP、LoRA target、rank/alpha、不同模型规模或架构的控制实验；不要在同一次比较中同时更换模型、数据、采样与训练预算。

本阶段不是从零预训练大模型；规模升级应有明确问题和预算，而非为了项目名称更大。

## 5. 关键实现约定：接手时不能破坏的不变量

### 5.1 SFT 数据与 token 对齐

每行 JSONL 是一条样本，不是一个训练文件。`SFTExample` 四字段为非空字符串：`id/system/prompt/response`。response 是文本，包括工具调用 JSON 字符串，不是预先给定 token ID。ID 的 set 用于检测重复并报错，不是静默删除重复样本。

`encode_sft` 分别编码 system/user＋assistant 生成开头与完整回答，要求 prompt_ids 真的是完整 token 序列前缀。若模板不匹配就报错，不猜回答边界。完整长度超过 max_length 也报错，不截断 JSON 或结束标记。

| 区域 | attention_mask | labels | loss_mask |
| --- | --- | --- | --- |
| system/user/assistant 开头 | 1 | -100 | 0 |
| 回答内容及模板结束区域 | 1 | token ID | 1 |
| padding | 0 | -100 | 0 |

attention_mask 控制有效输入位置，不是“决定哪里预测”；loss_mask/labels 控制训练损失的目标选择。prompt 无直接目标 loss，但其上下文计算仍影响回答和梯度。

collate 按原始长度补齐，不根据 pad token ID 判定位置；EOS 可以被复用为 PAD，真实 EOS 仍需保留训练。DemoChatTokenizer 按 ord(character)+1 编码，只是教学工具，不能拿它替代预训练模型 tokenizer。

因果对齐为 logits[t] 预测 token[t+1]。预处理不 shift；HF 模型内部 shift 一次。Phase 1 的 gather 不接受 -100，手写 CE 必须用 input_ids＋loss_mask，而非 HF labels。测试验证 HF labels 路径与手写 mask 路径的 loss 一致。

### 5.2 LoRA 与真实 SFT 训练

真实流程：加载现成 Instruct base → 挂载新 LoRA → response-only SFT 训练 LoRA → 保存/重载。Phase 1 没有训练出一个基础模型；当前 HF 训练并没有直接调用 Phase 1 的 masked_cross_entropy，使用的是模型内部等价的 causal CE。

LoRA 形式为 `Wx + (alpha/r)BAx`；冻结 W，训练 A/B。当前 q_proj/v_proj，rank=4、alpha=8、dropout=0、bias=none；只有名字包含 lora_ 的参数允许训练。可训练 230,400，总参数 134,745,408，约 0.171%。少量参数不等于只运行小矩阵，基础模型仍参与前向和必要反向。

训练 CPU float32、eager attention。AdamW 学习率 0.001、weight_decay=0、max_grad_norm=1、batch=1、seed=42、CPU threads=2。固定顺序循环取样，不 shuffle；恒定学习率，无 scheduler、梯度累积、混合精度、量化、断点续训或 GPU device abstraction。

每步清梯度→模型 loss→backward→clip_grad_norm_→optimizer.step。裁剪原地修改 grad，AdamW 使用裁剪后的 grad；日志 grad_norm 是裁剪前范数。每步样本不同，训练 loss 不要求逐步单调下降。

开发 loss 逐样本算，再按实际有效预测 token 数加权；model.eval 与 no_grad 分别管理模式和梯度。训练关闭 cache，生成显式 use_cache=True。

每次训练从 base＋新初始化 adapter 开始。v2 不是在 v1 adapter 上续训。保存内容不是完整 optimizer checkpoint；重载 logits 一致证明保存恢复，不证明泛化。

### 5.3 DPO/GRPO 教学损失的边界

DPO 使用每条回答的序列 log probability，比较 policy 与 reference 的 chosen/rejected 相对差，sigmoid loss 批量平均。chosen_rewards/rejected_rewards 是隐式诊断量，不是训练一个 reward model。

reference/old probability 未来应由 trainer 冻结或 detach；当前 dpo_loss/grpo_clipped_loss 没有自动对所有输入停止梯度。现有 DPO 测试中的 reference 本来就不 requires_grad，不能据此说函数内部自动冻结了 reference。

GRPO advantage 在 `[batch,generations]` 内按 prompt 标准化，std 使用 unbiased=False；std<=eps 时全组 advantage=0。损失输入 `[batch,group,tokens]`，completion advantage 广播到 token。

ratio=`exp(new_log_prob-old_log_prob)`，取 `min(ratio*A, clip(ratio)*A)`；可选 KL 为 `exp(reference-new)-(reference-new)-1`。当前按整个 batch 的有效 token 总数平均，不是先对每 completion 平均再按样本平均；与某些 GRPO 定义的长度权重不同。

clip_fraction 统计 ratio 越界比例，不保证所有越界 token 的优化项都落在平坦截断区，尤其要区分 advantage 正负。reference 缺省时 KL 为 0；当前也未强制 beta>0 时一定提供 reference，未来 trainer 需要明确配置契约。

### 5.4 工具 schema、计算器与 verifier

当前支持两种精确 schema，不接受额外键：

```json
{"type":"tool_call","name":"calculator","arguments":{"expression":"2*(6+5)"}}
```

```json
{"type":"final","answer":"Hello!"}
```

系统提示约定 arithmetic 用 calculator，greeting 给 final；这不是任意开放域路由任务。

JSON 整段解析，拒绝重复键、NaN/Infinity 等非标准常量；不抽取 JSON 子串、不去 Markdown 围栏、不自动修复。模型 raw_prediction 保留特殊 token，prediction 只通过 tokenizer 去特殊 token。

calculate 只解释 AST 的有限 int/float、+/-/*//、括号和一元正负，拒绝 bool、名称、调用、幂等；表达式最长 128 字符、AST 最多 64 节点、每步结果有限且绝对值不超过 1e12；除零/溢出报错。不是执行任意 Python。

评测字段：json_valid、schema_valid、route_correct、arguments_valid、arguments_correct、tool_execution_success、result_correct、end_to_end_pass、tool_result、failure。参数/执行/结果指标只用期望工具题作分母，JSON/schema/route/闭环用所有题。

严格 arguments_correct 比较 `ast.dump`，忽略空白和冗余括号，但保留操作数与运算顺序；它不是“变量抽象后的同一题型”，AST 含具体数字。交换乘法/加法可能数值正确但严格不通过。结果容差 rel_tol=abs_tol=1e-9。

final 题只查结构、非空 answer、路由；不评分回答语义。这里 end_to_end_pass 的名称只是请求执行闭环，不是完整 Agent 最终答复。

现有 score_output 可以成为以后 rule-based verifier 的基础，但目前没有显式 reward 映射/GRPO reward pipeline。不要把 bool 指标自动等同于已设计好的 RL reward。

### 5.5 审计与诊断指标

validate_dataset 检查 split 内重复输入、训练/开发/受保护题 ID 与 prompt 重叠、targets 一一覆盖、标准响应通过评分、跨 split 精确表达式 AST 重叠。

不检查所有数学等价题、不证明自然语言标签语义一定正确，也没有在训练主脚本内部自动调用完整审计。train_sft_lora 只执行 check_split/encode 等基础检查；扩充数据应先单独运行 validate_sft_dataset。

诊断的 semantic_pass=结构合法＋路由正确＋结果正确，只验证当前数字结果，不是符号等价证明；输出常数也可能在个别题上数值巧合正确。pair_semantic_pass 要求括号/无括号两题都正确，并选择结果不同的成对题，防止始终输出同一结构却偶然过单题。

诊断逐题保存 seen_training_prompt/seen_training_expression。seen_large 与 unseen_small 同时涉及数字范围和训练覆盖，不能将结果差异单独归因于“大数/小数”。

## 6. 数据版本与使用方式

| 数据集 | train | dev | 用途 |
| --- | --- | --- | --- |
| 中文 examples | 3 | 2 | 教学预处理/编码，不是正式中文能力实验 |
| 英语 examples | 4 | 2 | 3/20 步最小闭环 |
| expanded_v1 | 120：80 tool＋40 final | 24：16 tool＋8 final | 格式/路由/基础算术扩充 |
| expanded_v2 | 144：104 tool＋40 final | 32：24 tool＋8 final | v1 加括号乘法 |

v1 数据为确定性人工模板，未用另一个模型生成标签；覆盖加减乘除、括号后除/乘、负数、小数及问候/礼貌路由反例。按每两个工具样本后一个 final 排列。train/dev 共用模板、操作数不同，不是严格模板外泛化。

v2 保留全部 v1，新增 24 train＋8 dev，覆盖 `a*(b+c)`、`a*(b-c)`、`(b+c)*a`、`(b-c)*a`；每五条旧样本插一条新增训练题。未把原失败题 `2*(6+5)` 的精确表达式放入 train/dev。

`expanded_v*_targets.jsonl` 是审计路由、参考表达式/数值、类别的侧文件，不是额外模型输入；manifest 记录来源、数量、限制。JSONL 已生成并保存，当前没有单独的数据生成器脚本；若新增 v3，需另做可复现生成/审计方式。

原工具诊断 `data/eval/tool_calls_v1.jsonl`：20 条，16 tool＋4 final。括号主诊断 48 条=3 数字组×4 问法×4 结构；控制题 12 条=3 数字组×4 结构。

本次重跑 v2 audit：train 最大 89 token、dev 最大 90 token，最少回答区域 12 token，均低于配置 max_length=192。这证明本版本未因长度被拒绝，不代表其他 tokenizer 或未来数据也如此。

## 7. 已有真实实验与准确产物索引

以下路径均位于本项目，目录保留，不覆盖。

### 7.1 四次 SFT 训练

| 实验 | train/dev | 步数 | dev loss 前→后 | checkpoint 目录名 |
| --- | --- | --- | --- | --- |
| 最小 smoke | 4/2 | 3 | 2.163145→1.902697 | sft-lora-20261008-002302-e5bfa695 |
| 最小加步数 | 4/2 | 20 | 2.163145→0.251388 | sft-lora-20261008-005426-9d8f2036 |
| 扩充 v1 | 120/24 | 240 | 2.286667→0.036035 | sft-lora-20261008-180828-05ced0f0 |
| 扩充 v2 | 144/32 | 288 | 2.193550→0.029906 | sft-lora-20261008-202445-a3d10354 |

所有 checkpoint 在 `artifacts/checkpoints/`；四次保存重载均记录 logits_match=true、最大绝对差 0.0。浮点一致性来自这台机器的记录，不承诺不同硬件也零误差。

基础模型固定实际提交 `12fd25f77366fa6b3b4b768ec3050bf629380bac`，模型为 `HuggingFaceTB/SmolLM2-135M-Instruct`。小模型主要面向英语，所以正式训练用英语，不代表中文工具能力。

`configs/sft_lora_cpu.json` 当前 20 步、revision=main；3 步配置是历史记录，不是现在默认。expanded_v1/v2 配置固定提交、max_new_tokens=64，分别 240/288 步。batch=1 时各扩充集遍历两轮。

不同扩充版本 dev 集不同、数据量和步数都改变，不能把 loss 或行为差异全部归因于单一因素。v2 不一定处处优于 v1，两者都应保留作基线。

每次 checkpoint 含 adapter/、tokenizer/、train_log.jsonl、summary.json。summary 包含配置、实际 revision、依赖、参数量、训练/开发数量、前后 loss、单题生成、重载误差、完成步数。基础模型另从缓存加载，adapter 不能单独当完整模型。

### 7.2 原 20 题工具请求评测

| 模型 | JSON | schema/route | 算术数值/严格参数 | 请求执行闭环 |
| --- | --- | --- | --- | --- |
| base | 0/20 | 0/20 | 0/16 | 0/20 |
| 20-step LoRA | 14/20 | 9/20 | 7/16 | 9/20 |
| expanded_v1 LoRA | 20/20 | 20/20 | 15/16 | 19/20 |
| expanded_v2 LoRA | 20/20 | 20/20 | 15/16 | 19/20 |

base 0/20 是本任务严格 JSON 协议下的失败，不等于基础模型没有数学能力；它常输出自然语言，不能通过整段 JSON 解析。v1/v2 都有 16/16 可执行算术请求，但有一个表达式错误。

失败 holdout-012，题目 `Calculate 2 * (6 + 5).`，参考 `2*(6+5)`→22：v1 输出 `(2+5)`→7，v2 输出 `2*6+5`→17。v2 变化了错误形式，但仍然错，不能说原题已修好。

报告目录：

- 20-step：`artifacts/evaluations/tool-calls-20261008-141249-451b47ce/`。
- v1：`artifacts/evaluations/tool-calls-20261008-200500-59dfa859/`。
- v2：`artifacts/evaluations/tool-calls-20261008-203842-cc578a24/`。

每个含 base_predictions.jsonl、lora_predictions.jsonl、summary.json、report.md。3-step adapter 没有对应已保存的 20 题评测，不要杜撰其准确率。

### 7.3 最近的括号与文字问法诊断

只比较 v1/v2 adapter，使用同一 base commit、tokenizer/模板，CPU 贪心生成，最多 64 新 token。挂载两个命名 adapter、切换评测，没有更新权重。

数字组 original=(2,6,5)、unseen_small=(3,7,4)、seen_large=(3,41,12)。四结构 right_grouped=`a*(b+c)`、right_plain=`a*b+c`、left_grouped=`(b+c)*a`、left_plain=`b+c*a`。

| 主诊断指标 | v1 | v2 |
| --- | --- | --- |
| 严格 AST 闭环通过 | 28/48 | 33/48 |
| 当前数字结果正确 | 30/48 | 33/48 |
| 成对两题结果均正确 | 9/24 | 11/24 |

| 问法 | v1 数值正确 | v2 数值正确 |
| --- | --- | --- |
| spaced：Calculate＋带空格算式 | 10/12 | 11/12 |
| compact：Compute＋紧凑算式 | 9/12 | 12/12 |
| sequence：明确运算步骤 | 8/12 | 4/12 |
| natural：sum/product 自然语言 | 3/12 | 6/12 |
| 额外控制：Calculate＋紧凑算式 | 9/12 | 12/12 |

主实验 spaced/compact 同时改空格和前缀，有混杂；因此另加 12 条 compact_sameverb 控制。v2 中 `Calculate 2*(6+5).` 与 `Compute 2*(6+5).` 都正确，而 `Calculate 2 * (6 + 5).` 错；这提供空格敏感性的直接对照，不需要把成功归因于 Compute。

按结构：right_grouped 数值正确 v1=1/12、v2=9/12；left_grouped 两版12/12；right_plain 9/12→6/12；left_plain 8/12→6/12。增强部分括号题同时伴随部分无括号/措辞题退步。

例如 `Multiply 2 by 6 first, then add 5.`：v1 生成 `2*6+5` 正确，v2 仅 `2*6`。另一些自然语言题遗漏操作数。证据支持模板/措辞敏感、运算顺序理解不稳定这一观察；不能由这一次小样本诊断证明容量不足、过拟合根因或训练量不足。

主报告：`artifacts/evaluations/bracket-diagnostic-20261008-204902-76e6b53b/`。

控制报告：`artifacts/evaluations/bracket-diagnostic-20261008-210057-380ef02a/`。

含 v1_predictions.jsonl、v2_predictions.jsonl、summary.json、report.md，summary 记录数据 SHA256。详细设计/解读在 `reports/phase2-bracket-diagnostic.md`。共 60 题/adapter、120 次生成；目前没有多 seed 或采样式鲁棒性实验。

## 8. 已完成与未完成清单

### 已完成并有代码/产物

- [x] 独立 Python 3.11 环境、src 包、VS Code 解释器配置。
- [x] SFT/DPO/GRPO 教学损失与 10 项基础测试。
- [x] SFT JSONL、回答边界、labels/loss_mask、长度安全与 padding。
- [x] 真实 tokenizer/template/EOS 检查。
- [x] CPU LoRA SFT、开发 loss、生成日志、四次真实训练。
- [x] 只训练 adapter 的验证、保存重载一致性。
- [x] 受限计算器与严格请求评分。
- [x] 20 题 base/LoRA 对照、v1/v2 数据和审计。
- [x] 48 主诊断＋12 单变量控制，逐题报告。
- [x] 当前 78 项测试通过。

### 当前明确未完成

- [ ] v3 多措辞/空格/运算顺序训练数据与新的受控 SFT 实验。
- [ ] 从未查看、未用于调参的最终测试集；更广泛任务/语言泛化。
- [ ] 固定数据/步数/预算的消融，多 seed，LoRA rank/target 对照。
- [x] 小规模 chosen/rejected 偏好数据、来源记录、隔离与真实 token 审计。
- [x] DPO trainer、首轮 CPU 真实更新与保存重载验证。
- [x] SFT-only 与 SFT＋DPO 逐项行为对比与改善/退步记录。
- [ ] 组采样、显式 reward、reference/old policy 管理和完整 GRPO trainer。
- [ ] TRL 集成、vLLM/SGLang rollout、GPU/云端环境。
- [ ] 工具结果回传、多轮 Agent、错误恢复、轨迹 reward。
- [ ] 推理服务、第二种工具、RAG、多 Agent（均不属于目前实现）。
- [ ] 通用 device/批处理生成迁移；当前 helper 创建 CPU tensor。
- [ ] 训练断点续跑、optimizer/scheduler 状态、完整配置/数据锁定。
- [x] Git 初始化，Phase 1/Phase 2 与 Phase 3 基线冻结的两次本地提交。
- [ ] 远程仓库、CI、云端 Docker、权重/评测产物备份发布。
- [ ] 新项目的部署性能 benchmark 或岗位成果包装。

## 9. 阶段执行记录与下一步边界

下文保留各时点的诊断与下一步建议。当前以最新完成段、顶部状态和Phase 4报告为准，不要把早期建议当作尚未执行而重跑Phase 3。

用户已选择进入 Phase 3，四步最小闭环已完成。读完行为比较代码后，2026-10-10 已按受控改进路线完成偏好概率分组诊断：入口 `scripts/diagnose_dpo_preferences.py`，汇总 `evaluation/preference_diagnostics.py`，说明 `reports/phase3-preference-diagnostic.md`，产物 `artifacts/evaluations/dpo-preference-diagnostic-20261010-181453-22baa0cd`。重新评分38对数据，两模型共76次前向，复现训练后汇总；train25对相对改善中9对chosen下降，dev9对中4对chosen下降。括号题chosen平均增加，无括号题平均下降；left_plain训练仅3/6相对改善，开发两题生成均错。来源与题型混杂，不能因果归因于真实/合成来源。2026-10-11已完成同一v2起点的chosen-only继续SFT对照，控制28个chosen的曝光、28次更新、seed与学习率，逐步匹配DPO样本顺序。入口`scripts/train_sft_control.py`、`scripts/compare_sft_control.py`；说明`reports/phase3-sft-control.md`；训练目录`artifacts/checkpoints/continued-sft-20261011-181322-52e2bc03`，评测目录`artifacts/evaluations/sft-control-20261011-181518-0ef09401`。train/dev回答token加权CE 0.102634/0.096143→0.041592/0.044478；训练34.84秒、输入2277token、监督回答714token。仅LoRA更新、冻结参数不变、重载logits最大差0.0。原SFT/DPO预测经身份/数据/汇总核对后复用，新模型生成90题。严格通过继续SFT为19/20、33/48、12/12、8/10；括号数值34/48、成对12/24。括号组相对原SFT3改善3退步，且出现一条invalid JSON；两种方法收益分布不同，没有全面胜出。继续SFT对照代码已读完；2026-10-15进入均衡数据扩充，见下段。不能把固定28对数据结果当作样本规模的因果验证。后续DPO步数应按受控方案改变，不盲目调参；换大模型或租GPU需另行规划。

2026-10-15已构造并重审`data/preferences/dpo_v2/`：train168（144算术+24问候）、train_small28（24+4，完整family嵌套子集）、development56（48+8）。数字family为12/2/4，每个12条（4形式×3措辞），问候family6/1/2，每个4条；train/dev按family及精确ID/prompt/AST隔离，并保护444条现有SFT/eval/v1目标记录。全部负例为合成错误分组或错误路由，不是模型真实采样；chosen来自确定性模板。最长训练93、开发95token，无截断；真实tokenizer与评分器重审通过，81项测试通过。新增包内`data/balanced_preferences.py`、入口`scripts/build_balanced_preferences.py --audit-only`、说明`reports/phase3-balanced-preferences.md`。本步只构造数据，没有训练或采集新基线预测。旧v1保持不变；v1→v2同时改来源、措辞、数字和规模，不能作为纯样本量对照。该数据步骤当时只提出DPO新小集28步、小集168步、大集168步方案；数据代码已读完，2026-10-16执行情况见下段。现有train_sft_control.py仍是原28步专用入口，不能直接用于168步。没有启动后续Phase4或付费资源。
2026-10-16已完成v2 DPO三分支真实CPU训练：A新小集28对/28步、B同小集28对/168步、C大集168对/168步；三者从同一冻结v2 SFT初始化，共享56对开发集。`data/preferences.py`新增`load_preference_splits`，按manifest固定文件选择训练集，禁止把development当训练并校验全部数据哈希；旧配置默认train.jsonl。`scripts/train_dpo.py`新增v2设计/保护集检查、epoch与token/耗时/初始参数/源码哈希日志，核心DPO目标未改。配置`configs/dpo_v2_small_28.json`、`dpo_v2_small_168.json`、`dpo_v2_large_168.json`；新增`scripts/summarize_dpo_scaling.py`核对共同起点、固定配置、A/B前28步数值与顺序、每轮覆盖/曝光和token账目，全部通过。新增5项检查，完整86项测试通过。

训练目录：A `artifacts/checkpoints/dpo-lora-20261016-141339-249a77f0`；B `artifacts/checkpoints/dpo-lora-20261016-142317-5d529ace`；C `artifacts/checkpoints/dpo-lora-20261016-145728-bc1fc34c`。汇总`artifacts/evaluations/dpo-v2-scaling-training-20261016-152209-97915019`；批次日志/索引`artifacts/logs/dpo-v2-scaling-20261016-141008-2f738fcf/`；说明`reports/phase3-dpo-scaling-training.md`。三组reference不变无梯度、policy冻结参数不变/LoRA更新、重载logits最大差0.0。开发相对偏好改善45/56、53/56、55/56，loss 0.575274、0.258704、0.250463；开发chosen平均log probability相对原SFT下降1.790009、11.897777、13.847387，rejected下降更多。这些是当时训练步骤的固定答案评分，后续真实生成结果见下段；不可称为工具通过率或断言样本量是原问题主因。B/C回答token分别均4500 chosen+4788 rejected；padding输入28932/28964，只差32，但完整运行还含不同数量前后评分，成本不同。

2026-10-16已完成冻结SFT与A/B/C的同条件实际生成评测，入口`scripts/compare_dpo_scaling.py`，元数据/成对汇总`evaluation/scaling.py`，说明`reports/phase3-dpo-scaling-behavior.md`；产物`artifacts/evaluations/dpo-v2-scaling-behavior-20261016-163551-cb6e9477`。新56题严格通过SFT/A/B/C为36/33/7/4，数值通过44/41/7/4；48条算术成对数值通过16/13/0/0（分母24），问候通过8/8/4/4（分母8）。原工具题19/19/4/2（分母20），括号33/34/4/1（48），空格12/12/3/0（12），旧偏好开发7/5/1/1（10）。C的新算术数值0/48，结构/路由退化明显。新56题SFT→C为0改善32退步，B→C为0改善3退步；扩大覆盖没有挽回本次退化，不支持仅凭样本量解释。

正式584条记录：SFT旧90条经身份/数据/代码/设置校验后复用，另有3条实际生成探针完全重现；SFT新56与各DPO145种输入真实生成，三分支各有1条重复输入缓存。额外分别全新基础模型只加载B或C，各2题复现四项生成字段一致，见single_adapter_verification*.json；合计498次实际生成。逐条重算584评分、20组汇总和25组变化文件、核对20预测与6源码哈希通过，见verification.json。新增test_scaling_behavior.py的3项检查，完整89项测试通过。严格AST与数值、24道互补算术对与chosen/rejected偏好对分开统计，8问候不进入算术成对分母。历史组有重复/已见探针，不能合并独立测试总分。

下一步先阅读本次代码与失败样本，解释固定偏好改善但chosen绝对概率/实际生成退化，再设计一个受控实验。当前仍是单种子、单次子集、共享模板、合成负例，无最终未查看测试；没有新v2数据的SFT168步对照、没有开启GPU或Phase4。旧compare_sft_dpo.py/diagnose_dpo_preferences.py仍有train.jsonl默认假设，使用小集分支前必须适配实际train_file/selected_data_files；本次入口已按实际训练文件处理。

### 2026-10-16 已完成：四轮退化诊断与恢复尝试，仍未完全修复

用户要求解决C严重退化。四轮受控修复实验已完成，详见reports/phase3-dpo-repair.md。train_dpo.py支持chosen_nll_weight（默认0保持纯DPO）、可选reference缓存复用；compare_dpo_repair.py比较并做预先质量门；audit_dpo_repair.py独立重算。完整103项测试通过、1条已有PEFT提示；四轮冻结/LoRA更新/重载通过，logits最大差0。

四轮同冻结SFT起点、168训练/56开发、168步、seed/顺序/beta/裁剪/token曝光。权重1/LR1e-4、权重5/LR1e-4、权重5/LR2e-5、权重5/LR1e-5的新开发严格31/36/35/36、数值31/36/42/44。最后候选工具20/20、括号34/48、空格12/12、旧偏好7/10，新算术成对16/24，新56条结构/路由全通过；全部146条各145通过，有1条新增坏JSON，无上限命中。新56题相对SFT严格/语义均0改善0退步。四个回归恢复门通过，但“新开发严格或数值超过SFT”门未过，全部五门仍未通过；不晋升新baseline、不宣称纯DPO胜出。旧工具/括号各净增1只属小幅开发诊断收益。

总汇artifacts/evaluations/dpo-repair-summary-20261016-201033-4f337b8d。四轮训练/评测完整路径见总汇summary与阶段报告。最后训练artifacts/checkpoints/dpo-lora-20261016-194142-df9d5829；评测artifacts/evaluations/dpo-anchor-20261016-195656-ae7cb3a7。每轮438记录重算（含复用SFT/C）、15预测哈希、10对照/changes、源码/缓存/质量门核验；另有15份相邻单变量changes。580次新生成，584候选记录含4次重复输入；不合并独立测试准确率。

证据：原C chosen均值−3.64→−17.49而rejected下降更多；算术共享17至19前缀token，公共结构在直接概率差中抵消；固定权重5降低LR后恢复稳定。SFT49/56固定对chosen概率更高而生成语义44/56，12道实际错题仅3道与rejected算式/结果匹配。未发现shift/mask/符号/reference错误；机制线索不等于各项独立因果已确定。

第二轮保存224对reference分数及SHA，第三/四轮复用前核对模型/数据/编码/评分实现及4次前向；首步loss/logp/梯度与未复用控制精确一致。初始汇总明确复用相同模型已测值；训练后完整重新评分和冻结/重载。前两轮精确源码在评测source目录，SHA匹配清单。所有流水线已完成，不要重跑四轮；源码/方案提交49d976a、6d95dd3、f9174f8，最后汇总与质量拒收入口另行提交。

下一步优先采集训练输入真实SFT错误重建偏好，补同数据同预算chosen-only SFT控制；新开发净提升仍未完成。开发答案不加入训练，不继续盲扫当前合成数据LR。尚无新v2纯SFT168步控制、最终未查看测试、GPU或Phase4。先读修复代码与报告再推进。

## 最终审计补充：总分会掩盖失败类型恶化

第四轮旧括号题`bracket-seen_large-sequence-left_plain`，输入`Multiply 12 by 3 first, then add that result to 41.`。SFT输出合法JSON但表达式为`12*3`，参数错误；候选输出`{"type":"tool_call","name":"calculator","arguments":{"expression":"12*3+"41"}}`，变成非法JSON。两者端到端都失败，因此总通过数与端到端transition没有显示新结构退步。

审计现在额外逐题拒收任何新JSON/schema/路由退步，测试覆盖总成功数和总结构数都不变仍有单题恶化的情形。该门是审计发现后新增的独立保护，不改写原五项预先验收历史；第四轮此门也未通过，不能替换冻结SFT。完整提升与该结构问题都仍需解决。本轮没有把失败的候选标为成功。

### 2026-10-17 已完成：训练输入真实错误采集，收紧Phase 3范围

入口scripts/collect_sft_training_errors.py，仅对data/preferences/dpo_v2/train.jsonl的168输入做冻结SFT贪心生成；原v2数据与基线不改。产物artifacts/evaluations/sft-training-errors-20261017-200708-66935487，含sft_train.jsonl、candidate_preferences.jsonl、run_manifest.json、summary.json、report.md及补充synthetic_coverage.json。168条严格108/语义130，38候选覆盖12算术family（每族2—4条，无完整12格），36算式错误/2非法JSON。22个数值等价严格失败排除；无空、上限或超长过滤。候选完整编码chosen/rejected最大90/91；与56原开发和444条保护记录隔离审计通过。38个实际错误中12个与原同题合成负例AST/数值匹配，26个不匹配；不据此声称负例来源是退化的唯一原因。

已完成--audit-only：核验冻结文件和数据身份，重算168条评分/筛选、38候选内容、分组与审计，无须加载模型权重。新增8项筛选测试，全套111通过、1条已有PEFT提示。候选training_ready=false，不要直接当作平衡v3数据或已训练结果。

用户提出Phase 3耗时偏长。本阶段最小闭环早已完成，后续是扩展诊断；不以DPO必须胜出作为结业条件。下一步只冻结有限对照：优先复用38个相同输入/chosen，做chosen-only SFT、真实负例纯DPO，若归因来源再加同题合成负例DPO。配置、步数、曝光及评估方案先冻结，分别记录token量；现有train_sft_control.py仍为旧v1/28步专用，须适配才能用于新对照。本轮未实施新数据清单/新训练。若有限对照无收益，如实收尾Phase 3，不再无限调LR；仍保留旧能力与新增结构退步检查，不使用开发题补训练。

### 2026-10-17 后续已完成：冻结38同题三组数据与有限实验方案

入口scripts/build_matched_preferences.py，新增tests/unit/test_matched_preferences.py的7项匹配/预算测试，全套118通过、1条已有PEFT提示。冻结目录data/preferences/dpo_v3_matched_errors：train_real.jsonl、train_synthetic.jsonl各38对，train_chosen.jsonl为同38个chosen，development.jsonl为原56条；另有manifest.json与experiment_plan.json。三组input/chosen/target/family/order一致，真实/合成rejected中12条相同、26条改变。构造前完整重审原采集，构造后已用--audit-only重建并核对全部文件/方案，未改v2或历史训练源码。

方案：同冻结SFT起点、各38步单遍、LR1e-5、seed42、AdamW无weight decay、clip1；两组纯DPO beta0.1、chosen_nll_weight0。固定三次训练，不追加超参数扫描；同chosen曝光不等于同token算力。共同生成使用原五组146记录，保留逐题结构/上限门；DPO额外收益还需与本轮chosen-only比较，不能只胜原退化C。开发数据反复查看、数字外推、单种子及小样本限制均保留。

本步没有训练新模型；runner_adaptation_required=true，下一实现步骤是适配新训练与共同评估入口，再执行冻结方案。旧train_sft_control.py只适用v1/28步；旧train_dpo.py未完整审计新设计，不能直接拿旧命令声称已按新方案训练。完整回顾、可能原因与阅读顺序见reports/phase3-matched-control-plan.md。计划SHA256为a3520bf40a59e07e058b0cb0804571e4d999b197c3ca36de0b110b5601465405。

### 2026-10-17 已完成：38同题三组训练、生成和质量拒收

新入口scripts/matched_control_common.py、train_matched_control.py、compare_matched_control.py；旧训练/生成源码未改。新增test_matched_control_checks.py的7项曝光与质量门测试，全套125通过、1条已有PEFT提示；另用相同SFT结果自检六个对照/30份changes不能误报DPO收益。实现提交0bf89a7。

三组均38步、同初始参数SHA9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb、同ID/顺序/1108 chosen token、LR1e-5。训练循环耗时继续SFT64.24秒、合成DPO120.88秒、真实DPO133.21秒，不含评分/保存/生成。rejected token分别0/1116/1043。冻结base/reference、无reference梯度、LoRA更新、保存重载均通过，logits最大差0。真实DPO训练margin改善37/38，但平均原始policy margin仍-7.5635；不可当生成正确率。

训练目录artifacts/checkpoints/matched-control-v3-20261017-01；生成目录artifacts/evaluations/matched-control-v3-20261017-01。严格五组顺序新56/工具20/括号48/空格12/旧偏好10：baseline36/19/33/12/7、chosen_sft36/20/34/12/7、synthetic_dpo36/19/33/12/7、real_dpo36/20/34/12/7。新56语义均44，算术数值36/48、算术成对16/24、问候8/8，natural数值仍4/16。真实DPO与继续SFT只有1条文本差异且both_fail，所有组严格/语义均0改善0退步；相对合成DPO工具/括号各改善1题，但同样收益可由继续SFT获得。

三个新模型均在bracket-seen_large-sequence-left_plain输出相同坏JSON，原SFT为合法JSON但算式错误；每模型JSON/schema/route均145/146，原baseline146/146，无上限命中。新增结构错误也发生在SFT，不能全部归因DPO。恢复门因结构失败；两组DPO额外收益门也失败。未晋升baseline，不声称模型性能问题已解决。

新生成435次加3次冻结SFT探针，共438次，3条同模型重复输入缓存；baseline146历史输出复用，身份/版本/源码/评分和探针重现通过。--audit-only --require-admission实际退出2，verification.json artifact_audit_passed=true但any_dpo_added_value=false；584记录、20预测哈希、六个对照、30份changes、训练模型/日志及固定分数身份核对一致。supplemental_checks.json记录三组相同坏JSON与两个DPO首步相同负例时指标一致。质量拒收不是审计程序报错。

按事前方案本轮收尾，不追加LR/beta/anchor/步数扫描；如实保留条件性负结果。Phase 3最小闭环及本条扩展诊断已完成。后续可按项目路线讨论rollout/环境与预算，不因未胜出继续无边界训练；GPU/付费资源仍未授权或启用。

### 2026-10-17 已完成：Phase 4 CPU rollout接口验证

新增configs/rollout_cpu_smoke.json、rollouts/records.py、rollouts/hf_sampling.py、scripts/collect_rollout_smoke.py，以及unit/test_rollout_records.py和integration/test_rollout_sampling.py，共12项新测试；全套137通过、1条已有PEFT提示。未改旧生成器、评分器或训练源码。

固定baseline v2，采样来自dpo_v2/train的4题：12/47/20符号right_grouped、natural right_plain、sequence left_plain、greeting-0-0。4次/题，seed42+group_index*1000+sample_index，temperature1/top_p1/top_k0，不继承模型隐藏生成默认值。17次实际生成包含首样本复现。16条均EOS结束，400响应token，response_batch [4,4,41]；EOS16个保留、256个padding排除。strict8/16、semantic9/16、坏JSON4，文本unique组数1/3/3/4，四组严格得分均恒定；语义仅sequence组1/4成功。样本共享family且人为选题，不是开发评测或泛化提升证据。

采样raw policy logp与processed behavior logp在本配置实际相同，最大差0；每条实际响应ID直接重算完整前向，最大误差4.1962e-5，通过atol=rtol1e-4。模型初末SHA均9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb、全冻结无梯度，首样本同设备版本/seed复现。不可用跨硬件同seed必然同token的说法。

产物artifacts/trajectories/rollout-smoke-phase4-20261017-01：run_manifest、rollouts.jsonl、run_checks、response_batch.pt、summary、verification。--audit-only已完成，不加载模型，重算token解码、规则评分、分组、mask与张量并核对身份；概率前向重算发生在采集时，无模型审计不重跑前向。复核不必再采样。

当前仅Transformers CPU采样，不是GRPO trainer、vLLM或多轮Agent；未定义新训练reward。若直接以严格通过二元reward，四组advantage均为0；不能用文本多样性冒充可学习信号。下一步先规划GPU/引擎资源，或明确调整顺序做离线奖励审计；付费环境预算仍需用户明确，不能直接租机。官方环境与概率接口链接、代码阅读和下一步接入核验见reports/phase4-rollout-contract.md。

以下保留继续Phase2的可选修正路线。

若继续 Phase 2：

1. 保留 v1/v2 和全部现有报告，不覆盖，新增 v3。
2. 同一数学语义加入 spaced/compact、Calculate/Compute、左右乘数、显式表达式、sum/product、先后步骤的变体。
3. 同时加入 `a*(b+c)` 与 `a*b+c`、`(b+c)*a` 与 `b+c*a` 的对照，不只强化括号，避免无括号题退步；数据比例也需考虑原 final 路由样本。
4. 为同一语义变体设置 family 标识/分组划分策略。若比较措辞泛化，应明确哪些数字/语义族/模板被隔离；不要把同一族的简单改写跨训练和最终测试后称为独立泛化。
5. 先标注/审计/人工复查自然语言与表达式，再用真实 tokenizer 检查；只通过自动 calculator 校验不能证明题意一致。
6. 制定单变量对照：固定 base、LoRA、seed、生成方式，明确控制总步数、样本曝光或 token 预算中的哪一种；v1/v2 目前不是单变量实验。
7. 多指标看 JSON、路由、严格 AST、数值、成对通过、分组失败；训练 dev loss 不是唯一指标。
8. 开发评测可反复诊断，最终测试尽量在方案冻结后才查看；最终测试不应成为调参循环。

这些是建议，当前没有创建/训练 v3，也没有把新诊断题追加到训练集。

## 10. 可直接复制的操作命令

以下在 PowerShell 执行；首先切到真正项目根目录。使用本项目解释器，不依赖激活。

```powershell
Set-Location "C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab"
& ".\.venv\Scripts\python.exe" -c "import sys; print(sys.executable); print(sys.version)"
& ".\.venv\Scripts\python.exe" -m pytest tests -q
```

若迁移后缺包，按需安装，不需要在当前完整环境反复安装：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[train]"
& ".\.venv\Scripts\python.exe" -m pip install pytest
```

数据/模板检查：

```powershell
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py
& ".\.venv\Scripts\python.exe" scripts\inspect_hf_tokenizer.py --local-files-only
& ".\.venv\Scripts\python.exe" scripts\validate_sft_dataset.py --config configs\sft_lora_expanded_v2_cpu.json
```

最后一条审计默认仅使用本地 tokenizer；若只查结构、标签和隔离，可加 `--skip-tokenizer`。

以下会重新训练、产生新目录，接手时不必为读取交接文档而运行：

```powershell
& ".\.venv\Scripts\python.exe" scripts\train_sft_lora.py --config configs\sft_lora_expanded_v2_cpu.json --local-files-only
```

复用现有 v2 做原 20 题评测：

```powershell
& ".\.venv\Scripts\python.exe" scripts\evaluate_tool_calls.py --run artifacts\checkpoints\sft-lora-20261008-202445-a3d10354 --local-files-only
```

复用现有 v1/v2 做诊断：

```powershell
& ".\.venv\Scripts\python.exe" scripts\diagnose_brackets.py --run-v1 artifacts\checkpoints\sft-lora-20261008-180828-05ced0f0 --run-v2 artifacts\checkpoints\sft-lora-20261008-202445-a3d10354 --local-files-only
& ".\.venv\Scripts\python.exe" scripts\diagnose_brackets.py --run-v1 artifacts\checkpoints\sft-lora-20261008-180828-05ced0f0 --run-v2 artifacts\checkpoints\sft-lora-20261008-202445-a3d10354 --data data\eval\bracket_spacing_control_v1.jsonl --local-files-only
```

--run 指向含 summary.json/adapter/tokenizer 的整个训练目录，不是 adapter 子目录。脚本不自动选择“最新训练”，需用户/接手对话明确指定。

相对 --config/--run/--data 参数按调用时 cwd 解析；配置内部数据路径由训练/审计脚本相对项目 root 解析。换工作目录时使用绝对参数或先 Set-Location。

训练和评测都新建时间＋随机编号目录，不覆盖旧结果。仅诊断会生成新评测文件，但不会训练权重。

## 11. 测试覆盖与已知提示

| 测试文件 | 主要验证内容 |
| --- | --- |
| test_losses.py | shift、mask、sum/average、DPO 更新方向、GRPO 标准化/clip/KL |
| test_sft_data.py | 回答及终止训练、EOS-as-PAD、HF/manual loss 一致、过长/坏模板/重复/坏 JSON |
| test_hf_tokenizer.py | 适配器接口、PAD/EOS 回退等，用替身不联网 |
| test_training_helpers.py | split ID/prompt、空训练输入 |
| test_tool_call_evaluation.py | parser/schema/算术/路由/分母等评测器边界 |
| test_dataset_validation.py | 标签与数据隔离审计 |
| test_bracket_diagnostic.py | 48 题/24 对设计、数值与严格区别、前缀控制、成对要求 |
| integration/test_lora_training.py | 随机 tiny Llama＋真实 PEFT，基础参数不变、adapter 更新、保存恢复 |

集成测试不下载 SmolLM2；正式模型结果来自另外运行的训练/评测脚本。缺少 PEFT/Transformers 时集成测试会跳过，所以不能只看 pytest 命令结束就声称完整覆盖；当前本机确实安装了依赖，137 项无 skip。新增 test_preferences.py 检查偏好审计边界，integration/test_dpo_training.py 检查真实 PEFT 的 DPO 更新方向、reference 冻结与保存重载；test_behavior_comparison.py 检查 ID 对齐与改善/退步统计；test_preference_diagnostics.py 检查相对改善与chosen绝对变化的区分及分组汇总；test_balanced_preferences.py 检查v2合法偏好与隔离、重复模板格拒绝、小子集内容一致性；test_preference_selection.py 检查训练文件选择、共同开发集、禁止开发/未固定文件、全文件哈希与目录范围；test_scaling_behavior.py检查实际小集/大集暴露、24完整算术对及问候分离、数值等价与严格AST的区别；test_dpo_anchor.py检查混合目标与公共前缀梯度；test_dpo_repair_control.py检查声明的权重/LR变化及逐步曝光；PEFT集成同时覆盖纯DPO与混合目标。test_training_error_selection.py新增8项真实错误筛选测试，覆盖数值等价、空/截断/超长、完整坏JSON、错误chosen及来源记录。

当前唯一警告来自随机 tiny 模型保存时 PEFT 找不到原始 config 路径并假定 vocabulary 未修改；测试通过，不是实际模型训练失败。

用户以前遇到 Pylance 内部 TypeError(isStubFile)，属于编辑器分析器崩溃提示，不能当成已确认 Python 运行错误。是否当前仍出现未重新验证，不声称已经修复；以解释器/实际测试定位，必要时单独处理编辑器。

旧日志 torch_dtype deprecated 提示已在当前模型加载代码改为 dtype=torch.float32；不需要因此升级整个依赖组合。

## 12. 文档阅读索引与新对话启动方式

建议新对话阅读顺序：本文 → README → reports/phase3-dpo-repair.md → 最新括号诊断报告 → trainers/sft_lora.py、train_sft_lora.py → tool_calls.py、generation.py、diagnose_brackets.py → 相关数据/测试。

reports 中 phase2-sft-data.md、phase2-real-tokenizer.md 等是阶段性教学文档，其“当前尚未训练”等描述反映当时进度；与现状冲突时，以本文更新时间、实际代码和产物为准。不要因为早期文档一句“后续再接 LoRA”就重做已完成的训练闭环。

阶段文档索引：

- reports/phase1-foundations.md：张量目标函数。
- reports/phase2-sft-data.md：预处理、掩码和 causal shift。
- reports/phase2-real-tokenizer.md：真实模板与 EOS。
- reports/phase2-lora-training.md：最小训练与保存。
- reports/phase2-tool-evaluation.md：严格工具请求指标。
- reports/phase2-expanded-data.md：v1 数据。
- reports/phase2-bracket-data.md：v2 数据。
- reports/phase2-bracket-diagnostic.md：最近诊断设计、结果和限制。

可给接手对话的提示：

> 请完整阅读 C:\Users\app\Documents\Codex\2026-9-28\AgenticPostTrainingLab\PROJECT_HANDOFF.md，核对当前代码和训练/评测产物后继续。我们已完成 CPU SFT＋LoRA 与 Phase 3 的基线冻结、偏好数据、首轮 DPO 保存重载、逐项行为比较。随后完成v2规模对照、四轮修复与38同题三组对照，无DPO额外新集收益且仍有结构退步，Phase 3有限实验已收尾，不重跑。当前Phase 4的CPU随机rollout接口已验证，先看phase4-rollout-contract.md与phase3-matched-control-results.md，再规划GPU/引擎与预算。vLLM/GPU未接入，GRPO尚未真实训练。不要复用 MiniServe 环境，也不要把现有诊断集当最终测试。先说明当前进度，再按学习节奏推进。所有新增 Markdown 用中文。

若接手对话没有本机文件访问能力，需要上传本文和相关代码/summary；Markdown 中的本地路径本身不会把文件传给它。

后续每完成一次数据/训练/评测，请更新本文的日期、状态、产物路径、指标和限制；不要覆盖历史实验记录或把未来计划改写成已实现事实。
