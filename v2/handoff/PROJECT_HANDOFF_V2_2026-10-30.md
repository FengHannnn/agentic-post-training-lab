# AgenticPostTrainingLab — V2 Project Handoff
更新：2026-10-30（Asia/Shanghai）。用途：供新对话接续项目。正文为已核实状态及后续建议，不自动授权执行云端训练或覆盖历史结果。

## 0. 先读这一页

**项目主体已从 135M 单轮后训练实验迁移到 1.5B 多轮 Agentic RL。V1 已完整封存；V2 环境、奖励与更新内核、4090D 验证、原始模型 baseline 已完成。下一步是建立 V2 的专家轨迹 SFT 数据及训练流程。**

- 用户已确定：Qwen/Qwen2.5-1.5B-Instruct；Agentic RL 为主体；优先实现可验证的 turn-level credit assignment 与错误恢复。
- token-level 路由 / multi-teacher OPD 留作扩展。用户说的 GDPV 指 GDPval；目前借鉴其真实工作任务方向，没有实现或运行 GDPval。
- 最新真实模型 baseline：development 4 个任务，成功 0/4、故障恢复 0/2、严格动作格式通过 26/48。0 次训练更新。
- 主要失败是工具协议、SQL schema 和证据 ID 的依赖关系。下一步先训练基本执行能力，再从同一 SFT 检查点分别验证 DPO 和 GRPO。
- 当前有手写 RL 单次更新内核及小模型测试；尚无完整 1.5B 随机轨迹采集—训练—刷新—评测流水线，也没有正式 V2 SFT/DPO/GRPO 训练结果。
- 4090D 与 1.5B 权重已验证可用，模型下载问题已解决。无需重新做网络排查、重新下载或默认重跑 baseline。
- V1 结果按需读 [V1 History](PROJECT_HISTORY_V1_2026-10-30.md)。开始 V2 工作不必通读所有 V1 报告。

交接依据：项目内保存的原始 2026-10-27 handoff、现有源码与配置、归档凭据、云端回传包和本地复核记录。用户原 Desktop 路径现已不存在，保留副本位于项目根目录。发生冲突时，采用最新用户决定和具体运行证据；历史文档中的待办不是当前执行指令。

## 1. 用户目标、工作方式和约定

目标是形成可解释、可复现、有实际多轮任务表现的后训练项目，服务学习与简历展示。需实测 SFT、DPO、GRPO 的效果，而不是只展示 loss 下降或工程检查通过。

已经确定的约定：

1. 文件名全部英文；报告正文可用中文。报告先给结果，简洁解释适用范围与扩展边界，避免反复防御性措辞。
2. 给文件链接，不自动打开报告。讲解代码时给阅读顺序、关键函数和学习重点。
3. 云端运行由用户上传 ZIP/脚本并执行命令，再返回 review ZIP。本地负责实现、测试、打包、核查；没有已连接的云端 shell。
4. V2 命令不再要求时价参数，也不计算费用。保留有限步数、超时和明确停止条件。
5. V1 归档及历史实验保持原样，V2 代码放独立目录。此前“先把 V1 GRPO 做到显著改善”的方向已被用户选择架构升级所替代。
6. 对合理的实现细节可继续工作；数据规模、训练预算等尚未确定项应先形成具体方案，必要时询问。不要把本文建议误记为已批准的超参数。

## 2. 目录与环境

| 用途 | 路径 |
|---|---|
| 本地项目根目录 | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab |
| V2 主目录 | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2 |
| 本对话产物 | C:/Users/app/Documents/Codex/2026-10-28/new-chat/outputs |
| 本地回传包解压与复核工作目录 | C:/Users/app/Documents/Codex/2026-10-28/new-chat/work |
| V1 归档目录 | C:/Users/app/Documents/Codex/archives |
| 云端 V2 | /root/rivermind-data/AgenticPostTrainingLab/v2 |
| 云端环境 | /root/rivermind-data/envs/apt-vllm |
| 云端模型缓存（相对 V2） | .cache/qwen2.5-1.5b-instruct |
| 本地 Python | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/.venv/Scripts/python.exe |

注意本地日期目录是 **2026-9-28**，不是 2026-09-28 或 2026-10-28。

Git：分支 main，当前 HEAD 为 eda30590b021798edf8c6209f732917aabeeb38e。许多后期 V1 文件及整个 V2 仍为 untracked，根 README 有修改；Git HEAD 并不包含全部成果。不要 git clean/reset 丢弃工作。V1 另有完整 ZIP 归档。

云端从 L20 克隆迁移至 RTX 4090 D。已验证环境：Python 3.11.17、torch 2.8.0+cu128、vLLM 0.11.0、transformers 4.57.6、peft 0.17.1、accelerate 1.15.0、safetensors 0.8.0、huggingface_hub 0.36.2；驱动 580.119.02、计算能力 8.9、BF16 可用。不要直接在云端安装根项目依赖并覆盖 torch/vLLM 的已验证组合；V2 脚本自行加入 src 路径。

## 3. V2 主体框架：业务数据对账 Agent

当前任务是合成业务净收入对账，核心是多轮读取规则、查询数据、保留证据、计算、校验与提交。

流程：

Task → system/request → 模型生成严格 JSON 动作 → 环境执行工具 → 观察回到上下文 → 下一轮动作 → submit 结构化交付物 → 确定性验证 → 完整轨迹与评测；后续在这个闭环上接训练。

### 3.1 业务任务与数据隔离

- SQLite 两张表：orders(order_id, order_month, region, segment, status, gross_cents)；refunds(refund_id, order_id, refund_month, refund_cents)。
- 所有金额为整数 cents。目标是 current_net_cents、prior_net_cents、change_cents。
- 收入按订单月份统计 paid orders；退款按退款发生月份统计，并关联原 paid order 的 region/segment，即使原订单属于前一个月；取消订单排除。
- 当前生成器固定 train 12、development 4、test 4，family 分别使用 1000+、2000+、3000+，种子与 family 隔离。每个分区约一半任务有首次查询临时故障。
- 模型可见请求与工具观察，不直接得到 seed、split、family、故障标签或标准答案。
- 当前是一个业务任务族的合成实例集合。正式泛化对比需进一步扩充任务与 family 覆盖。
- 运行轮数来自 Task.max_turns，目前 12；仅修改配置中的 max_turns 不会自动修改任务对象。task_count 只选取现有任务，不负责扩充 make_tasks。

### 3.2 工具协议与证据链

动作必须是只含 tool 和 arguments 两个顶层字段的 JSON 对象。工具为：

| 工具 | 作用 |
|---|---|
| read_policy | 读取业务口径 |
| inspect_schema | 获取实际表结构 |
| query_sql | 只读查询并生成可验证月度证据 |
| calculate_change | 使用已验证的月度 evidence ID 计算差额 |
| check_result | 检查候选提交 |
| submit | 交付三个指标及证据引用 |

当前 observation 以 user message 进入后续 chat，不是原生 tool-role API schema。SFT 格式必须与实际 runtime 一致。

SQL 约束包括只读 authorizer、表与函数限制、5000 字符、最多 20 行、虚拟机执行预算；拒绝写库、附加数据库等操作。正确数值之外，还检查查询读取 orders/refunds，并在两个独立金额扰动世界重放 SQL，防止写死答案或无关查询获取奖励。

成功提交需要业务规则/schema 已读取、三个整数指标正确、真实有效证据和 policy_id=revenue-v1。专家会先 check_result；当前成功判定并不强制必须先调用它。查询证据形如 qN，计算证据形如 cN，必须来自实际执行。

恢复指标要求最终成功，且在发生工具错误后取得新的已验证指标。报告中的故障恢复率使用故障任务作分母；“说要重试”或单次工具调用成功不等于恢复成功。

### 3.3 奖励与 turn-level credit

现有 environment.py / rewards.py 已实现：

- outcome_reward = 成功终局奖励 1 − 每次动作成本 0.01；错误动作同样计成本。
- potential 为 0.2 × 已验证业务指标数。
- process_reward = Phi(next) − Phi(current)，终局无论成败都将 next potential 设为 0。
- potential 模式使用 outcome + process；outcome 模式只用 outcome。
- gamma=1 时，完整 episode 的 process 奖励和为 0。它改变过程中的 return-to-go，而不增加完整轨迹总分。重复获取同一进度不会额外刷分，没有固定“恢复奖励”。
- 同一个任务至少两条完整 episode；每条每轮 advantage = shaped return-to-go − 其他 episode 的平均 outcome 总分（其中包含调用成本）。
- 这里采用 leave-one-out baseline，不是 V1 按组标准差归一化的实现。采样独立性还需未来 collector 保证，函数检查任务 ID 与数量不等于验证随机独立。

后续重点是同一起点比较 outcome-only 与 turn-level potential 两支，检验成功率、恢复率和效率是否改善。目前已实现数学/执行逻辑，实际 1.5B 训练效果待测。

## 4. Phase 安排与完成状态

下表是为交接统一整理的 V2 工作分期；旧 V1 Phase 编号见历史文件。具体实验规模和超参数尚待确定。

| Phase | 交付目标 | 截至本次交接 |
|---|---|---|
| V2-0 封存与迁移 | V1 可恢复归档、明确模型与主体 | 完成 |
| V2-1 可执行环境 | 任务、工具、多轮 runtime、证据验证、奖励、重放 | 已实现并测试；脚本专家 20/20 |
| V2-2 硬件与 baseline | 4090D、固定模型、真实模型原始表现 | 完成；baseline 0/4 |
| V2-3 SFT 基础执行能力 | 专家/恢复轨迹、assistant-only loss、训练、前后评测、冻结检查点 | 下一步；数据与正式训练 runner 尚未建立 |
| V2-4 DPO | 同一 SFT 起点、有效偏好对、相同评测 | 待实现与运行 |
| V2-5 Agentic GRPO | 随机多轮采集、真实 LoRA 更新、刷新、outcome/过程奖励对照 | 更新内核已有；完整流水线和正式实测待完成 |
| V2-6 正式效果对比 | 扩充并冻结划分、多种子/配对评测、独立 test、简历结果 | 待完成 |
| V2-7 场景扩展 | 更长依赖、多文件证据、复杂交付物；可选 multi-teacher OPD | 后续扩展 |

比较关系：原始 1.5B → SFT；从同一冻结 SFT 分别出发做 DPO 与 GRPO，再比较 GRPO outcome-only / turn-level credit。不要默认把 DPO 权重作为 GRPO 的唯一父模型，也不要把现有诊断用一步 SFT adapter 当成正式 SFT。

## 5. 已完成实验与证据

### 5.1 工程与 4090D 验证

脚本专家在全部 20 个任务成功，10 个故障任务全部恢复；这验证环境和专家轨迹可执行。最新 V2 本地测试 47 passed；另有硬件诊断脚本和下载恢复脚本的独立测试记录，不合并冒充一次全套运行。

成功环境回传包是 v2-4090d-environment-review-20261030-46d42d16.zip；之前两个回传包包含未完成下载阶段，状态已被此包取代。

实际 1.5B 诊断包括：

- 7 个模型文件身份核对。
- FP32 策略与冻结 reference，512 token 的一次 LoRA SFT 更新：112 个训练张量、1,089,536 参数，56 张量发生变化；底座未变化、reference 无梯度，old/new 同路径和保存重载差值均为 0。
- 峰值 allocated 14.4072 GiB，reserved 14.6582 GiB。
- vLLM BF16 推理 max_model_len=8192、单序列、memory utilization=0.6；128 与 8176 token prompt 探针均完成。每条只输出约 2 token，这不是长输出吞吐测试。
- 该运行只证明上述探针设置可执行。8192 推理成功不代表 8192 训练显存已验证；正式 SFT 需按真实轨迹长度再做容量检查。

详见 [4090D Review](C:/Users/app/Documents/Codex/2026-10-28/new-chat/outputs/v2-4090d-environment-review-20261030-46d42d16.md)。

### 5.2 当前原始模型 baseline

配置：BF16、temperature=0、seed=2026102901、max_model_len=8192、每动作 max_new_tokens=256、每任务 12 轮、development 4 个任务、potential 奖励、training_enabled=false、wall limit 1800 秒。

| 指标 | 实测 |
|---|---|
| 任务成功 | 0/4 |
| 故障任务成功恢复 | 0/2 |
| 严格动作格式通过 | 26/48，54.2% |
| 工具错误 | 39/48 |
| 已验证业务指标 | 0 |
| 有效 check_result / submit | 0 / 0 |
| 总轮数 / 生成 token | 48 / 3659 |
| 结束方式 | 48 次全为 EOS；不是输出长度截断 |
| 优化器更新 | 0 |
| 作业时间 | 78.9326 秒；无费用估算 |

互斥错误：格式 22、错误月度证据引用 11、不存在的表 4、预设临时查询故障 2。成功的 9 次工具调用只是 read_policy 2 次与 inspect_schema 7 次。

四个任务首步都直接查询，未先读规则/schema；SQL 使用不存在的 metrics / reconciliations 表。读取 schema 后也没有形成有效查询；calculate_change 使用 observation 等占位证据。优先修复协议、工具顺序和证据链，比直接对全失败轨迹增加 RL 更新更具体。

本地重放核对了 prompt、原生 token 解码、动作、观察、奖励和终止状态；15 个运行源码文件与冻结记录及本地一致，6 个 tokenizer/config 文件一致。4 个 development 用于初始诊断；模型尚未做独立 test 评测。ScriptedPolicy 的 test 成功不是模型 test 成功。

详见 [Baseline Review](C:/Users/app/Documents/Codex/2026-10-28/new-chat/outputs/v2-baseline-review-20261030.md) 与 [Audit](C:/Users/app/Documents/Codex/2026-10-28/new-chat/outputs/v2-baseline-review-20261030-audit.json)。

## 6. 已有 RL 内核与后续必须补齐的接口

trainer.py 已有 response_scores、prepare_scores、update：

- 只对 response token 做 causal 对齐评分；log_softmax 用 FP32；prompt/工具观察不作为 action loss token。
- p_old 与 p_new 使用训练侧相同评分路径；prepare_scores 在更新前核验差值 <=1e-5，绑定 prompt/response，并保存 detached old/ref 分数。
- reference 必须完全冻结且无梯度；现有内核只允许 LoRA 参数训练。
- q 为实际采样行为概率；p_old 为采样策略在训练侧重算概率；p_new 为更新中策略；p_ref 为固定参考。四者角色不同。
- PPO ratio=exp(log p_new−log p_old)，clip=0.2；TIS=min(2,exp(log p_old−log q))，权重 detach。
- sampled k3 KL 使用 ref/new 分数，beta=0.02；这是采样估计，不应写成精确全分布 KL。
- 归一化为每个 action 内 token mean，再累加 action，除以 episode 数。**没有按每条轨迹轮数再平均，也不是全 batch token mean。**
- 逐 action backward 累积后一次 optimizer step；max_grad_norm=1。监测 mean action KL >0.05 或 clip fraction >0.5 时清梯度并在 step 前返回 updated=false；也检查梯度有限、非零。
- training_samples 要求 mode=sample、原生 prompt/response token ID、逐 token 对齐的有限 behavior logprob。当前 greedy baseline 没有训练所需 q，不能直接作为 RL rollout。
- 小型真实 PEFT 更新测试已覆盖内核；1.5B 完整 Agentic RL 尚未运行。

仍需实现：带采样概率的 1.5B 多轮 collector、输入/权重身份绑定、训练调度与有限预算、采样后 old/ref 缓存、更新后模型刷新、完整 checkpoint/Adam/RNG 恢复与重放、前后评测和分支比较。学习率、批量、组大小、轨迹长度与总更新量也需正式实验配置。

V1 的 adapter、Adam、概率缓存不能迁移到 1.5B。V1“延续 Adam”是当时同模型迭代的用户决定，并不授权跨模型续接。V2 首个正式 SFT 检查点尚不存在；未来 RL reference 应由该冻结 SFT 明确建立。

## 7. 下一对话建议从这里开始

**当前最具体的任务：设计并实现 V2 首轮 SFT，以完整可执行轨迹教会模型工具协议和恢复流程。此项是下一步建议，本交接任务没有启动训练。**

建议执行顺序：

1. 阅读第 9 节中的任务、环境和 runtime，核对当前基线及训练数据边界。
2. 用 train 分区生成专家成功轨迹，加入临时故障后的正确重试，以及可验证的 schema/SQL 修复情形。当前 ScriptedPolicy 可作轨迹种子；扩充数据时明确哪些轨迹由脚本、模型或人工提供。
3. 回放每条示范，校验完整成功、证据引用和标签；记录 family、来源、长度与文件摘要，避免 development/test 混入训练。
4. 构建与真实 chat template 一致的 SFT 样本，仅对正确 assistant 动作 token 监督，用户请求和工具观察 mask。不要无差别监督失败动作。
5. 定义 LoRA、学习率、epoch/step、梯度累积和有限训练预算；按实际完整上下文检查 4090D 显存。超过长度时采用明确的数据策略，不能悄悄丢掉证据链。
6. 本地验证格式、mask、数据隔离、更新与恢复；再提供有唯一英文版本名的云端 ZIP 和命令，不含价格参数。
7. 训练后评测相同 development 任务，保留原始 baseline；报告成功率、格式率、有效 SQL/证据、故障恢复、调用数、输出 token 和失败类型。
8. 首轮形成可执行策略后冻结 SFT 检查点，为 DPO 与两种 GRPO 奖励分支建立共同起点；正式结论前扩充任务、冻结独立 test 和统计方案。

尚未确定：扩充任务数量、示范比例、SFT 超参数、随机 RL 组大小与步数、DPO 对构造、独立 test 的正式验收标准。“显著提升”应预先定义为任务指标和统计标准，而不是无限训练直到挑出一个更高分。

## 8. 固定模型、版本与云端运行约定

模型 Qwen/Qwen2.5-1.5B-Instruct，配置 revision=989aa79，完整版本：
989aa7980e4cf806f80c7fef2b1adb7bc71aa306。

主权重 3,087,467,144 bytes，SHA-256：
dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee。

LoRA：q_proj/v_proj、r=8、alpha=16、dropout=0；112 个张量，共 1,089,536 参数。云端已缓存，完整 1.5B 权重不在 baseline review ZIP 中，也未在本地重新下载。

下载历史：Hugging Face 直连失败，镜像出现限流/超时/SSL EOF，代理变量为空；ModelScope 可达。已通过固定文件版本与 SHA/小文件 git blob 跨源核对下载，完整权重校验完成。恢复脚本为 diagnostics/download_v2_model_modelscope_20261030_v1.py；旧 retry_4090d* 不再是默认步骤。

当前 release manifest：v2-starter-v1-no-price-20261030，包含 17 个文件。无价格补丁已用于成功 baseline。cost_and_run.json 只是沿用的文件名，记录 cost_estimation_enabled=false，不代表仍在估价。

当前 baseline 已存在，runner 会拒绝覆盖现有输出和 ZIP。以下仅记录既有入口，不是要求重跑：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_baseline.py
```

未来训练应提供新的 runner/输出目录。上传新版本时先确认实际文件位置和名字；之前同名脚本被用户改名、find 深度与静默 test -n 都曾导致“命令没启动”。应显示找不到文件的原因或使用确定的上传路径。

run_v2_baseline.py 预算 1800 秒并预留打包时间。audit_v2.py 会读取当前源码核对运行 manifest 且独占创建审计输出；直接对旧目录重复运行可能因已有 audit.json 或后续源码变化失败。历史复核应使用包内冻结源码和独立输出，不覆盖旧审计。

## 9. 代码阅读地图

下列路径均相对 V2 根目录。

| 顺序 | 文件 / 函数 | 重点 |
|---|---|---|
| 1 | PROJECT_REQUIREMENTS.md；configs/baseline.json | 已确认产品方向与当前 baseline 配置 |
| 2 | src/apt_v2/tasks.py：Task、make_tasks、metric_sql、truth | 任务划分、数据口径、金标准与可见信息 |
| 3 | src/apt_v2/environment.py：execute_readonly、Environment._certify_sql / _call / _check / step | SQL 边界、证据链、反写死验证、终止和奖励 |
| 4 | src/apt_v2/runtime.py：SYSTEM、Generation、parse_action、run_episode、ScriptedPolicy | 真多轮上下文、严格动作、专家路径 |
| 5 | src/apt_v2/rewards.py：discounted_returns、assign_turn_credit、training_samples | 完整轨迹、LOO、训练 token 与 q 契约 |
| 6 | src/apt_v2/trainer.py：response_scores、prepare_scores、update | 因果对齐、四类概率、损失归一化、更新边界 |
| 7 | scripts/prepare_v2.py、evaluate_v2.py、audit_v2.py、run_v2_baseline.py | 模型准备、评测、重放、打包和预算 |
| 8 | diagnostics/check_environment.py | GPU inventory、模型就绪、训练/vLLM 探针 |
| 9 | diagnostics/download_v2_model_modelscope_20261030_v1.py | 固定文件身份、断点续传和完整哈希 |
| 10 | tests/test_agentic_v2.py | 工程行为的可执行规格 |

直接入口：[Tasks](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/tasks.py)、[Environment](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/environment.py)、[Runtime](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/runtime.py)、[Rewards](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/rewards.py)、[Trainer](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/trainer.py)。

vLLM 负责高效生成/采样；PyTorch + Transformers + PEFT 负责评分、反向与 LoRA 更新。当前没有 TRL 托管训练，也没有多机分布式或 multi-teacher OPD 实现。

## 10. 证据索引与交接核验

| 证据 | 位置 |
|---|---|
| 原始 handoff 保留副本 | 项目根 PROJECT_HANDOFF_2026-10-27.md |
| V2 要求 / 当前打包身份 | v2/PROJECT_REQUIREMENTS.md；v2/bundle_manifest.json |
| V1 封存回执 | v2/V1_ARCHIVE_RECEIPT.json |
| 成功硬件回传解压 | 本对话 work/reviews/environment-4090d-20261030-46d42d16 |
| baseline 保留 ZIP 与本地审计 | v2/artifacts/reviews/baseline-1.5b-20261029-01 |
| baseline 解压 | 本对话 work/reviews/baseline-1.5b-20261029-01 |
| baseline 本地复核脚本 | 本对话 work/review_v2_baseline_20261030.py |
| 无价格补丁 | 本对话 outputs/v2-baseline-no-price-patch-20261030-v1.zip |

baseline ZIP SHA-256：
97cd124cc625f43adf5ee96e6dcbbf979ae8229a81fdabe1f0db648843d577bb。
episodes.jsonl SHA-256：
97d77ba04e68b8b68482abb6113167464199b996f78dbaa3981649082eafb31a。
成功环境 ZIP SHA-256：
c59976d763e9fdddddcb83995c44eff542f19d3c394b59891b84b41ffe36b2e7。

2026-10-30 编写本交接时重新核对：V1 ZIP 哈希匹配回执；2015 个 V1 原始文件逐项匹配归档 manifest；V2 bundle 中 17 文件全部匹配。本次只整理文档，没有训练或改写实验结果。此交接另存于 v2/handoff，未修改已有 release manifest 绑定文件。

### 阅读历史文档时避免误接续

- 根目录 SFT “v2”、dpo_v2 是 135M 阶段的数据/检查点版本，不是当前项目 V2。
- 旧 handoff 的“下一步刷新采样”已经完成；旧报告的“待补 24 个实际训练题评测”也已完成，结果见 V1 History。
- V2 initial implementation 报告里的 baseline 待跑、46 tests、费用参数属于当时状态；以本交接与最新记录为准。
- baseline ZIP 文件名仍为 20261029，是发布/运行 ID 命名；实际回传审查在 2026-10-30。不要仅凭文件名判定时序。
- V1 6.46× 是限定工作负载的 microbatch benchmark，不是 V2 端到端性能。
- 当前可写进项目成果的是多轮环境、验证/重放、概率契约和基线诊断；SFT/DPO/GRPO 的 V2 效果数字待完成实验后填入。
