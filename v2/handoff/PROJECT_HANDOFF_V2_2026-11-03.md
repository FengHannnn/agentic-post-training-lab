> 历史交接归档：当前接续请读 PROJECT_HANDOFF_V2_GRPO_2026-11-05.md。截至2026-11-05，DPO短程第12步已达收尾标准，用户确认下一对话进入GRPO；下文保留当时状态。

# AgenticPostTrainingLab — V2 最新项目交接

> 2026-11-04 更新：E1 已完成本地复核（384 条轨迹、2,692 步）；86 步随机完整任务 13/16、greedy 完整恢复 21/28，258 步降至 0/16、6/28。C1 继续未通过，GRPO 暂缓。下一作业 E2 dpo-uniform-control-v1-20261104，仅将困难权重 4 改为 1，其余训练项匹配；四模型 528 条配对轨迹，169 对与 238 探针评分。报告见 reports/dpo-regression-checkpoints-v1-review-20261104.md，执行见 reports/dpo-uniform-control-v1-preparation-20261104.md。


> 用户最新验收决定：加权DPO退化过大，C1未通过；按原因逐项排查，GRPO暂缓。当前先运行E1 dpo-regression-checkpoints-v1-20261103（8模型、238评分探针、384条轨迹、零训练更新），后续E2权重、E3保真目标、E4续跑覆盖按结果逐项冻结。计划见reports/dpo-regression-investigation-plan-20261103.md。此前“结束C1进入GRPO”属于被本条替代的旧建议。

> 沟通约定更新：后续报告完成后，在对话回答中简要概括主要结果并提供报告链接，不自动打开；报告和回答直接陈述结果与证据，避免防御性表述。

> C1 结果更新（2026-11-03）：matched DPO补验与本地复核已完成。随机完整任务父模型8/16、加权SFT16/16、DPO0/16；随机完整恢复45/84、59/84、13/84。DPO最终SHA256为80c32888da7cea2f1962632e6261068ed47b955317077d124f0165842f2d1db0，SFT255 / DPO269，不晋升。60条完整任务、336条恢复、2317步严格重放通过；源码88份与溯源18份核对通过。下一步进入C2结果奖励GRPO，共同起点仍为f8fb95…（SFT513 / DPO11）。当前结果报告：reports/hard-recovery-dpo-matched-v1-review-20261103.md；完整证据：artifacts/reviews/hard-recovery-dpo-matched-v1-20261103/。以下“待补验/待准备”等表述保留为历史状态，最新状态以本条和roadmap第7节为准。

> C1 接续更新（2026-11-03）：首份匹配困难加权 DPO 配置、训练/完整恢复评测/审计与运行包入口已实现，真实数据和模型身份预检通过。新入口 scripts/run_v2_matched_dpo.py，run_id hard-recovery-dpo-matched-v1-20261103；GPU 训练与返回 review 复核仍待执行。下文“未实现/未准备运行包”等为本次接续前状态，当前进度以 reports/matched-dpo-preparation-20261103.md 与最新 roadmap 为准。

更新：2026-11-03，Asia/Shanghai。用途：新对话读完本文件即可继续当前项目。本文综合原 V1 历史、2026-10-30 V2 交接、后续已审阅实验和用户最新 grilling 决定。事实、已确认安排和实施建议分别标注；历史报告中的下一步按其当时状态理解。

## 0. 接续摘要：先读这一页

**项目当前主体是 Qwen2.5-1.5B 的多轮 SQLite 业务对账 Agent 后训练。SFT、DPO、chosen-only SFT、困难恢复加权 SFT及独立评测已完成；V2 的完整 GRPO 在线训练尚待实现。用户已确认约 10 天 / 60 小时收尾，下一步先做一次有边界的 DPO 补验，然后进入 GRPO。**

- 求职优先级：大模型后训练 / 强化学习算法实习，其次 Agent 开发；每天平均投入约 6 小时，一周半内形成可投递版本。
- 范围：沿用 SQLite，做一种具有实际前后证据依赖的轻量业务任务扩展。GDPval、多文件办公环境、网页界面与额外算法分支列为可选；multi-OPD 排除。
- DPO：争取比 chosen-only SFT 好；补验后仍无明显收益就分析原因、进入 GRPO。主目标看最终任务成功，完整恢复为核心辅助指标。
- GRPO：结果奖励与细粒度分支共用当前恢复加权 SFT，SHA f8fb95…，历史 SFT513 / DPO11。原纯 SFT 基线另行保留。当前候选登记尚未覆盖冻结 baseline。
- 最新独立模型结果：父 chosen-only SFT → 恢复加权 SFT，随机完整成功 24/32→28/32；标准恢复首动作有效 36/84→66/84；新增双指标均未验证的完整恢复，两模型各 0/32。
- 已有 RL 内核是逐动作 PPO surrogate + 逐轮 return/LOO baseline。组内 GRPO 优势、零方差处理、在线采样 / 更新 / 刷新 / 恢复全流程需补齐。
- 当前云端是 NVIDIA L20；本地代码、报告、数据和 review 位于项目 v2。用户上传运行 ZIP 到云端 v2，自己执行命令，返回 review ZIP。本地没有已连接的远端 shell。
- 本次只更新计划和整理交接，未准备新的 DPO 补验运行包、未启动新训练。接续从第 2 节执行，不再从首轮 SFT或环境迁移开始。

**文档优先级**：最新用户指示 → [当前项目要求](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/PROJECT_REQUIREMENTS.md) / [已确认收尾计划](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/project-roadmap-v2-20261103.md) → 本交接及具体运行证据 → 旧交接。原 V1 和旧 V2 文档保留作历史。

## 1. 已确认的用户决定与工作方式

1. 用户认可约 10 天 / 60 小时安排；第 7 天检查进度并冻结新增功能，最后约 3 天用于评测、分析和投递材料。使用相对实施日安排，未冻结具体日历日期。
2. 先进行 DPO 补验；与 chosen-only SFT 匹配起点、数据、困难场景权重和曝光。开发集至多两个配置后冻结候选。训练总次数未设“6 次”的硬上限，后续实质预算扩展共同商议。
3. 两条新 GRPO 均从当前 hard-recovery-sft-weighted-v1-20261103 出发。此前“新建纯 SFT M0 / GDPval-first”的大规划已被本收尾方案取代。
4. 长程只做一种 SQLite 分阶段业务对账扩展，依靠前后依赖增加难度；保留现有短任务作为回归。多文件环境和 GDPval 移至可选扩展。
5. 主指标是最终任务成功率，核心辅助是错误上下文后的完整恢复成功率。局部首动作、证据、SQL、重复调用、token、耗时用于解释行为。
6. 必交付：可复现代码、模型 / 配置身份、训练曲线 / 对照表、恢复案例、README、技术报告、简历要点与面试讲解。演示采用报告、终端或录屏。
7. 用户已读过 tasks/environment/runtime、SFT、DPO、恢复数据和加权 SFT 的主要代码；今后重点读采样概率、GRPO 损失、优势、奖励和关键评测。重复 runner / 日志 / 打包只说明流程与差异。
8. 中文沟通，报告直接给结果与证据，尽量避免“不能说明”“不代表”等防御性重复。英文文件名；文件存项目 v2，给链接，不自动打开。
9. 云端运行命令直接贴出；无需价格参数或费用估算。有限步数和时限、实时进度条、失败日志和回传打包继续保留。约 4 小时为近期作业基础配置，按 profile 共同调整。
10. 本对话结束时用户已认可最终规划，无需重新开展 grilling。只有新的实质取舍、缺失信息或平台权限需要再确认。

## 2. 新对话的第一项工作：DPO 补验准备

已确认的策略是“先一次补验，再 GRPO”。推荐按下面顺序落地；具体新运行配置和文件名尚待实现。

1. 阅读 [加权 SFT 配置](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/configs/hard-recovery-sft-weighted-v1.json)、[困难权重实现](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/hard_recovery_sft.py)、[现有多 epoch DPO 入口](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/train_v2_dpo_recovery_long.py)、[DPO 训练与审计契约](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/dpo_recovery_training.py)。
2. 以已完成的加权 chosen-only SFT 为比较对象，匹配它的父模型 chosen-only-sft-v1-20261102，SHA 078182…；新 DPO 的冻结 reference 也取此父模型。这个父模型已有 SFT255 / DPO11。比较对象 final 是 f8fb95…，SFT513 / DPO11。
3. 建议复用 169 对已审计偏好、困难原始权重 4、其余 1、全数据均值归一；对应 6 epochs、每批 4、258 次更新、1014 次曝光。DPO 对每对损失加权，保留 sequence-sum log probability 目标；chosen-only 对照使用 token-mean CE。记录差异和实际计算量。
4. 新建优化器，重新缓存 078182… 的参考分数。历史学习率 5e-6、beta 0.1 可作候选起点；最终学习率 / beta 及最多两个开发配置在新 config 冻结。现有 verify_update_history 对 source_dpo_updates=11 和 169 对来源有硬编码，移植时同步核对 source_sft_updates 与新父模型，审计检查真实继承链。
5. 评测当前共同父模型、加权 chosen-only SFT、新 DPO：同一次运行内、相同任务与种子，汇报完整成功和完整恢复。已有标准恢复探针仅看首动作；要额外复用 /扩展完整续跑评测来满足新验收指标。现有四 test 任务已多次诊断，最终新任务族先冻结再使用。
6. 本地完成有意义的测试与实际数据 / checkpoint 预检，再创建唯一新 run_id / 输出目录 / ZIP。用户执行云端后，本地使用包内冻结源码重放、核对摘要和产物。
7. 补验后，解释 chosen/rejected 概率、相对 margin、错误类别、数据覆盖、权重和训练量的关系。收益接近或有限也按实际结果结束该阶段，随后进入 GRPO。

新 DPO 若确为上述 258 步建议配置，其履历应为 SFT255 / DPO269；这是计划的计数关系，不是已经存在的新模型。当前没有可直接发送的补验命令或新运行包。

## 3. 项目目录、硬件和模型身份

| 项目 | 当前值 |
|---|---|
| 本地项目根 | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab |
| V2 主目录 | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2 |
| 本地 Python | C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/.venv/Scripts/python.exe |
| 本对话 outputs | C:/Users/app/Documents/Codex/2026-10-30/new-chat/outputs |
| 云端 V2 | /root/rivermind-data/AgenticPostTrainingLab/v2 |
| 云端 Python | /root/rivermind-data/envs/apt-vllm/bin/python |
| 云端基础模型缓存 | /root/rivermind-data/AgenticPostTrainingLab/v2/.cache/qwen2.5-1.5b-instruct |
| 云端 GPU | NVIDIA L20，约 44.39 GiB，计算能力 8.9，BF16 可用 |
| 已验证版本 | Python 3.11.17；torch 2.8.0+cu128；vLLM 0.11.0；Transformers 4.57.6；PEFT 0.17.1 |

当前硬件依据 [L20 迁移复核](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/l20-environment-review-20261101.md)。原 4090D 没有资源后，用户将数据克隆到 L20；旧交接中相反方向的早期迁移记录是历史。基础模型缓存和已成功训练产物继续复用。

基础模型 Qwen/Qwen2.5-1.5B-Instruct；固定 revision：989aa7980e4cf806f80c7fef2b1adb7bc71aa306。主权重 3,087,467,144 bytes；SHA256：dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee。完整基础权重主要在云端，回传 review 通常省略。

LoRA：q_proj/v_proj，r=8、alpha=16、dropout=0；112 个张量，1,089,536 参数。vLLM 负责 BF16 生成；HF / PyTorch / PEFT 负责 FP32 评分、反向和 LoRA 更新。当前未采用 TRL 托管训练。

探针事实：512 token 的 FP32 策略 + 冻结 reference 真实更新峰值 allocated 14.4072 GiB；8192 是已通过的推理上下文上限。实际 SFT / DPO 使用最长约 1.4k 的训练样本和 2048 限制，训练显存约 6.4～6.6 GiB；完整长上下文 RL 训练需重新测量。

Git 当前核对 HEAD 为 eda30590b021798edf8c6209f732917aabeeb38e，大量后续文件 / v2 未提交；项目成果以文件、发布清单和 review 身份为依据，保持现有工作树。新对话的实际写权限由其运行环境确定；本次仅获得 reports、handoff、PROJECT_REQUIREMENTS.md 的临时写权限。

## 4. 主体框架：任务、工具、证据、执行

~~~text
Task / 业务规则 / SQLite 数据
  → SYSTEM + 用户任务
  → 模型输出一个严格 JSON 工具动作
  → Environment 执行、验证、返回 observation
  → observation 进入后续上下文
  → 模型重试 / 修复 / 继续取得证据
  → submit 结构化值与证据
  → 最终校验、轨迹、token / 概率与重放
~~~

当前数据库：orders(order_id, order_month, region, segment, status, gross_cents)；refunds(refund_id, order_id, refund_month, refund_cents)。金额均为整数 cents。

业务口径：当月净额是指定 region/segment 的 paid 订单按订单月份累计收入，减去该月发生的退款；退款关联原 paid order 的 region/segment，原订单可发生在其他月份。cancelled 排除。change_cents = current_net_cents − prior_net_cents。

当前 make_tasks 固定 train12、development4、test4，Task.max_turns=12。实例的 family/seed 分别取 1000+、2000+、3000+；约一半任务首个查询注入一次临时故障。同业务模板实例的隔离与新业务任务泛化在报告中分别描述。增加配置 task_count 只选择现有实例，扩任务需修改 / 新建生成逻辑；仅增加 max_turns 是增加预算。

Task.public 隐去 seed/split/family_id/fault_mode；实际 runtime 用户请求是 Task.request。模型只见任务、规则/schema 及工具观察。脚本专家是按公开任务字段、规则和观察生成动作的程序，提供正确示范。

| 工具 | 行为 |
|---|---|
| read_policy | 读取 revenue-v1 业务口径 |
| inspect_schema | 获取真实表结构 |
| query_sql | 执行只读 SQL，返回 rows、evidence_id、verified_metrics |
| calculate_change | 用已验证月度证据 ID 计算差额并登记证据 |
| check_result | 检查 policy/schema、三个值和证据绑定 |
| submit | 校验通过后结束 episode 并输出 artifact |

动作顶层严格为 tool 和 arguments，拒绝重复 JSON key。工具观察包装成 user message 的 observation JSON；该 user 内容是环境反馈，不是新的人类反馈。SFT、DPO 和新采样保持同一模板与内容序列。

SQL authorizer 限制只读表与函数，最多 5000 字符 / 20 行并有虚拟机执行预算。环境在原库验证后，还在两个金额独立扰动世界检验返回整数、正确指标 alias 和 orders/refunds 来源。

**证据**是环境中的来源登记。qN 来自 query_attempts 计数，临时故障也增加查询计数；cN 来自当时证据登记数量。编号可因轨迹不同变化。每个 ID 映射到本次调用实际已验证的 metric→value。SQL 执行成功但 verified_metrics=[] 时可以得到 qN，其登记内容为空，仍需重查。calculate_change 要求两个引用分别拥有对应已验证指标。最终提交绑定值、指标和来源，并要求规则 / schema 已读。

专家标准路径为 7 动作，临时故障任务通常 8 动作。专家先 check_result，当前环境成功规则实际允许直接正确 submit。恢复字段 recovered 定义为最终成功且工具错误后取得新已验证进展；报告的“标准恢复首动作有效”则只表示局部指标进展或检查通过，两者按各自分母分列。

## 5. 数据、训练目标与场景标签

### 5.1 SFT 与 chosen-only

首轮示范使用 12 个 train 任务 × expert/sql_repair/evidence_repair 三种变体，形成 36 条成功轨迹、252 个准入动作。错误动作保留在后续上下文，直接监督的 response 只取正确准入动作。

encode_example 使用真实 tokenizer chat template，验证 prompt 前缀一致、response 包含 EOS、长度未超限；prompt label=-100。action_loss 只监督 assistant response 的下一 token CE，用 logits_to_keep=n+1 对齐响应后缀。训练目标为动作内 token mean，再按 batch 样本平均。

SQL 加权续训将 query_sql 动作原始权重设为 4，其余 1，均值归一；沿用原数据并增加训练量，Adam 63→126。

chosen-only SFT 从偏好数据投影 chosen，保留逐行权重。当前 169 行对应 123 组不同上下文/chosen，重复保留是为匹配 DPO 数据分布。其损失使用 chosen CE；rejected/reference 不进入 SFT 损失。

### 5.2 DPO 与恢复偏好

DPO 用同一上下文的 chosen/rejected，按响应所有 token log probability 求和，再计算相对冻结 reference 的 margin 与 -logsigmoid(beta*margin)。reference 完全冻结。原生 rejected token 保持原采样序列和 EOS，chosen 由模板编码；检查 native prompt、解码和长度一致。

最初 44 对由冻结 SFT 的 train 随机轨迹真实失败构造：在动作前状态生成脚本纠正动作，单独重放并验证其有进展，组成同上下文偏好。临时服务故障不作为模型错误偏好。

恢复数据另主动注入错误前缀并真实执行，构造 12×7=84 个 train 上下文；再采样模型动作，经验证仅保留失败 rejected。最终 169 对 = 原44 + 脚本失败84 + 原生模型失败41。chosen 仍是已验证的脚本反事实动作；失败候选被过滤，不自动递归变成新偏好。

scenario 是 build_context 主动构造时写入、make_pair 继承的标签，非事后模型分类。当前 169 行分布：

| scenario | 行数 | 困难加权 |
|---|---:|---|
| 原始 SQL 偏好，无 scenario | 44 | 1 |
| format_repair | 16 | 1 |
| unknown_tool_repair | 12 | 1 |
| sql_unverified_requery | 24 | 4 |
| repeat_evidence_error | 12 | 4 |
| stale_evidence_repair | 13 | 1 |
| repeat_sql_error | 24 | 4 |
| failed_check_repair | 24 | 4 |

困难行共84，其余85；均值归一后困难权重总占比约79.8%。这里的84个困难样本与84个脚本 rejected 是两种不同统计维度。backward_batch 使用 sum(weight_i*loss_i)/batch_size，均值归一基于全数据。

恢复数据 SHA256：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。最初44对 SHA256：508464a0314982c7e6ade95c57afa3be081afc80f237e4fb20c8abe2079ffb78。

已审计恢复数据路径：[原始169对](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-collect-v1-20261102/extracted/dpo-recovery-collect-v1-20261102/pairs.jsonl)；[项目数据集索引目录](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/datasets/dpo-recovery-v1-20261102)。

训练契约目前固定169对、来源组成和 train身份。build_context 已允许显式 train/development/test 以支持评测，但默认 train，make_pair 的默认 assess 和训练校验仍为 train；修改任务扩展时保持训练准入与评测分支清楚。

## 6. 已完成工作和主要结果

所有“提升”优先引用同一次运行内的配对结果。历史各轮随机种子、采样条件不同，保留其原始分母；贪心相同检查点也曾跨运行出现少量文本变化。

| 已完成阶段 | 主要结果 | 当前用途 |
|---|---|---|
| 原始1.5B baseline | dev0/4；格式26/48；无有效业务指标 | 早期基础能力基线 |
| SFT pilot，实际63步 | train0/12；格式144/144；SQL仍未验证 | 协议学习与SQL瓶颈诊断 |
| SQL加权续训，累计SFT126 | train12/12、dev4/4；后续test4/4 | 历史冻结纯SFT基线 |
| 首轮DPO，新增11步 | train随机26/48→28/48；独立随机test12/32→13/32 | 已完成、增益有限的DPO对照 |
| 恢复DPO一遍，累计DPO54 | 同轮dev随机10/16→12/16；标准恢复29/84→31/84 | 恢复专项初步实验 |
| 恢复DPO三遍，累计DPO140 | 同轮dev随机11/16→11/16；恢复29/84→34/84 | 增加训练量实验 |
| chosen-only SFT，累计SFT255/DPO11 | 同轮三模型完整dev均11/16；恢复起点29/84、DPO35/84、chosen37/84 | 匹配数据和129更新的直接对照 |
| 训练/开发上下文诊断 | 训练上下文同样未掌握关键SQL恢复；工具选择、证据ID和SQL内容分解 | 定位训练信号不足与具体错误 |
| 困难恢复加权SFT，累计SFT513/DPO11 | dev随机11/16→15/16；dev恢复随机37/84→74/84 | 当前强候选 |
| 当前候选独立测试 | 随机24/32→28/32；标准恢复36/84→66/84；组合完整恢复0/32→0/32 | 最新完成的评测与瓶颈 |

chosen-only 直接对照中，DPO 与 chosen 完整dev均11/16；DPO完整随机轨迹证据错误7次，chosen25次；恢复有效动作DPO35/84、chosen37/84；重复上一动作DPO10/84、chosen15/84。SFT新增2次有效检查修复。可据此解释各自行为特点；加权SFT后续还新增258步，因此其收益按整套方案描述。

训练/开发诊断的1344个首动作本地重放通过。加权前关键SQL目标验证在train/dev均为0；已有正确证据时ID选择通常较好。加权后dev贪心工具选择28/28、目标SQL验证12/12；剩余贪心失败主要是检查时沿用被故意加1的差额。

### 6.1 最新独立评测细节：已复核

运行 hard-recovery-independent-test-v1-20261103。before为 chosen-only 078182…，after为困难加权 f8fb95…。四个固定 test任务，各模型4次贪心、32次随机（温度0.8）。另外36个错误上下文：28个标准、8个新增双指标组合；共288个首动作，新增组合做64条完整续跑。

| 指标 | before | after |
|---|---:|---:|
| 贪心完整成功 | 4/4 | 4/4 |
| 随机完整成功 | 24/32 | 28/32 |
| 随机工具错误，含预设暂时故障 | 65 | 32 |
| 随机证据错误 | 29 | 12 |
| 标准恢复贪心有效首动作 | 12/28 | 26/28 |
| 标准恢复随机有效首动作 | 36/84 | 66/84 |
| 新增组合完整恢复，贪心+随机 | 0/32 | 0/32 |

完整随机配对6改善、2退化。候选剩余失败四条均未验证出任何指标：test000/rollout4与test001/rollout7使用PAID替代paid；test001/rollout4退款列别名错误及遗漏指标alias；test002/rollout5聚合/月份口径错误、遗漏alias并尝试常量查询。

候选标准随机恢复各场景（每项12次）：format12、unknown12、sql_unverified11、repeat_evidence11、stale_evidence10、repeat_sql7、failed_check3。贪心检查修复2/4，另两次ID正确但差额仍错1。

新增 both_sql_unverified 与 both_sql_unverified_repeated_evidence：两个指标的gross-only查询已执行但都未验证，后一类再加入两次无效证据计算。候选改为query_sql的首动作达到贪心7/8、随机20/24，但全部未验证；所有续跑均无新验证指标。典型错误是查询不存在的current_net_cents列、错误退款列，或只返回月份/订单ID等辅助数据。专家对8个上下文都可在5个后续动作内完成，处于相同12步总预算。

原始 complete+continuation 共1012个模型动作，1006次stop、6次length；6次截断集中在候选2条组合随机轨迹，均达到256token。288个首动作全部stop。首动作中64条与对应续跑共享同一生成，计数不相加冒充独立样本。

详见 [最新独立评测报告](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/hard-recovery-independent-test-v1-review-20261103.md)、[本地审阅摘要](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/hard-recovery-independent-test-v1-20261103/local_review_summary.json)、[轨迹行为分析](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/hard-recovery-independent-test-v1-20261103/behavior_analysis.json)。

## 7. 模型履历、身份和保存位置

~~~text
Qwen2.5-1.5B base
  → SFT pilot63
  → SQL weighted SFT126（纯SFT冻结基线）
  → DPO11（SFT126/DPO11）
      ├─ recovery DPO一遍：SFT126/DPO54
      ├─ recovery DPO三遍：SFT126/DPO140
      └─ chosen-only SFT三遍：SFT255/DPO11
          └─ hard recovery weighted SFT六遍：SFT513/DPO11（新GRPO共同起点）
~~~

恢复DPO三遍从DPO11独立开始，第一遍权重与单遍最终相同；不是从DPO54再任意接三遍。chosen-only和恢复DPO均fresh Adam。SQL加权63→126明确继承原Adam。未来新分支的优化器和reference由新配置明确建立。

| 标识 | 完整 adapter_model.safetensors SHA256 | 履历 |
|---|---|---|
| SFT126 | e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0 | SFT126/DPO0 |
| DPO11 | fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d | SFT126/DPO11 |
| DPO54 | 933ff3f97674d8b7a9bbafe6bc81292b8c0608ba7205cca3842f35559989820f | SFT126/DPO54 |
| DPO140 | 7393906da901516cdfa05b3f987808a604c6f49c438a410a8d5e5dc06a9c0725 | SFT126/DPO140 |
| chosen SFT | 078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55 | SFT255/DPO11 |
| hard weighted SFT | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f | SFT513/DPO11 |

本次重新计算上述6份实际adapter哈希，均匹配complete.json。各模型的本地路径：

- [纯SFT冻结索引](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/baselines/current_sft_baseline.json)；[纯SFT快照adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/baselines/sft-sql-weighted-v1-20261102/adapter)。
- [DPO11 adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-train-v1-20261102/extracted/dpo-train-v1-20261102/final/adapter)。
- [DPO54 adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102/extracted/dpo-recovery-train-v1-20261102/final/adapter)。
- [DPO140 adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v2-20261102/extracted/dpo-recovery-train-v2-20261102/final/adapter)。
- [chosen-only adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102/final/adapter)。
- [当前恢复加权adapter](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/hard-recovery-sft-weighted-v1-20261103/extracted/hard-recovery-sft-weighted-v1-20261103/final/adapter)；[候选登记](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/candidates/hard-recovery-sft-weighted-v1-20261103/review.json)。

云端真实训练产物通常位于 /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/<run_id>/final/adapter。纯SFT freeze快照创建在本地，云端可复用 sql-sft-weighted-v1-20261101/final。读取云端前由prepare核对实际存在和哈希。

评测review包经常省略基础模型、Adam state.pt和中间adapter；云端保留状态，本地按返回内容审计。候选登记目录有时只有review.json，实际权重优先按上面的review/extracted路径寻找。最早pilot解压仍在 C:/Users/app/Documents/Codex/2026-10-30/new-chat/work/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030；其项目review目录只保留ZIP和审计索引。

## 8. 奖励、信用与现有 RL 内核：接 GRPO 前必读

Environment现有奖励：每步 outcome_reward = 1{本步成功} − 0.01；Phi=0.2×已验证指标数；process_reward=Phi(next)−Phi(current)，终局next Phi设0；potential模式用outcome+process。gamma=1完整过程奖励和为0，重复已验证指标不增加Phi。

rewards.assign_turn_credit 对同任务至少2条完整轨迹，以其他轨迹平均outcome总分作LOO baseline；各动作 advantage = shaped return-to-go − 该baseline。当前没有按组标准差归一化，没有正式GRPO零方差组策略。分组独立性、采样新鲜度由未来collector保证。

trainer.py已实现：
- response_scores 对动作response做causal对齐；FP32 log_softmax；当前会先物化整个序列词表logits，长上下文内存应优先检查，可参考SFT/DPO的后缀logits_to_keep。
- prepare_scores 冻结reference，按同一HF路径缓存detached old/ref；更新前old/fresh最大差门槛1e-5并绑定prompt/response。
- update 只训练LoRA；PPO ratio=exp(new−old)，clip0.2；TIS=min(2,exp(old−q))并detach；sampled k3 KL系数默认0.02，属于采样估计。
- 动作内token mean，动作累加后除以episode数量；当前不再按轨迹轮数平均。不同长度分支的损失归一化需明确，防止把额外动作权重混作信用收益。
- 每动作反向累积，最终一次optimizer.step，max_grad_norm1。平均action KL>0.05或clip_fraction>0.5时清梯度、updated=false，更新计数应按真实step记录。

四类概率：q是实际生成行为分布；p_old是采样策略在训练侧重算；p_new是更新中的策略；p_ref是冻结参考。生成端BF16、训练端FP32可能有差异。

现有DPO collector记录mode=dpo_sample、raw_token_log_probs，behavior_token_log_probs=None；RL training_samples要求mode=sample、原生prompt/response、每token行为logprob。温度0.8的raw概率和真实采样概率有区别，不能只改mode就准入RL。首次RL可考虑温度1、top_p1、关闭top_k以简化行为分布，仍需验证引擎返回的概率语义和HF差异。

正式GRPO待办：组大小K和有效轨迹准入；轨迹级组内优势、标准差/零方差规则；有限训练调度；checkpoint绑定采样；old/ref缓存；更新后刷新vLLM；完整Adam/RNG/计数恢复；before/after与控制分支评测。小型PEFT更新和硬件探针已有，生产闭环待实现。

细粒度第一版按用户已确认方案只做可验证进展＋逐轮return/advantage。潜势塑形是一个可解释起点；其gamma1 return中含状态潜势项，不能仅靠总分或公式认定信用有效，要用实际更新和同预算对照验证。定义基线、归一化、KL和可见信息时尽量保持其它条件一致。局部恢复RL、更多credit算法是后续可选。

## 9. 收尾 Phase：已完成、待完成和验收

| Phase | 内容 | 状态 / 完成条件 |
|---|---|---|
| V2-0 | V1封存、模型迁移与固定身份 | 已完成，V1按需追溯 |
| V2-1 | SQLite工具环境、证据、奖励、runtime、重放 | 已完成并持续验证 |
| V2-2 | 4090D/L20验证、原始baseline | 已完成，当前L20 |
| V2-3 | 专家/恢复SFT、SQL加权、冻结纯SFT | 已完成 |
| V2-4 | DPO采集、首轮/恢复训练、chosen对照、困难加权SFT | 已完成；还需C1限定补验 |
| C1 | DPO补验与逐因诊断 | 补验表现未通过；E1已复核，接续E2均匀权重对照，GRPO暂缓 |
| C2，第3～5天 | 结果奖励GRPO在线闭环 | 未完成；至少采样→更新→刷新→再采样和冻结前后评测 |
| C3，第6～7天 | 细粒度GRPO、单一SQLite依赖扩展 | 未完成；逐轮信用进入损失，记录实际依赖与对照结果 |
| C4，第8～10天 | 新任务族正式评测与交付 | 未完成；曲线/表/案例/复现入口/简历讲解齐全 |

第7天冻结新增功能，最后3天以评测和交付为主。首轮单训练种子，关键结果优先复验；更多种子按时间商议。训练时限和扩展边界共同讨论，不以无限调参追逐正结果。

长程扩展沿用SQLite，前一阶段已验证结果参与后续核对，记录专家最短路径、依赖深度、有效动作和上下文。多个月份独立查询可能被一个SQL合并，设计时检查是否确实增加依赖深度。选一种业务流程做清楚，不开启文件平台重建。

最终主表分开两类比较：历史/补验DPO与chosen-only的同起点对照；当前f8fb95共同起点的GRPO outcome / fine-grained对照。比较GRPO与历史DPO时同时列起点与累计训练量。用户希望GRPO争取更好；完整可信实验和按时投递是收尾条件。

## 10. 评测与审计关键环节

1. 训练数据仅来自train，development选配置，最终新test先冻结。test-000～003已多次评测并做失败分析，后续作为历史回归集；新测试按任务族/模板/派生数据隔离，重复rollout按任务聚合。
2. 始终分列最终成功、完整恢复、局部首动作有效、工具选择、ID选择和SQL验证。已有标准7类恢复主要单步探针，新增2类有完整续跑；扩成统一完整恢复评测需要真正实现。
3. primary outcome保持统一业务校验；同一轮按任务和rollout/seed配对，保留两模型都成功/都失败、改善、退化。历史不同种子结果各自记录。
4. 原SQL金额扰动保持订单/退款月份关系，曾发现某错误月份过滤偶然命中同一退款，照样通过。最新独立评测补充两个金额＋退款月份重排世界，并追溯最终使用证据；未来新评测继续保留或独立升级校验版本。
5. 最新跨月补查：完整任务父模型56条、候选64条已验证SQL全部通过；恢复首动作父1条、候选43条全部通过。此项是SQL重执行语义校验，未重新让模型在重排数据库中生成轨迹。
6. review先CRC/路径检查，按run_id保留原ZIP、安全解压；验证冻结source、input_binding、模型/数据/tokenizer哈希；再导入包内source重放，重算summary/CSV。修改当前源码后，历史包按自己的快照审阅。
7. 最新独立审阅重放72完整episode/596步、288首动作、64续跑/416步；81源码、16溯源文件一致。本地SQLite3.42对父模型同一错误SQL的两次调用报near order，云端报near dot；加载已安装SQLite3.53.1后严格逐字段重放通过，原始结果保留。
8. 该兼容审阅入口为 [独立复核脚本](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/diagnostics/work/review_hard_recovery_independent_test.py)；其本地DLL路径为 C:/Users/app/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/DLLs/sqlite3.dll。新环境按实际SQLite版本处理，先查真实差异，再选择兼容运行时，不全局放宽错误匹配。
9. 最新发布测试证据：新增9项通过；全回归初次128通过、1旧断言失败，修复错误提示包含train后两项定向复测通过；compileall通过。receipt记录129项覆盖，不记作同一次129全绿。本次交接没有另跑训练或全测试。
10. 历史SFT诊断audit失败已定位：Generation.mode覆盖base/sft分支标签。原GPU记录保留失败，本地恢复副本只修分支元数据并独立复核；后续已修实现，当前无需重跑该诊断作业。

## 11. 云端运行、本地复核与文件约定

本地源码在项目v2；review通常位于 C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/<run_id>/extracted/<run_id>/。其中source保存运行时快照，local_review_summary等位于其上层review目录。

运行包内部通常含v2/。用户默认上传至云端v2目录，解压应采用“云端v2内ZIP的绝对路径”，目标设为 /root/rivermind-data/AgenticPostTrainingLab。随后进入v2并用 /root/rivermind-data/envs/apt-vllm/bin/python -u 执行新runner。前次相对路径unzip失败的原因是命令cd到了项目父目录而ZIP仍在v2；继续使用绝对ZIP路径。

新包准备完成后直接给完整命令和预期回传名。runner使用唯一输出目录，实时显示训练与评测进度，超时/失败也打包日志和状态，保护已存在结果。当前新DPO包尚未创建，旧run_v2_dpo_recovery_train_long.py不能冒充新补验入口。

本地exec_command曾因Windows进程创建失败不可用，可用node_repl的node:fs读取、node:child_process调用项目Python做核验；优先rg搜索。当前会话node_repl对项目外写入可能不继承临时权限，原生apply_patch已能在获批路径更新文档。下一会话按实际工具与权限处理，保持原文件和新输出清楚。

输出原件存v2/reports和v2/handoff；当前桌面会话另镜像至本对话outputs以提供可点击链接。新对话按其工具/渲染要求给链接，不自动打开报告。日志不输出凭据，云端访问URL不是远端shell连接。

## 12. 核心代码阅读与接续地图

通用runner按prepare/train或sample/evaluate/audit/package划分；用户已读过多数已有SFT/DPO文件，后续先解释新增差异。

| 接续触发条件 | 文件与重点 |
|---|---|
| 核对任务和口径 | [tasks.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/tasks.py)：Task、make_tasks、connect、metric_sql、truth |
| 核对工具和证据 | [environment.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/environment.py)：execute_readonly、_certify_sql、_call、_check、step |
| 核对消息和采样对象 | [runtime.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/runtime.py)：SYSTEM、Generation、parse_action、run_episode |
| SFT编码和mask | [sft.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/sft.py)：build_dataset、encode_example、action_loss |
| DPO损失与首轮偏好 | [dpo.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/dpo.py)：corrective_action、verified_pairs、encode_pair、response_logps、dpo_loss |
| 受控恢复与来源 | [dpo_recovery.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/dpo_recovery.py)：build_context、assess、make_pair、encode_recovery_pair |
| DPO补验复用 | [dpo_recovery_training.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/dpo_recovery_training.py)、[train_v2_dpo_recovery_long.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/train_v2_dpo_recovery_long.py)：批次、参考缓存、继承与更新记录 |
| 加权对照复用 | [hard_recovery_sft.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/hard_recovery_sft.py)、[train_v2_hard_recovery_sft.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/train_v2_hard_recovery_sft.py) |
| 工具/ID/SQL分解 | [recovery_diagnostics.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/recovery_diagnostics.py)：diagnose_action、diagnostic_contexts |
| 独立完整恢复与SQL结构 | [independent_recovery_test.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/independent_recovery_test.py)、[audit_v2_hard_recovery_test.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/audit_v2_hard_recovery_test.py)：原生续跑、重放、calendar audit |
| 新GRPO核心必读 | [rewards.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/rewards.py)、[trainer.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src/apt_v2/trainer.py)：q准入、逐轮信用、old/ref缓存、PPO/TIS/KL与归一化 |
| 运行与失败打包 | [streamed_process.py](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/streamed_process.py)、[已有流式总入口示例](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts/run_v2_hard_recovery_sft.py) |
| 改动后的关键行为测试 | [基础环境/RL内核测试](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/tests/test_agentic_v2.py)、[恢复续跑与SQL结构测试](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/tests/test_hard_recovery_independent_v2.py)、[DPO多轮更新记录测试](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/tests/test_dpo_recovery_long_v2.py) |

## 13. 按需证据索引与 V1 简史

### 当前接续最重要的报告

- 决定新补验前读：[匹配DPO/chosen-only对照](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/chosen-only-sft-v1-review-20261102.md)、[困难加权SFT结果](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/hard-recovery-sft-weighted-v1-review-20261103.md)。
- 分析当前瓶颈读：[训练/开发恢复诊断](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/recovery-context-diagnostics-v1-review-20261102.md)、[最新独立结果](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/hard-recovery-independent-test-v1-review-20261103.md)。
- 追溯旧纯SFT / 首轮DPO读：[纯SFT独立评测](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/sft-independent-test-v1-review-20261102.md)、[首轮DPO独立评测](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/dpo-independent-test-v1-review-20261102.md)。
- 追溯169对数据读：[恢复偏好数据审阅](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/dpo-recovery-collect-v1-review-20261102.md)。
- 完整复核与日程读：[最新收尾计划](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/reports/project-roadmap-v2-20261103.md)、[本次交接核验回执](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/handoff/PROJECT_HANDOFF_V2_2026-11-03_RECEIPT.json)。

### V1 只保留接续所需背景

V1为SmolLM2-135M-Instruct的单轮算术/JSON/括号后训练学习阶段，已有真实LoRA SFT/DPO/GRPO及采样概率、检查点与诊断对照。SFT对严格结构化输出有效；后续DPO/GRPO最终开发收益有限，平衡示范加GRPO出现1个开发题改善，未晋升冻结基线。用户决定封存并转入V2。

根目录旧sft-v2、dpo_v2是135M时期版本名；与当前v2/1.5B不同。V1的adapter、Adam与rollout缓存保留历史用途。V1单轮GRPO已跑过，V2生产Agentic GRPO仍待实现，状态分开描述。

归档：C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029.zip；历史记录SHA256为6348f933a916f976eb16670caff132a54cbf09710e99a3a7d86e2a73dfbec500，共2015个原文件。本次交接未重新核验整个大归档；2026-10-30原文记录已做逐项核对。只有追溯旧实验时再读 [V1 History](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/handoff/PROJECT_HISTORY_V1_2026-10-30.md)；原架构背景见 [旧V2交接](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/handoff/PROJECT_HANDOFF_V2_2026-10-30.md)。

## 14. 本次交接的更新范围与接续提示

本次更新当前PROJECT_REQUIREMENTS、收尾计划及同日roadmap指针，新增本交接和核验回执；原V1、旧V2交接、模型、数据、实验快照和结果保留。镜像副本位于当前chat outputs。

重新核验6份adapter、169对数据身份、关键配置/报告/代码路径，并检查本文件中的本地链接。机器回执记录每项状态。仅整理文档，训练与optimizer更新数为0。

新对话可直接这样开始：

> 已阅读最新V2交接和收尾计划。先核对当前文件及权限，准备一次与已完成困难加权chosen-only SFT匹配的DPO补验：使用其相同父检查点、169对数据、困难权重与曝光，最多两个开发配置；完成本地测试和数据/模型身份预检，再提供云端运行包与命令。补验后分析效果与原因，进入以f8fb95…为共同起点的结果奖励/细粒度GRPO。继续遵守约10天收尾、第7天冻结新增功能和GDPval可选的范围。
