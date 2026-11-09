# AgenticPostTrainingLab V2：DPO 收尾与 GRPO 接续交接

更新：2026-11-05，Asia/Shanghai。面向下一对话。本文整合原始项目交接、已确认收尾计划、当前源码和本轮已复核实验。当前状态以本文和最新用户指示为准；历史报告中的“下一步”按当时进度理解。

## 1. 接续摘要：先读这一节

**DPO 阶段已结束，下一对话进入 C2：结果奖励 GRPO。** 最终短程 DPO 第12步达到预先固定的“与最佳加权 SFT 接近”标准。用户已明确将下一阶段交给新对话。本次只整理交接，没有启动或发布新的 GRPO 云端作业。

- 项目主体：Qwen2.5-1.5B-Instruct + LoRA，真实多轮 SQLite 业务对账 Agent；核心是工具执行、证据绑定、错误恢复与逐轮信用。
- 两条 GRPO 分支共同起点：困难恢复加权 SFT，adapter SHA256 为 f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f，累计 SFT513 / DPO11。
- 最终 DPO12 为阶段比较候选，累计 SFT513 / DPO23。GRPO 继续使用上面已确认的加权 SFT 起点。
- V2 环境、SFT/DPO、完整恢复评测和严格重放已具备；正式 GRPO 分组采样、组内优势、零方差规则、在线刷新与完整恢复流程仍待实现。现有 trainer.py 是可复用的 PPO surrogate 内核。
- 下一步顺序：核对起点和运行环境 → 冻结首版 GRPO 算法契约 → 完成本地行为测试 → 云端最小真实闭环预检 → 有限预算训练与配对评测 → 本地复核。第8节给出完成条件。
- E1–E4 退化机制排查作为已完成扩展成果保留，详见附录A；四分支末期推理确认包仅已准备，未执行，属于可选扩展。
- 报告用中文，直接陈述结果与证据；对话简要写主要结果并给链接，文件不自动打开。云端仍由用户上传 ZIP、执行命令、返回 review ZIP。

阅读顺序：本节 → 第2节阶段安排 → 第3节身份 → 第7–9节 GRPO 与评测 → 第10节运行/审计。需要理解环境时读第4–5节，需要复述 DPO 实验时读附录A。

## 2. 用户已确认的范围与 Phase

目标是形成可投递的大模型后训练 / 强化学习算法实习项目，其次兼顾 Agent 开发。原收尾安排约10天、平均每天6小时，实施第7天冻结新增功能、最后3天用于评测和材料。这是相对工作日规划；DPO排查已经扩展，接续时按剩余实际时间整理进度，不将旧表的日期当成新的强制日历承诺。

| Phase | 内容 | 当前状态与完成条件 |
|---|---|---|
| V2-0 | V1封存、迁移到1.5B、固定身份 | 已完成；V1只作历史参考 |
| V2-1 | SQLite任务、工具、证据、runtime、奖励、重放 | 已完成，后续扩展继续保留契约 |
| V2-2 | 基座评测与GPU验证 | 已完成；最新已观测平台L40S |
| V2-3 | 专家SFT、SQL加权、纯SFT冻结基线 | 已完成 |
| V2-4 | DPO、恢复偏好、chosen-only对照、困难加权SFT | 已完成 |
| C1 | DPO补验、E1–E4排查、早期比较、短程收尾 | 已完成：DPO12通过固定开发集实用收尾门槛 |
| C2，原第3–5天 | 结果奖励GRPO在线闭环 | 待实现/运行；实际采样→更新→刷新→再采样，冻结前后配对评测 |
| C3，原第6–7天 | 一种细粒度GRPO + 一种SQLite依赖扩展 | 待实现/运行；信用进入逐动作损失，同预算对照，证明任务真实依赖 |
| C4，原第8–10天 | 新任务族确认、必要复验、交付 | 待完成；复现代码、曲线表格、案例、报告、简历和讲解齐全 |

必做范围：结果奖励 GRPO、一套“可验证进展 + 逐轮 return/advantage”细粒度方法、一种轻量真实依赖业务扩展、完整对照与交付。GDPval、多文件办公平台、网页界面、额外算法、局部恢复RL和更多种子为可选；multi-OPD已排除。演示采用报告、终端或录屏。

首轮可用单训练种子，优先复验关键结果。总训练次数没有固定“6次”上限；每个作业都应有明确步数、时限、停止标准和失败回传。近期约4小时是单作业基础上限，先profile再确定新配置。重大预算或范围变化再与用户商议。用户已经确认总体方向，接续从实现推进，无需重做规划访谈。

## 3. 路径、运行环境与模型身份

本文所有项目相对路径均以以下V2目录为根：

- 本地项目：C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2
- 本地Python：C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/.venv/Scripts/python.exe
- 本次工作目录：C:/Users/app/Documents/Codex/2026-11-03/dpo；临时材料在work，交付镜像在outputs。
- 云端项目：/root/rivermind-data/AgenticPostTrainingLab/v2
- 云端Python：/root/rivermind-data/envs/apt-vllm/bin/python
- 云端基座缓存：/root/rivermind-data/AgenticPostTrainingLab/v2/.cache/qwen2.5-1.5b-instruct

最近成功作业记录为 NVIDIA L40S，UUID 92e14e85-628d-e5a4-af16-0db4defb2afb，显存50,867,666,944 bytes，计算能力8.9。Python3.11.17、torch2.8.0+cu128、CUDA12.8、cuDNN91002、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1。历史经历4090D、L20；新作业prepare应采集当次环境并冻结，云端机器可能已变更。

基座：Qwen/Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。主权重3,087,467,144 bytes，SHA256 dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee。完整基座主要保留云端。

LoRA：q_proj/v_proj，r8、alpha16、dropout0，112个张量、1,089,536个可训练参数。生成使用vLLM BF16，评分与训练使用HF/PyTorch/PEFT FP32，当前训练为项目自实现。

### 3.1 主线模型履历

~~~text
固定Qwen基座
  → SFT pilot63
  → SQL weighted SFT126（纯SFT冻结基线，DPO0）
  → 首轮DPO11（SFT126/DPO11）
      ├─ recovery DPO43新增 → SFT126/DPO54
      ├─ recovery DPO129新增 → SFT126/DPO140
      └─ chosen-only SFT129新增 → P：SFT255/DPO11
          ├─ matched weighted DPO258新增 → SFT255/DPO269
          │   └─ 其中第86步候选 → SFT255/DPO97
          ├─ E2/E3/E4独立控制分支（见附录A）
          └─ hard recovery weighted SFT258新增 → W：SFT513/DPO11
              ├─ uniform短程DPO12新增 → SFT513/DPO23（DPO收尾）
              ├─ outcome GRPO（待做）
              └─ fine-grained GRPO（待做）
~~~

P、W名称中的SFT表示该阶段训练方法，两者都继承11次历史DPO。恢复DPO三遍从DPO11独立开始；其第43步与单遍最终一致。GRPO两支分别从W初始化，新增RL更新另记，不覆盖SFT/DPO历史计数。

| 模型 | adapter_model.safetensors SHA256 |
|---|---|
| 纯SFT126 | e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0 |
| DPO11 | fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d |
| P chosen-only SFT | 078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55 |
| W / GRPO共同起点 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| 原matched DPO86 | 43100acde91f178de59ac7a65f29867862a02a7f1a6cbcd134b7d81a7318f46c |
| 最终短程DPO12 | 77c737965c25dfd3110177bb0856938a651cb387bc7153b588b3ce56adf6fc7e |

W实际本地adapter目录：artifacts/reviews/hard-recovery-sft-weighted-v1-20261103/extracted/hard-recovery-sft-weighted-v1-20261103/final/adapter。云端：artifacts/hard-recovery-sft-weighted-v1-20261103/final/adapter。

DPO12实际本地adapter目录：artifacts/reviews/dpo-sft-start-short-v1-20261105/extracted/dpo-sft-start-short-v1-20261105/training/checkpoint-0012/adapter。云端：artifacts/dpo-sft-start-short-v1-20261105/training/checkpoint-0012/adapter。

两者登记分别为artifacts/candidates/<run_id>/review.json。纯SFT冻结索引artifacts/baselines/current_sft_baseline.json，快照artifacts/baselines/sft-sql-weighted-v1-20261102/adapter；W作为已确认GRPO起点保留，历史纯SFT索引独立存在。候选目录通常只有登记文件，实际权重按review路径查找。

本次交接重新读取并计算了W、DPO12、纯SFT实际权重和169对数据哈希，四项均匹配。云端当前状态由下一作业预检确认。项目后续成果大量位于未提交文件，复现以冻结源码清单、配置和review身份为准；保护已有工作树与历史产物。

## 4. 主体框架：任务、工具、证据与执行

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

## 5. 已有训练数据与目标

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


### 5.3 已实现的完整恢复评测

早期标准7类恢复只评下一步有效动作；matched DPO起已经实现标准7类完整续跑。apt_v2/matched_dpo.py的context_environment重建固定development上下文并逐条重放前缀；run_recovery从该状态继续到终止；replay_recovery验证身份、原生生成、逐条环境结果、总终局和artifact。前缀消耗原12步预算，续跑使用剩余预算。新增工具错误扣除前缀错误，total_tool_errors包含前缀。

当前每模型常规开发评测132条：4条greedy完整任务、16条采样完整任务、28条greedy恢复、84条采样完整恢复。恢复84=4任务×7上下文×3次。首动作诊断可来自完整恢复的同一首步，作为解释指标单列。

## 6. 已完成结果与 DPO 阶段最终结论

最早基座dev为0/4，SFT pilot63学会JSON格式但train0/12；SQL加权后累计SFT126达到train12/12、dev4/4，之后历史test贪心4/4。后续恢复专项和chosen-only、困难加权SFT形成W；完整DPO实验链见附录A。

W的历史独立评测hard-recovery-independent-test-v1-20261103：与P比较，完整采样24/32→28/32，配对改善6、退化2；贪心均4/4。标准恢复有效首动作36/84→66/84；新增双指标均未验证的组合完整恢复双方均0/32。W剩余完整任务失败集中在SQL大小写、列名/alias、退款月份口径与聚合；组合恢复中专家可在相同总预算内5个后续动作完成。该结果保留为新GRPO需要关注的瓶颈。

最终DPO作业：dpo-sft-start-short-v1-20261105。W同时作策略起点和新冻结reference，169对池、uniform权重、LR5e-7、beta0.1、batch4、clip1、无辅助CE。预定检查4/12/24步，沿同一Adam和RNG连续训练；实际第12步停止，新增12次更新、48对曝光，累计SFT513/DPO23。

| 同轮指标 | W | DPO4 | DPO12（选定） |
|---|---:|---:|---:|
| 采样完整任务 | 16/16 | 16/16 | 16/16 |
| 采样完整恢复 | 62/84 | 60/84 | 59/84 |
| greedy完整任务 | 4/4 | 4/4 | 4/4 |
| greedy完整恢复 | 20/28 | 19/28 | 21/28 |

运行前固定收尾门槛：采样完整≥15/16且至多落后W1次；采样恢复至多落后4/84；两项greedy成功数均不下降。DPO4在greedy恢复少1次，DPO12全部通过，24步未运行。用户的“DPO更好或差不多即可收尾”目标据此完成。

DPO12与W的采样恢复配对59共同成功、22共同失败、3退化、0改善；greedy恢复20共同成功、7共同失败、1改善、0退化。20条完整任务响应token序列与W对应轨迹相同；真实LoRA已更新，整体相对起点L2变化约0.0273%。最终材料表述为“保守短程DPO保持完整任务表现，恢复处于预定容差内”。本轮是开发集候选选择，最终新任务族确认属于C4。

本地已严格复核147份冻结文件、11份来源文件、真实adapter和参考token绑定，重放396条轨迹/1919个生成动作，重算配对、SQL日历校验、小质量检查和停止选择；云端10阶段成功、约93分钟。state.pt保留云端，返回包包含SHA和Adam/RNG核验记录。review ZIP SHA256：c08bd6f57da8266fa4e2c9be1a588b4ce0877850bf3a4ff94d865854035249f7。

报告：reports/dpo-sft-start-short-v1-review-20261105.md。完整复核：artifacts/reviews/dpo-sft-start-short-v1-20261105/local_review_summary.json、locally_recomputed_selection.json、actual_tensor_changes.json。

## 7. 现有 RL 内核与 GRPO 必须明确的契约

以下实现事实已在本次重新阅读src/apt_v2/rewards.py、trainer.py、runtime.py确认。GRPO具体超参数、组大小、训练日程尚未冻结；本节列出需要在下一阶段落实的设计点。

### 7.1 奖励与优势

Environment当前每步outcome_reward=1{本步成功}−0.01。Phi=0.2×已验证指标数；process_reward=Phi(next)−Phi(current)，终局next Phi=0；potential模式reward=outcome+process。gamma1时完整轨迹过程奖励和为0，重复验证同一指标不增加潜势。

rewards.assign_turn_credit当前要求同task至少2条完整轨迹，以其他轨迹outcome总分均值作leave-one-out baseline，各动作advantage=该动作起的shaped return-to-go−baseline。当前没有组内标准差归一化和正式零方差组策略。这是现成逐轮信用原型，正式outcome GRPO还需单独实现轨迹级组内优势。

首版outcome GRPO需在配置和测试中明确：终局奖励是否沿用步成本；分组键和K；组均值/标准差定义、epsilon与零方差处理；失败轨迹、截断和中断的准入；每组几次更新；KL、clip、损失归一化。若使用典型A_i=(R_i−mean(R))/(std(R)+epsilon)，应将具体实现写入配置与审计，所有动作共享该轨迹优势。全成功/全失败组可能很多，记录组内回报方差和有效训练比例。

细粒度分支按已确认方向使用可验证进展与逐轮return/advantage；需要固定baseline和归一化，使差异集中在信用分配。终局潜势归零使总过程奖励为0，但return-to-go仍包含当前状态潜势项；效果由逐动作优势、实际梯度和同预算对照验证。奖励只使用环境合法状态和可验证信息，模型输入继续隐藏truth和split等内部字段。

### 7.2 四类概率与token准入

| 符号 | 含义 | 记录/使用要求 |
|---|---|---|
| q | 实际生成行为分布 | 原生response token逐token概率，包含真实temperature及采样处理 |
| p_old | 采样checkpoint在训练侧的概率 | 固定token上重算并detach，绑定采样模型版本 |
| p_new | 当前更新策略概率 | 可求梯度，进入ratio |
| p_ref | 冻结reference概率 | 模型与cache固定、detach，进入KL |

现有DPO评测NativePolicy采用raw_logprobs，温度0.8的原始logp与实际行为分布有区别；Generation的behavior_token_log_probs通常为空。rewards.training_samples明确要求mode=sample、原生prompt/response ID、长度一致且有限的behavior logp。旧DPO轨迹作为诊断资料；GRPO训练由新collector生成合格样本。

建议首个最小预检考虑temperature1、top_p1、关闭top_k等额外截断，并实测vLLM概率语义；具体配置仍待下一阶段冻结。BF16生成和FP32训练之间的差异需要记录，不能只将raw字段复制或改mode。EOS、finish_reason、response mask、环境非法JSON动作与推理异常分别处理；完整失败轨迹有训练价值，基础设施中断轨迹应单独保存。

### 7.3 已有更新内核

trainer.response_scores按prompt长度做causal对齐，以FP32 log_softmax取response token分数。当前前向先物化完整序列词表logits，长上下文需要profile；可参考SFT/DPO后缀logits_to_keep优化，并用同token评分一致性测试验证。

prepare_scores冻结reference并缓存old/ref；更新前同HF路径old/fresh最大差门槛1e-5，cache绑定prompt/response ID。update只允许LoRA训练，ratio=exp(new−old)，clip默认0.2；TIS=min(2,exp(old−q))且detach；sampled k3=exp(ref−new)−1−(ref−new)，KL系数默认0.02。

现有loss为动作内token mean，再累加动作并除episode数；没有再按轨迹轮数平均。因此长轨迹贡献更多动作项。两条GRPO应固定相同损失归一化并记录response token/动作数，避免把长度和计算量变化混入信用结论。k3是采样估计，报告与实现保持一致。

每动作反向累积，最后clip_grad_norm1并optimizer.step。平均action k3>0.05或clip_fraction>0.5时清梯度、updated=false。记录真实更新数与尝试次数，跳过更新不增加optimizer计数。当前内核使用model.eval保持dropout行为一致，同时仍启用梯度。

### 7.4 生产闭环仍待补齐

需要实现：同一任务独立多条完整rollout的分组collector；固定checkpoint与sampling seed；old/ref缓存身份；group advantage与零方差处理；有限步数runner；更新后可靠刷新vLLM adapter；刷新后再次采样的模型身份验证；Adam/RNG/组游标/采样版本/更新计数恢复；评测与失败打包。根目录V1曾跑过单轮GRPO，V2当前仍处于上述接续状态。

现有8192是推理上下文通过值；512-token双模型FP32真实更新探针峰值约14.4072GiB；历史SFT/DPO训练样本约1.4k、max_length2048。真实多轮GRPO上下文和组大小需重新测显存、耗时与token吞吐。

## 8. 下一对话的执行顺序与完成条件

本节是落实已确认GRPO方向的实施建议；新算法数值配置在实际预检后冻结。当前没有可直接发送的正式GRPO运行命令或run包。

1. **起点预检。** 读取W候选登记、真实adapter、complete与源配置，核对基座/tokenizer/LoRA身份；采集当次GPU、软件版本和可用显存。完成条件：prepare生成可复核input_binding，实际权重哈希与第3节一致。W的完整任务基线已接近饱和，应同时保留完整恢复和新依赖任务的评测空间。
2. **冻结首版算法契约。** 先完成结果奖励分支，明确第7节所有开放设置；建议以W作初始策略和冻结KL reference、新建Adam，明确历史SFT/DPO计数与新增RL计数。完成条件：配置、优势/损失公式、零方差处理和数据准入相互一致，用户可以直接阅读核心实现。
3. **本地行为测试。** 覆盖组任务身份、优势正负与零方差、prompt mask/response/EOS、q与old/ref绑定、reference冻结、真实LoRA梯度和更新、拒绝中断轨迹、跳过更新计数、保存恢复一致性；对新增奖励验证重复取证、伪证据和终局清零。完成条件：相关测试通过，关键实际数据/adapter预检通过，发布快照可校验。
4. **最小真实GPU闭环。** 先限制任务、组数与更新数，实际采样一组→评分→更新→保存→刷新推理策略→再采样。采样日志必须关联前后checkpoint，检查刷新后加载真实新权重；如完整任务组内无方差，记录事实并按冻结规则处理。完成条件：至少一轮真实有效梯度更新及刷新后采样被审计，显存/耗时/概率差异有实测数据，失败状态可回传。
5. **有限预算C2运行。** 根据profile冻结训练步数、组大小、评测节点、时限和停止规则；同轮W与GRPO候选使用相同任务/种子进行完整任务及完整恢复评测。完成条件：训练曲线和actual updates、有效组比例、KL/clip/TIS统计、工具行为与最终成功均可回查；本地使用冻结source重放并重算报告。
6. **C3配对扩展。** 从同一个W重新训练细粒度分支，尽量固定采样预算、训练配置、reference、seed调度与评测协议，记录实际生成token/动作和计算量差异。在线策略分化后会产生不同轨迹，这是各自策略采样结果。完成条件：信用差异实际进入损失，训练前后与outcome分支按同预算比较。
7. **C4冻结交付。** 完成新任务族/种子确认和关键结果必要复验，整理表格、曲线、成功与失败案例、复现命令、技术报告、简历要点、面试讲解。完成条件：每条主结论对应模型、配置、数据、轨迹和审计证据，保留真实效果与失败分析。

原计划C3同时包括一种SQLite依赖扩展，可在C2闭环稳定后落地；实现日程按剩余时间排序。新对话优先推进GRPO主体，不继续搜索DPO候选。

## 9. 任务扩展与评测设计

轻量长程只做一种分阶段业务对账：后阶段必须使用前阶段已经验证的结果/证据。记录专家最短路径、实际依赖深度、有效动作、上下文长度。多个独立月份查询可能被单条SQL合并，设计需检查是否真的形成依赖；单纯提高max_turns记录为预算增加。

保持原短任务回归，扩展训练/开发/最终确认任务在族、模板或派生数据与种子上明确隔离。现有make_tasks固定train12、development4、test4；增加配置task_count只会选择现有任务，扩任务要实现生成逻辑。test-000～003已有多轮评测和诊断，后续标为历史回归集；最终新任务族先冻结、再评测。

主指标是最终任务成功率；核心辅助是错误前缀之后的完整恢复成功率。首步有效、环境recovered、工具选择/证据ID/SQL验证、无进展重复、调用数、tokens和耗时分列。固定同轮任务与seed配对，报告共同成功/共同失败/改善/退化与明确分母；重复rollout按任务聚合，保留任务级结果。

原环境SQL认证包括金额扰动世界；独立测试增加金额+退款月份重排的calendar audit，回查最终使用证据。原有calendar audit属于SQL语义重执行；新数据库上重新生成轨迹属于另一种评测，应分开记录。扩展任务时同步升级真实业务规则、专家、证据、奖励和audit版本。

当前主线评测采用每模型独立进程、max_loras1、固定LoRA ID1/name、规范化target_modules顺序、固定7上下文各2次重复greedy检查。E4曾观察相同adapter/相同FP32评分在多LoRA BF16生成中出现轨迹分叉；独立进程协议降低评测配置差异，质量记录随结果返回。GRPO策略刷新需要新增自己的验证，不能仅沿用离线评测假设。

已有latest评测参数供回归复用：推理max_model_len8192、max_new_tokens256；完整采样temperature0.8，每任务4次；恢复每上下文3次。精确种子和seed_for规则读configs/dpo-sft-start-short-v1.json及scripts/dpo_early86_common.py。GRPO训练采样参数单独冻结，与评测参数分开。

## 10. 云端执行、审计和恢复约定

### 10.1 云端作业

本地准备源码、测试、配置、身份预检、发布清单和唯一run_id。ZIP内部含v2/；用户通常上传至云端v2，再解压到项目父目录。以下是未来新作业的命令格式，RUN和SCRIPT在包实际完成后替换：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/RUN-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/SCRIPT.py
~~~

绝对路径Python直接选择apt-vllm环境，终端提示符显示base也可以正确运行；如用裸python或notebook内直接import，先核对sys.executable/内核路径。新作业不重复套用旧GPU UUID，作业内身份校验保持严格；硬件变更需要新运行配置和fresh同机对照，保留历史日志。

runner保留实时进度、有限时限、唯一输出目录、同名保护；失败和超时仍打包status、阶段日志、partial evidence和已完成产物。云端state.pt保留Adam、Python/Torch/CUDA RNG、采样组游标、模型版本和更新计数；回传是否含大状态文件由包契约明确。执行命令与回传文件名直接给用户，不要求价格参数。

Jupyter卡顿是曾出现的运维问题。先查实际进程、CPU/内存/磁盘/空间、日志增长和GPU任务状态，再决定操作；避免对仍在训练的作业盲目重复启动或覆盖结果。当前没有已连接的远端shell，用户提供的页面/文件名不等于远程执行权限。

### 10.2 本地复核

按run_id保存原ZIP并计算SHA，检查CRC、路径安全后解压至artifacts/reviews/<run_id>/extracted/<run_id>。先核对source发布清单及provenance，再导入返回包冻结源码；当前源码改动后，历史审阅仍使用各自快照。

核对基座/tokenizer/adapter/config/数据身份，prompt/response原生token、采样参数与种子、EOS和预算；核对模型继承、reference缓存、真实更新、曝光/组计数、Adam/RNG证据。逐步重放环境动作、observation、奖励、证据和artifact；重算summary、CSV、配对与选择规则、calendar audit。报告区分“云端执行过的状态核验”和“本地实际拿到的文件”。

Windows默认SQLite3.42曾造成同一错误SQL的报错文字与云端不一致。已使用本地SQLite3.53.1严格重放成功；兼容DLL：C:/Users/app/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/DLLs/sqlite3.dll。必要时在import sqlite3前用ctypes.WinDLL加载，先核对版本与真实差异，保留原轨迹。最近审阅脚本位于artifacts/reviews/dpo-sft-start-short-v1-20261105/review_sft_start_short.py。

本地exec_command曾因Windows进程创建失败不可用；node_repl可用node:fs与node:child_process调用项目Python。工具问题与项目bug分别诊断。CPU torch测试可固定OMP/MKL线程数减轻开销。读取目录规则和当前权限后工作，保护已有模型、数据、运行快照与用户改动。

### 10.3 已解决问题与保留扩展

- E2审计：跨硬件旧reference末位数值差触发门槛，补审计采用独立数值策略；208个原文件保持一致，新增训练/生成均0。报告与修复证据已封存。
- E4初次prepare：GPU UUID从086e2842…变到92e14e85…，prepare停止，更新0。随后paired作业在同一当前GPU fresh训练两支并完成复核。
- E4相同adapter多LoRA推理分叉：证据已保存；四分支dpo-inference-confirmation-v1-20261105仅已准备，未执行。后续主线采用第9节协议；该根因确认保留为可选扩展。
- 更早SFT诊断Generation.mode覆盖分支名：已修复实现，原始失败记录与恢复副本独立保留。

## 11. 核心代码与按需证据入口

用户已读过主要tasks/environment/runtime、SFT、DPO、恢复数据和加权代码。下一对话重点讲解新采样概率、GRPO损失、优势、奖励和关键评测；重复runner/日志/打包只说明差异。

| 需要解决的问题 | 优先读取项目相对路径 |
|---|---|
| 任务、真值、业务口径 | src/apt_v2/tasks.py；src/apt_v2/environment.py |
| 消息模板与Generation字段 | src/apt_v2/runtime.py |
| RL样本准入、现有信用与更新 | src/apt_v2/rewards.py；src/apt_v2/trainer.py |
| 原生推理采样语义 | scripts/test_v2_hard_recovery.py中的NativePolicy；scripts/evaluate_v2_dpo_early86.py |
| 完整恢复与严格重放 | src/apt_v2/matched_dpo.py；src/apt_v2/independent_recovery_test.py |
| 固定评测与质量检查 | scripts/dpo_early86_common.py；scripts/audit_v2_dpo_early86.py |
| 最新候选选择和完整发布链 | src/apt_v2/dpo_sft_start.py；scripts/dpo_sft_start_common.py；scripts/run_v2_dpo_sft_start.py；scripts/audit_v2_dpo_sft_start.py |
| response评分、CE和训练参考 | src/apt_v2/sft.py；src/apt_v2/dpo.py；src/apt_v2/dpo_chosen_ce.py |
| 场景构建、权重、续跑数据 | src/apt_v2/dpo_recovery.py；src/apt_v2/hard_recovery_sft.py；src/apt_v2/dpo_continuation_ce.py |
| 流式日志与失败打包 | scripts/streamed_process.py及最近run_v2_dpo_sft_start.py |
| 已有测试起点 | tests/test_agentic_v2.py；tests/test_hard_recovery_independent_v2.py；tests/test_dpo_sft_start_v2.py |

资料按触发条件读取：

- 制定GRPO范围：PROJECT_REQUIREMENTS.md；reports/project-roadmap-v2-20261103.md（原计划，当前状态由本交接覆盖）。
- 复述DPO整体结论：附录A及reports/dpo-control-experiments-overview-20261105.md。
- 解释最终选择：reports/dpo-sft-start-short-v1-review-20261105.md及对应review目录。
- 定位W的瓶颈：reports/hard-recovery-independent-test-v1-review-20261103.md；reports/recovery-context-diagnostics-v1-review-20261102.md。
- 追溯169对偏好：reports/dpo-recovery-collect-v1-review-20261102.md和对应pairs.jsonl。
- 追溯历史平台：reports/l20-environment-review-20261101.md；最新运行的execution/config环境记录优先。
- 追溯V1：handoff/PROJECT_HISTORY_V1_2026-10-30.md。V1是SmolLM2-135M的单轮算术/JSON/括号SFT/DPO/GRPO学习阶段，已封存；根目录旧sft-v2/dpo_v2属于V1版本名。归档C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029.zip，历史记录SHA6348f933a916f976eb16670caff132a54cbf09710e99a3a7d86e2a73dfbec500。

## 12. 本次交接更新与接续提示

原桌面文件C:/Users/app/Desktop/PROJECT_HANDOFF_V2_2026-11-03.md本次已不在该位置；已读取项目handoff目录保留的同名原交接，以及11月5日旧交接、正式roadmap、当前requirements/README、最终报告和实际源码。本文件整合当前结论，旧交接作为历史保留。

本次新增完整交接和机器核验回执，更新当前文档入口以指向本文件；训练更新0、GPU作业0。核验回执记录真实模型/数据哈希、文档路径与正文引用文件检查；本次不重复运行此前已通过的模型评测。

可在下一对话附上本文件并发送：

> 请阅读这份GRPO接续交接，按已确认主线进入C2。先核对加权SFT起点、现有RL内核和云端运行契约，实现结果奖励GRPO最小在线闭环，完成本地验证后提供有限预算云端包和命令。随后做同起点细粒度对照和一种SQLite真实依赖扩展。DPO阶段已结束，E1–E4作为扩展材料保留。报告在对话简要给结果并附链接，不自动打开。

---

## 附录A：DPO控制实验与机制排查完整记录

下文从已完成的控制实验总览纳入，作为项目扩展成果。各轮均使用各自同轮对照，早期首动作有效率与后期完整恢复成功率分开。其结论用于后续报告与面试材料；当前执行入口为第8节。

### 模型基准与共同设置

所有实验的基座是固定 revision 的 Qwen2.5-1.5B-Instruct，采用 LoRA；可训练参数 1,089,536 个。任务为 SQLite 多轮取证、计算、检查与提交。

| 简称 | 模型及形成过程 | 累计更新履历 | 主要用途 |
|---|---|---|---|
| S0 | 最早 SQL 加权 SFT，126 次 SFT 更新 | SFT126 / DPO0 | 首轮 DPO 起点 |
| D11 | S0 上用 44 对 SQL 偏好更新 11 次 | SFT126 / DPO11 | 早期 169 对恢复 DPO / chosen-only SFT 的共同起点 |
| P | D11 上用 169 对数据的 chosen 做 3 遍、129 次 SFT | SFT255 / DPO11 | matched DPO、E1–E4 和 DPO86 的父模型/reference；SHA 前缀078182 |
| W | P 上对同 169 对 chosen 做困难加权 SFT，6 遍、258 次更新 | SFT513 / DPO11 | 最佳加权 SFT 对照；最终短程 DPO 的起点/reference；SHA 前缀f8fb95 |
| 最终短程候选 | W 上进行 uniform DPO，新增12次更新 | SFT513 / DPO23 | DPO 阶段收尾模型；SHA 前缀77c737 |

“P”“W”都保留此前的 11 次 DPO 履历。后文“DPO86”“短程12步”表示各自当前阶段新增的更新数。

除首轮使用 44 对数据外，后续 DPO 主线固定同一批 169 对 train 偏好数据，SHA256 c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。常见训练设置为 batch4、beta0.1、max_grad_norm1、fresh AdamW；6 遍对应258次更新、1014对曝光。E1与DPO86直接重评新增训练为0。

DPO 对 chosen/rejected 的响应 token logp 分别求序列和，与冻结起点 reference 的相对差比较，计算负log-sigmoid。场景权重乘在每对 DPO loss 外，再对 batch 求均值。原困难权重为84行hard4、其他85行weight1，经均值归一后困难行占总权重约79.81%。

下文完整任务采样通常是4个development任务×4次，共16回合；完整恢复是4任务×7上下文×3次，共84回合。早期恢复探针统计下一步有效动作，后期完整恢复统计最终成功；两种指标分别记录。每轮结论采用同轮对照数字。

### 早期 DPO 与 SFT 目标对照

| 实验 | 基准与改变条件 | 结果梗概 | 结论 |
|---|---|---|---|
| 首轮 SQL DPO | S0作策略和reference；44对SQL偏好，uniform，1遍11步，LR5e-6 | 训练任务新采样完整成功26/48→28/48；后续独立test12/32→13/32；greedy均4/4 | 小规模DPO带来小幅SQL与任务收益 |
| 恢复 DPO 1遍 | D11作策略和reference；扩展到169对，uniform，43步 | 同轮完整任务10/16→12/16；恢复首步有效动作29/84→31/84 | 收益主要来自初次SQL更正确，重点恢复场景仍薄弱 |
| 恢复 DPO 3遍 | 同D11、数据、reference和优化设置，增加到129步 | 完整任务11/16→11/16；首步有效动作29/84→34/84；第43步权重与单遍结果完全相同 | 增加训练主要改善格式和重复动作，完整任务收益停滞 |
| chosen-only SFT目标对照 | 从同D11出发，同169行、3遍129步；用chosen token CE替换DPO目标 | 同轮完整任务D11/DPO/SFT均11/16；恢复首步有效动作29/84、35/84、37/84 | 直接监督正确动作略好，形成后续父模型P |
| 困难加权SFT基准形成 | 从P出发，同169行chosen，hard4/1，新增6遍258步 | 同轮完整任务11/16→15/16；首步有效动作37/84→74/84；后续DPO同轮对照中完整任务通常16/16 | 得到更强的加权SFT基准W |

### matched DPO 后的控制实验

#### 初始 matched DPO：匹配加权 SFT 的训练安排

基准：P。以W为同轮性能对照。DPO从P出发并冻结P作reference，沿用与W相同的169对、hard4/1、6遍258步、1014对曝光、LR5e-6、batch4；训练目标为加权DPO。

结果：W的采样完整任务16/16，DPO0/16；采样完整恢复59/84对13/84；greedy恢复21/28对6/28。

结论：匹配曝光和更新安排后的DPO出现严重退化。偏好loss从0.693降至约0.164，同时完整任务失败增多；平均margin增量约96.8%来自rejected概率下降。stale场景chosen CE由0.004793升到1.065367。后续排查重点是更新积累、困难权重、正确动作保真和续跑覆盖。

#### E1：更新阶段定位

基准：同一次matched DPO训练的43/86/129/172/215/258步检查点，加P与W。改变比较的更新阶段，新增训练0。

| 新增DPO步数 | 43 | 86 | 129 | 172 | 215 | 258 |
|---|---:|---:|---:|---:|---:|---:|
| 采样完整任务成功/16 | 10 | 13 | 11 | 10 | 1 | 0 |
| 贪心完整恢复成功/28 | 15 | 21 | 18 | 17 | 14 | 6 |

结论：86步保留较好行为，后期更新累积伴随明显退化；正确计算、检查、提交的概率随后期训练受损。训练loss与终局成功需要分别监测。

#### E2：困难权重4改为统一权重1

基准：P。新训练分支保持169对、P reference、6遍258步、LR5e-6、batch4等设置，仅把训练hard_weight从4改为1；原加权模型作为冻结对照。

结果：同轮原加权→uniform，采样完整任务1/16→2/16，完整恢复15/84→33/84。E3又在L40S重新训练加权控制，同轮均匀模型恢复33/84、新加权控制10/84。

结论：这套长程DPO配方中，困难加权放大了恢复退化；统一权重有缓解作用，长程uniform分支仍明显落后W。

执行记录：E2与历史加权分支的硬件/跨作业reference数值差异单独保存；随后E3补同硬件加权重训。原审计的数值门槛报错通过独立补审计修复，208个原文件不变，新增训练和生成0。

#### E3：加入chosen CE维持正确动作保真

基准：P。同机fresh训练两支，保持hard4、169对、258步、LR5e-6和精确同一reference缓存；目标分别是mean(w×DPO)与mean(w×DPO + 1×chosen CE)。CE在动作内取响应token均值，动作间均匀平均。

结果：纯加权→DPO+chosen CE，采样完整任务0/16→10/16，完整恢复10/84→37/84。stale chosen CE由1.065365降到0.055512；同轮W为16/16、59/84。

结论：显式维持正确动作概率能明显缓解退化。相对偏好margin增大与正确动作绝对概率保持是两个需要同时观察的方向。

#### E4：把单动作chosen CE替换为完整续跑CE

基准：P。两支均保留hard4 DPO、相同reference、258步与CE系数1；把CE数据从原169个chosen替换为train的84条专家完整恢复续跑，共288个动作，覆盖查询、计算、检查和submit。两支均1014次CE动作曝光；续跑CE额外前向及token计算量单列。

结果：同机chosen CE→完整续跑CE，采样完整恢复36/84→60/84，greedy恢复14/28→25/28；有效首动作后的失败33次→5次。采样完整任务8/16→7/16；同轮W完整任务16/16、恢复57/84。

结论：完整续跑监督明显改善修复后的校验、提交与终局恢复；完整任务前段SQL和格式仍拖累表现。完整任务与恢复成功作为独立指标保留。

#### 早期DPO86直接对照

基准：W与已有DPO86。新增训练0；DPO86仍是从P训练的hard4模型，累计SFT255/DPO97。固定单模型独立进程、单LoRA编号1、规范化配置顺序、固定任务/种子/温度并补必要质量检查。

结果：W→DPO86，采样完整任务16/16→12/16，完整恢复60/84→51/84；greedy完整任务均4/4，greedy恢复均21/28。

结论：早期检查点优于末期模型，完整任务仍落后25个百分点，阶段验收继续推进到强SFT起点的保守短程方案。

#### 最终短程DPO：强SFT起点、小学习率与预定早停

基准：W，同时用W重新计算新reference。相同169对池，统一权重1，LR从5e-6降到5e-7，beta0.1、batch4；候选累计步数固定为4/12/24，按首个达标节点停止。

| 同轮指标 | W | 第4步 | 第12步 |
|---|---:|---:|---:|
| 采样完整任务 | 16/16 | 16/16 | 16/16 |
| 采样完整恢复 | 62/84 | 60/84 | 59/84 |
| greedy完整任务 | 4/4 | 4/4 | 4/4 |
| greedy完整恢复 | 20/28 | 19/28 | 21/28 |

预定标准：完整任务至少15/16且最多比W少1次，采样恢复最多少4/84，greedy完整任务和恢复均不下降。第4步在greedy恢复上少1次；第12步全部通过，停止后续训练，24步未执行。

结论：这组保守配方保留W的完整任务表现，并达到恢复容差。实际12步、48对曝光；真实LoRA已更新，第12步整体相对起点L2改变量约0.0273%。20条完整任务的响应token序列与W对应轨迹相同，恢复轨迹出现少量改变。该轮用于配方与阶段目标验收。

### 对整条实验线的归纳

1. 相对偏好loss下降与最终任务成功存在脱节。压低rejected能增大margin，同时局部正确动作概率可能受损。
2. 在本项目的旧起点、hard4、LR5e-6、6遍配方下，后期累计更新和困难权重共同伴随明显退化；checkpoint与uniform对照分别定位了这两条线索。
3. chosen CE改善正确动作保真，完整续跑CE改善修复后的流程推进与提交；完整任务前段SQL能力仍需单独维持。
4. 本阶段有效的收尾配方是强SFT起点与匹配reference、统一权重、较小学习率、短程更新和预定停止标准的组合。最终完整任务持平、采样恢复−3.57个百分点、greedy恢复+3.57个百分点。
5. E1–E4按机制排查扩展保存；DPO阶段以第12步候选收尾，后续GRPO共同起点保持原先确认的W。

各轮结论采用各自同轮对照。机制排查与最终候选选择使用固定development任务；最早DPO11的历史独立test单独记录。最终新任务/种子确认按项目总计划保留。

### 推理与执行补充

E4发现相同adapter及相同FP32评分在多LoRA BF16生成中有轨迹分叉，已保留证据。随后主线直接对照采用单模型独立进程、固定单LoRA位置与必要的重复贪心检查。另准备过四分支推理确认包，未执行，未计入上述结果。最初E4因GPU UUID变化在prepare停止，更新0；新作业在同一当前GPU上fresh训练对照，并保留作业内严格身份检查。

### 原始报告索引

- dpo-train-v1-review-20261102.md；dpo-independent-test-v1-review-20261102.md
- dpo-recovery-train-v1-review-20261102.md；dpo-recovery-train-v2-review-20261102.md
- chosen-only-sft-v1-review-20261102.md；hard-recovery-sft-weighted-v1-review-20261103.md
- hard-recovery-dpo-matched-v1-review-20261103.md
- dpo-regression-checkpoints-v1-review-20261104.md
- dpo-uniform-control-v1-review-20261104.md
- dpo-chosen-ce-control-v1-review-20261105.md
- dpo-continuation-ce-paired-v1-review-20261105.md
- dpo-early86-vs-weighted-sft-v1-review-20261105.md
- dpo-sft-start-short-v1-review-20261105.md

原始报告位于项目v2/reports；返回包、冻结源码及本地复核记录位于v2/artifacts/reviews/<run>。本次已将总览纳入本交接附录。
