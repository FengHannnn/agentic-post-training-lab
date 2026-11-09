# V2 GRPO Continuation Handoff

更新：2026-11-09，Asia/Shanghai。文件2：下一对话的主要执行交接，覆盖 C2 起至 alias curriculum attempt-003 结束。SFT/DPO结果与模型祖先见同目录文件1 `PROJECT_HANDOFF_V2_SFT_DPO_RESULTS_2026-11-09.md`。

## 1. 接续先读：当前结论与本轮任务边界

**W仍是正式基线；GRPO在线闭环、outcome/fine同预算比较、独立确认、后续信用/课程实验都已运行，但“最终超过W”的效果目标仍待达成。用户现在要求新的两文件交接，尚未确定 alias 之后的新实验，也未授权下一轮GPU预算。**

最新完成作业为 `grpo-alias-curriculum-v1-20261109`，第三次返回包：

```json
{"error":null,"stop_reason":"completed_with_quality_rejections","actual_updates":{"control":25,"curriculum":37},"committed_groups":{"control":48,"curriculum":48}}
```

作业约5小时20分，训练/证据完整性通过，质量验收未通过，`selected_candidate=null`。简化缺别名课程增加有效更新，但最终原复合缺别名成功数没有提升。标准恢复有局部收益，完整任务主指标无增益；两支还暴露空退款SQL边界，control另有重复greedy不一致。

接续优先顺序：读最新结果 → 核对最新代码位置并整理主目录 → 和用户共同确定下一轮干预、对照与预算 → 本地验证 → 提供云端包。不要直接照旧命令重新启动已结束的 alias 作业。

事实来源优先级：最新用户指示 → 本文件及最新已审计结果 → 各轮冻结配置/源码/审计 → 历史计划。`PROJECT_REQUIREMENTS.md`和2026-11-05旧交接的进度描述尚停留在GRPO开始前，接续时按本文件更新理解；历史冻结产物继续保留。

## 2. 路径、实际存储状态与代码来源

以下短名用于避免反复写长路径。表格中的代码相对路径均相对所标记的V2根目录。

| 短名 | 绝对路径与用途 |
|---|---|
| PROJECT | `C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2`，用户本地主项目 |
| PYTHON | `C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/.venv/Scripts/python.exe` |
| SESSION | `C:/Users/app/Documents/Codex/2026-11-05/new-chat`，本对话工作区 |
| OUTPUTS | `C:/Users/app/Documents/Codex/2026-11-05/new-chat/outputs`，已交付报告、运行包、修复包及回执 |
| LATEST | `C:/Users/app/Documents/Codex/2026-11-05/new-chat/work/alias-results-review/v2`，alias第三回包全量解出并用于独立审计的最新有效运行源码与产物 |
| CLOUD | `/root/rivermind-data/AgenticPostTrainingLab/v2`，用户云端项目 |
| CLOUD_PYTHON | `/root/rivermind-data/envs/apt-vllm/bin/python` |
| CLOUD_BASE | `/root/rivermind-data/AgenticPostTrainingLab/v2/.cache/qwen2.5-1.5b-instruct` |

### 2.1 本次确认的本地未同步事项

PROJECT 已存在最新 alias 结果报告 `reports/grpo-alias-curriculum-results-v1-20261109.md`，该文件在用户指出缺失后已实际复制并核验与OUTPUTS一致。

**PROJECT尚不是最新完整运行树。** 本次逐文件比较：

- PROJECT缺少 `src/apt_v2/grpo_alias_curriculum.py`、`scripts/run_v2_grpo_alias_curriculum.py`、`scripts/audit_v2_grpo_alias.py`、`configs/grpo-alias-curriculum-v1.json`。
- PROJECT的 `src/apt_v2/grpo_curriculum.py`、`scripts/evaluate_v2_grpo.py`、`scripts/grpo_vllm.py` 与LATEST不同。
- `src/apt_v2/grpo.py`、`grpo_credit.py`、`grpo_optimizer.py`、`period_closing.py`和 `scripts/grpo_sampler_probe.py` 本次抽查与LATEST一致。此处是关键文件抽查，不是全树等价声明。
- fine和hard预检的最新结果报告尚未实际放入PROJECT/reports；OUTPUTS中的同名报告与结果包可用，见第13节。

下一代理先以LATEST和发布/修订记录核对差异，再把需要继续开发的最新代码、测试、配置与报告整理到用户主目录。保护本地已有改动和运行产物；不要用盲目覆盖代替差异检查。本次交接只写文档，没有执行代码同步。

LATEST适合读最新有效源码与已有产物；原始review ZIP仍是历史证据。新实验应从核验过的代码生成新的冻结发布包，使用新run_id。历史复核一律用该轮冻结source及source_revisions，不直接导入任意当前源码。

### 2.2 现有运行包与补丁的关系

OUTPUTS保留 alias 原始run包，以及 `grpo-alias-midpoint-fix-v1-20261109.zip` 和 `grpo-alias-quality-outcome-fix-v1-20261109.zip`。原始run包代表初版发布，不等于已打完两次修复的最新代码。两个安装器为已有云端作业迁移而设计，包含原状态前置校验。新实验直接整理修复后的源码并新发布；历史续跑按修复文档对应前置状态操作。

## 3. 基本配置、模型身份与运行平台

| 项目 | 当前常用设置 |
|---|---|
| 基座 | `Qwen/Qwen2.5-1.5B-Instruct` |
| revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 主权重身份 | SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`，3,087,467,144 bytes |
| W adapter | SHA256 `f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f`；SFT513 / DPO11 / RL0 |
| W输入目录 | PROJECT和CLOUD中的 `artifacts/inputs/grpo-w-20261105/adapter` |
| LoRA | q_proj/v_proj，r8、alpha16、dropout0，112张量、1,089,536可训练参数 |
| 生成 / 评分更新 | vLLM BF16；HF/PEFT FP32，只有LoRA可训练 |
| 常用GPU | L40S；当前最新作业内GPU、软件身份固定 |
| 软件 | Python3.11.17、Torch2.8.0+cu128、CUDA12.8、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1；近期SQLite3.54.1 |
| 新组默认 | K4、每组最多1次optimizer.step；W初始化、W reference、独立新AdamW |
| 训练解码 | T1、top_p1、top_k=-1、min_p0、无额外惩罚；processed_logprobs；回复256、上下文8192 |
| 更新超参数 | LR1e-6（outcome curriculum曾比较1e-5）、epsilon1e-8、beta0.02、clip0.2、TIS cap2、grad clip1 |
| 保护条件 | sampled k3均值>0.05或clip fraction>0.5则跳过更新；非有限值/无效身份直接报错 |
| 训练mode | model.eval()保持确定性路径；autograd单独开启，eval不会自动关闭梯度 |

所有后续实验逐轮冻结实际参数，不能把上表视为未经讨论可任意改动历史配置的许可。不同实验独立从W开始；没有按时间顺序把所有GRPO检查点继续叠加。

### 3.1 硬件克隆与比较约定

用户曾在C2完成后克隆到L20，明确确认发生在该轮结束之后；C2全部资源记录仍属L40S。C3又在另一张L40S上完成同机双分支，之后也多次换过L40S实例。

换机器保持权重/数据身份，但浮点路径、kernel、采样输出与速度可能变化。每个新作业记录GPU、UUID、驱动、软件，在同机fresh评测W和候选。作业内及同作业resume沿用冻结身份与累计预算；若迁移跨卡，先按新运行设计，不改旧UUID来冒充同一运行。同seed作为配对索引，仍核验实际原生tokens和q。

## 4. 项目安排：已完成与未完成

| 工作 | 现在的状态 | 接续含义 |
|---|---|---|
| V2基础任务/工具/证据/重放 | 已完成；后续有独立任务扩展 | 保留契约，按版本扩展 |
| SFT/DPO与C1收尾 | 已完成，文件1 | 不继续自动搜索DPO |
| C2 outcome GRPO闭环 | 已完成，预检、真实更新、刷新和有限训练均审计 | C2开发候选RL16保留 |
| C3 outcome/fine同预算 | 已完成，outcome RL15/fine RL17 | 初版fine仅小幅开发收益 |
| 一种SQLite真实依赖扩展 | 环境、专家因果证明和开发评测已完成 | 模型0/4，训练覆盖及依赖任务能力仍待改善 |
| C4独立确认 | 已完成12新任务族评测，审计通过 | 两候选效果门槛未通过；最终超W仍待完成 |
| outcome课程/LR对照 | 已完成，48组/支，29/28更新 | 无合格候选 |
| legacy/local信用对照 | 已完成，46组/支，28/37更新 | local部分改善，仍无合格候选 |
| 困难恢复冻结预检 | 已完成，96条，仅采样0更新 | 找到缺别名简化起点 |
| alias课程正式训练 | 已完成48组/支，25/37更新 | 原复合迁移无提升；质量拒绝，W保留 |
| 最新代码/部分报告归入PROJECT | 部分未完成 | 第2.1节列出实际差异 |
| 训练verifier空退款边界 | 已识别，尚未实现新训练版本 | 不能把质量收尾补丁当作verifier已修复 |
| alias之后新实验方案/预算 | 尚未确定 | 与用户共同设计后冻结 |
| 新的最终独立确认 | 待出现合格开发候选后安排 | 新任务、种子、门槛在看结果前冻结 |
| 最终README/技术报告/曲线案例/简历面试材料 | 历轮已有大量证据，统一最终交付尚待整理 | 不把完成各轮报告等同于总交付完成 |

C2/C3/C4暂定必做；其它扩展选做。GDPval、多文件办公、网页UI、更多算法/种子选做，multi-OPD已排除。原10天、日均6小时是历史相对工作量规划，不是新的固定日期期限。重大范围和作业预算变化继续与用户确认。

## 5. 主体框架：任务、真实执行与证据

```text
Task + SQLite业务数据/规则
  → 初始提示 / 恢复前缀真实重放
  → vLLM生成JSON动作 + 原生response token IDs + q
  → Environment实际执行
  → observation、认证指标/证据、奖励进入轨迹与后续上下文
  → 成功submit或环境预算终止
  → K条同起点轨迹计算信用
  → HF重算old/ref并冻结cache → GRPO LoRA更新
  → 不可变checkpoint + Adam/RNG/事务提交
  → fresh vLLM进程加载真实新adapter → 再采样
  → W/候选固定配对评测、质量检查、严格重放与独立审计
```

### 5.1 原业务与认证规则

orders含订单月份、region、segment、status、gross_cents；refunds含原order_id、退款发生月份和refund_cents。当前/上月净收入：目标region/segment的paid订单按订单月份汇总gross，减去该月发生的、关联原paid订单的退款。原订单可以来自其它月份。`change_cents=current_net_cents-prior_net_cents`。

模型只见公开任务、规则/schema与观察；seed/split/family、truth等内部字段不进入用户任务提示。query_sql只读执行，有表/函数白名单、长度/行数/VM预算。三个认证指标为 current_net_cents、prior_net_cents、change_cents。

query_sql的qN证据登记与查询尝试绑定，临时故障也影响编号；执行成功可能有rows和qN，但 `verified_metrics=[]`，该ID没有可用指标。calculate_change必须引用分别认证current/prior的证据。check_result/submit要求指标名→对应证据ID、正确值及已读规则/schema。不能依赖固定q1/q2/c3编号。

训练时原SQL verifier在原库和两个独立金额扰动世界验证整数值、精确alias和来源；近期最终评测另有退款月份重排calendar audit。空退款NULL反例正是两者覆盖差异暴露的问题。新verifier版本与训练改变仍待设计，不回改历史成功标志。

### 5.2 复杂度究竟如何增加

- C4：仍为原12步业务模板，增加全新任务族/年份月份/区域客群和评测覆盖；主要提高实例变化与确认强度，没有新增结账工具。
- outcome curriculum及fine study：训练扩为12旧+12新任务，增加3类复合恢复：missing_alias_pair、schema_query_repeat、requery_stale_change；新开发任务隔离。增加task_count本身不会自动创造新任务。
- alias课程：从原两个月都缺alias的复合起点，拆出上月已认证、只需修当前月alias的中间起点；匹配前缀长度和剩余预算。
- period_closing：另一个真正增加依赖深度的环境，见第5.3节；不要和C4实例扩充混为同一改变。

标准7类恢复：format_repair、unknown_tool_repair、sql_unverified_requery、repeat_evidence_error、stale_evidence_repair、repeat_sql_error、failed_check_repair。新增困难场景与标准指标分表。

所有原短任务/恢复总预算12包含前缀；前缀是已执行上下文，继续执行动作才进入本轮训练损失。完整失败且已正常终止可训练；基础设施中断轨迹保存诊断并拒绝训练准入。

### 5.3 SQLite真实依赖扩展的精确状态

`src/apt_v2/period_closing.py`：原对账认证 → close_period写入period_closings → 查询该结账净额加下一期ledger_movements得到next_balance_cents → 提交四指标、证据与closing_id。工具控制写入，query_sql仍只读。删除结账行使后续查询无结果；篡改净额改变下一期余额并触发完整性检查，专家因果证明已审计。

4个closing-development任务，总预算16，专家9/10动作可完成。C3的W/outcome/fine均0/4，均未调用close_period，在旧check_result循环。C3只训练旧月度任务；后续curriculum/fine/alias的closing_evaluation关闭。现状是扩展环境实现和评测已完成，模型尚未学会该流程；如需改善，需另定训练覆盖与预算。

## 6. 算法契约：outcome、legacy fine和local

### 6.1 Outcome

每步 `outcome_reward=1{本步成功}-0.01`。episode return含前缀与继续执行全部动作成本：

`R_i = 1{最终成功} - 0.01 * total_turns_i`。

同组K4必须同train task、start_id、policy_version、sampling，rollout seed独立。总体标准差用分母K：

`A_outcome_i=(R_i-mean(R))/(population_std(R)+1e-8)`。

每个继续执行动作共享其轨迹优势。全同回报组优势0、提交skip，不拒绝采样、不补组；全成功也可能因长度差异有非零优势。组数、训练曝光、实际更新数、跳过数分别记录。

### 6.2 初版细粒度：FINE / legacy

方法名 `potential_rtg_depth_v1`，C3 fine RL17与后续legacy都用该规则。势函数 `Phi(s)=0.2*已认证指标数量`；重复认证不增加数量；终止状态Phi强制0。恢复起点可能已有Phi>0。

每动作奖励 `r_fine_t=outcome_reward_t+Phi(s_{t+1})-Phi(s_t)`，gamma1；从当前继续执行动作累加到结束得到 `G_fine_t`：

`G_outcome_t = sum_{u=t}^{T-1} outcome_reward_u`

`G_fine_t = G_outcome_t - Phi(s_t)`，因为终止Phi为0。

这里G_outcome是“当前动作到结束的outcome剩余回报”，不是所有t都相同的完整episode return。恢复前缀成本在episode return中固定保留；逐动作RTG对继续执行段计算，起点Phi由前缀决定。

在相对继续执行位置t，把尚活跃的轨迹放在一起，按该位置RTG均值/总体标准差标准化。至少2条活跃且数值不全同时该depth有效，否则动作优势0。任一depth有效即可更新。

已观测局限：相同相对t未必处于相同业务状态；低进展状态的 `-Phi` 较少，可能使无新进展动作获得正优势；终局清零还会影响RTG。信用诊断见OUTPUTS的 `grpo-fine-rl17-credit-diagnostic-20261108.json` 和fine结果报告。这是已记录的具体诊断线索，后续通过local对照检验过。

### 6.3 Local

方法名 `outcome_local_verified_v2`：

`A_local[i,t]=A_outcome[i]+0.2*len(new_verified_metrics[i,t])`。

`seen`从前缀已认证指标初始化，后续去重；重复取证、无新认证动作局部部分0。终止Phi清零不扣local加成。实现仍验证潜势变化与process_reward一致性。只要综合动作优势中有非零项就eligible；即使outcome组内回报全同也能更新。

这是明确登记的 **outcome + 局部认证信号混合、自定义动作信用**，有正向偏置，不沿用potential等价塑形的目标解释。它没有引入额外SFT/DPO更新。局部部分为0并不意味着综合优势必为0，综合仍含A_outcome。

local已提高有效组和部分标准恢复，但所有轨迹均无新认证且终局回报全同时，仍无学习信号；“首个正确动作采不到”是当前重点。

## 7. 概率、目标函数、显存与更新事务

| 符号 | 项目里的来源 |
|---|---|
| q | 实际vLLM行为分布的processed逐token logp，绑定原生tokens与真实采样设置 |
| old | 当前采样checkpoint在HF评分路径重算的逐token logp，detach |
| new | 当前策略HF逐token logp，参与梯度 |
| ref | 冻结W的逐token logp，detach |

代码中的new/old/ref/q都是log probability。每token ratio `r=exp(new-old)`；TIS `w=min(2,exp(old-q))` detach，用于修正BF16生成与FP32训练路径的分布差异；截断保留方差/偏差权衡。与PPO ratio的作用分别记录。

动作loss：

```text
delta = ref - new
k3 = exp(delta) - 1 - delta      # 实现用double + expm1减小消减误差
surrogate = -min(r*A, clip(r,0.8,1.2)*A)
action_loss = mean_response_tokens(w * (surrogate + 0.02*k3))
group_loss = (1/K) * sum_i [ (1/continuation_actions_i) * sum_t action_loss_i,t ]
```

所以动作内token平均、轨迹内动作平均、组内轨迹平均。仅响应token进入目标，不对prompt或脚本前缀监督。actual_updates按真实optimizer.step计，不用组游标替代。

`response_scores`使用 `logits_to_keep=len(response)+1` 做因果后缀对齐；prepare_group_scores冻结old/ref/q并校验cache绑定，HF同路径old与开启梯度的fresh差≤1e-5。reference和非LoRA参数冻结。每动作backward累积，最后保护检查、梯度裁剪和一次step。

显式 `enable_eval_checkpointing` 包装decoder forward：有梯度时用非reentrant activation checkpoint，反向重算以节省显存，保持eval模式；它不是保存磁盘训练checkpoint。无梯度时直接普通forward。

更新后新建vLLM进程加载不可变adapter，核验实际112个worker LoRA张量和BF16有效投影；缓存身份、reference、原生token、权重差与Adam/RNG共同审计。新增RL更新与继承SFT/DPO计数分开。

### 7.1 当前概率探针的版本边界

旧curriculum在collect-control-0018两次停止：prefill/decode跨路径差超0.25；第一补丁的跨请求同seed概率逐位一致假设也失败。第二版 `same_forward_sampler_v2` 观察**同一次采样前向**的logits和原生采样器q，首动作前最多16token，FP32 log_softmax误差≤1e-5。观察器不更改logits、采样输出和RNG。

旧prefill探针保留为诊断，额外约1token/组；跨请求差异不再当作当前q准入证据。历史前18组按当时快照审计，后续组按新版本审计。不要把prefill阈值简单放宽来替代同次前向证据。细节引用 `grpo-outcome-curriculum-same-forward-repair-20261108.md`。

### 7.2 原生EOS与动作截断

C2曾在collect-0017因原生EOS验证失败中断，补丁已修复并沿用：读取tokenizer及冻结generation_config支持的EOS集合，不把Qwen合法的另一终止token当作数据错误；stop要求末尾为合法EOS，length保持实际采样到的长度，不能事后补token或重编码替换。非法JSON仍交给环境记录为失败动作；推理基础设施异常单列interrupted。原修复见PROJECT的 `reports/grpo-outcome-formal-v1-interruption-review-20261106.md` 与OUTPUTS的EOS修复包，历史组/Adam/RNG保持原样后完成续跑。

## 8. 已完成实验索引与结果摘要

所有数字使用该轮同机W；不要跨行用不同开发集/硬件的W计数直接推断训练趋势。完整结果、逐例配对与审计引用第13节报告。

| 实验 | 相比前轮主要改变 | 模型 / 组数 / 实际更新 | 主要结果与状态 |
|---|---|---|---|
| C2正式outcome | 实现原生K组采样、outcome优势、真实更新与刷新 | W、outcome；24组，RL16；另评第8组RL6 | sample完整均16/16；恢复61/84→63/84，greedy恢复19/28→20/28；开发候选RL16 |
| C3 matched | 同机从W重训outcome/fine，只变信用；加入结转开发评测 | 各24组；outcome RL15、fine RL17 | sample完整均16/16；恢复W/outcome/fine=60/60/61（各/84）；greedy恢复均21/28；结转均0/4 |
| C4 v2确认 | 不训练；12全新任务族；原8族方案在启动前扩为12，预算4→7小时 | W、C2 RL16、C3 fine RL17 | 完整28/30/29（各/48）；恢复171/175/171（各/252）；greedy恢复61/58/58（各/84）；均未过门槛 |
| outcome curriculum | 12旧+12新train、新dev、3类困难场景；LR1e-6/1e-5对照 | 各48组，control RL29、lr10x RL28 | 完整sample7/7/8（各/16）；恢复62/64/63（各/84）；困难均0/24；无合格候选 |
| fine credit | 固定LR1e-6，旧fine重新训练为legacy，与local比较；新dev | 各46组，legacy RL28、local RL37 | 完整11/11/12（各/16）；恢复60/56/59（各/84）；困难均0/24；无合格候选 |
| hard preflight | W固定、3错误×2难度×4train×K4，只采样 | 24组96轨迹，0更新 | 缺别名简化11/16成功、14/16进展、4/4有效；复合三项0；其它类别仍困难 |
| alias curriculum | 固定local与LR，50%维护+50%缺别名；直接复合 vs 先简化后复合 | 各48组，control RL25、curriculum RL37 | 恢复61/60/65（各/84）；完整14/15/14（各/16）；复合均1/12；质量拒绝，保留W |

C2、C3预检和真实更新均已成功；本表“无合格候选”指对应效果选择结论，不等于训练或审计没有发生。fine study的legacy RL28不是把fine RL17继续训11步，而是W新初始化、相同信用方法的新实验。

### 8.1 C4冻结验收规则与数据使用

C4 v1原8族、792条episode、4小时；用户在尚未启动、未查看结果时允许扩到12族，v2为1188条episode、42次质量动作，7小时上限；实际4小时28分22秒。

最终实用门槛：sample恢复至少比W多6/252（+2.38pp）、sample完整和两项greedy不下降、主指标严格增加。原v1对应多4/168，扩容后保持比例为6/252。W完整sample28/48未饱和，所以主指标为完整任务。outcome恢复+4未够6，两支greedy恢复均-3，最终均未通过。

后续开发已用这些失败模式形成课程，因此这批C4任务保留冻结结果，现为历史确认/诊断资料；下一次最终确认另冻新任务。C4本身没有训练更新。

### 8.2 初版fine和local的诊断结论

- 初版fine在C3多产生一些逐动作信用，包含终局全失败但有认证进展的组；净收益仅恢复+1/84。C4收益未保持，greedy恢复回退。
- legacy/local同轨迹离线重算验证：local修正一部分无进展动作正优势，并提高有效组率；实际训练legacy28/46、local37/46。局部改善没有带来全指标合格候选。
- fine两支的三类新增困难训练32条/支全部失败、无新增认证，8组/支均0更新；局部信用也没有可奖励的动作。outcome学习率增10倍同样无法让两类全失败场景产生更新。
- hard preflight只诊断，不优化；它找到的缺别名中间起点同时提供正确上月SQL示例和有效证据锚点，二者是该起点的整体定义，尚未拆分各自作用。

## 9. 最新alias实验：下一步必须保留的细节

### 9.1 固定安排

control/curriculum分别从W独立开始，新AdamW、W reference、local、LR1e-6、K4，额外SFT/DPO0。每支48组=24维护+24目标；维护为12完整+12标准恢复。训练12任务为6旧+6此前课程任务，两阶段重复相同任务。

前24组：control12目标全原复合，curriculum12目标全简化；后24组：两支12目标都原复合。每两个组按维护→目标，阶段切换保留Adam/RNG。两起点前缀长度与剩余预算相同；简化仅恢复上月SQL的AS prior_net_cents，本月缺alias、后续无效计算保留。

新dev为alias-development-000～003，seed81000～81003，2032/2033年份及1/4/8/12月配对；排除历史dev/test/C4/fine实例。任务、组序和rollout seed配对。24组固定中点各16条，只作诊断。最终每模型132标准+32定向，另14次质量动作；两支中点合计32条。

### 9.2 效果与信号

| 标准指标（原始环境成功） | W | control | curriculum |
|---|---:|---:|---:|
| 完整sample | 14/16 | 15/16 | 14/16 |
| 恢复sample | 61/84 | 60/84 | 65/84 |
| 完整greedy | 4/4 | 4/4 | 4/4 |
| 恢复greedy | 20/28 | 20/28 | 21/28 |

curriculum恢复相对W为4改善0退步，+4/84（4.76pp）；相对control6改善1退步。完整主指标对W无增益。

三模型定向成功掩码逐条相同：简化greedy4/4、sample11/12；原复合greedy0/4、sample1/12。每模型原复合16条只有1条产生新认证并最终成功，其余15条无新进展。课程支复合sample失败中9条首动作为执行成功却未认证的SQL，2条为无效证据calculate_change；首个有效认证之前仍是主卡点。

训练阶段1目标：control0/12有效、0/48成功；curriculum12/12有效、34/48成功。阶段2目标：两支均1/12有效、1/48成功。维护各24次有效更新，最终25/37更新，额外12次全部来自简化阶段。中点简化control8/8、curriculum7/8，复合均0/8。简化探索/更新增加，但迁移回原复合未出现收益。

### 9.3 质量失败和状态语义

- 两支均有alias-development-003完整sample rollout1的上月净额SQL：退款子查询SUM缺COALESCE，原库和金额扰动通过；calendar世界移走上月退款后返回NULL，正确值36080。
- control重复greedy7组中6组一致，repeat_evidence_error同提示/同seed两次输出分别calculate_change/query_sql；curriculum与W为7/7一致。底层数值机制尚待专项诊断。
- 日历合格的完整sample：W14/16、control14/16、curriculum13/16；恢复仍61/60/65。原始成功数保留，不改写原评测JSON。
- alias原冻结选择同时要求全部质量门槛通过；任一质量失败阻断候选，本轮 `integrity_passed=true`、`passed=false`、`evaluation_quality_passed=false`、`selected_candidate=null`。
- `completed_with_quality_rejections` 表示采集与收尾完成、质量验收拒绝，不是运行器仍需重试的基础设施失败。

### 9.4 两次修复与证据

attempt1：48组训练已结束，末尾重复调用24组midpoint，exclusive创建既有日志触发FileExistsError；修复重复调用、保留训练/中点与计时。

attempt2：control完整评测已存，末尾质量门槛抛错阻止另一支评测。修复后复核复用control原始JSON、写quality-status、完成curriculum既定评测与总审计。结构、配置、token、重放、权重错误仍致命；模型质量/日历失败作为负结果记录，验收门槛未变。此补丁没有修复训练verifier空退款漏洞。

最终累计19212.93秒，低于21600秒，三次attempt连续计时。返包SHA256 `4a97429fae19131191e0dec3f53e21922091644ead514dddaacdbcf5ac3c1c34`，290041670bytes、2319项。255项根源码绑定核验；SQLite3.54.1本地重放392训练/刷新轨迹、2560动作、396标准最终、96定向最终、32中点，独立重算25/37次目标，与云端语义审计除scope字段外逐项一致。真实LoRA和BF16有效张量均改变。

最新最终检查点相对LATEST和CLOUD的根作业目录 `artifacts/grpo-alias-curriculum-v1-20261109/`：

| 分支 | 最终路径 | adapter SHA256 |
|---|---|---|
| control RL25 | `branches/control/checkpoints/group-0046/adapter` | `3e2be6d59457e03409e10ae4b936c112b920e3bcfb1c0f5d52238a93b2b3781b` |
| curriculum RL37 | `branches/curriculum/checkpoints/group-0046/adapter` | `75abcc84a9efc1cd23ce061447d42d2df68004d3db5e620923041037ea132550` |

两支runner_state的next_group=48；最后组未更新，grpo_g0048是已提交48组后的评测标签，不是RL48或group-0048权重。

## 10. 评测、数据隔离与实验边界

1. 用户最终要求GRPO超过W，允许改学习率、奖励、课程等参数；涉及额外监督或混合方法要单独登记。当前local已登记为混合动作信用，不把它称作纯outcome。
2. 最终任务成功率为主；同轮W完整sample达到100%才以恢复为主。当前后续dev多数未饱和，不能因恢复改善就跳过完整任务主指标。
3. 近期开发选择：sample完整、sample恢复、greedy完整、greedy恢复均不低于同轮W，主指标严格增加，且质量全过。C4另有冻结+6/252门槛；该数是C4协议，不自动套用所有新dev。
4. 标准每模型132条=完整greedy4+sample16+恢复greedy28+sample84。评测T0.8，完整每task4次、恢复每context3次；训练T1另行记录。7上下文各2次greedy一致性生成单列14次质量动作，不混入episode分母。
5. 成功率、恢复终局、首动作有效、环境recovered、新认证率、重复/工具错误、tokens/动作数分别报告。配对按task/scenario/mode/rollout，列共同成功/共同失败/改善/退步；任务族聚类统计用于描述不确定性。
6. 训练只准入train身份。专家和verifier可访问内部truth以检查，但模型提示和训练准入不泄露split/seed/gold。新任务generator须明确实现，增加配置task_count仅选已有实例。
7. 旧test和已看C4结果是历史回归/诊断；出现候选后再冻结新的最终任务。开发结果不得回称为新的独立最终确认。
8. 先冻结任务、方法、预算、停止/选择规则，看到结果后不改本轮门槛。若更换训练verifier，使用新版本、新run_id，W与候选在同一评测版本重评，并分列与历史结果差异。
9. 专家须在同预算内可完成，恢复前缀计入总动作；增加max_turns记录为预算改变。真正依赖深度要用删除/篡改前阶段状态的反事实证明。
10. 初轮可单训练seed，报告真实seed与局限；多seed复验和更大确认由后续预算决定。两支对齐采样组数并列实际更新与计算量，不把更新数不同隐藏成等剂量训练。

## 11. 未完成工作与下一轮待决定事项

### 11.1 先做的工程整理（不需要新GPU）

- 核对PROJECT与LATEST差异、保护现有改动，将最新alias代码/测试/配置纳入主项目；补齐fine/hard结果报告和对应归档入口。
- 更新项目README/requirements进度入口指向本交接及当前正式基线W，历史快照保持原样。
- 保留原ZIP、补丁迁移、source_revisions、runner_state、评分缓存和独立审计。不要只保留报告而丢原始概率/轨迹/权重证据。

### 11.2 当前优先科学问题

**A. SQL认证可靠性。** 空退款边界已出现真实反例。下一轮可先为verifier添加金额+日历/无匹配退款世界，确认NULL/COALESCE和退款月份口径，再冻结新版本。明确这会改变训练反馈，单列对照，不把修复后的数字覆盖旧实验。

**B. 原复合缺别名的首个有效动作。** 最新三模型大部分轨迹在取得第一个新认证前停滞。先用冻结采样记录首次补alias正确率、SQL能执行率、指标认证率、calculate_change时证据是否有效，再设计更细中间起点或探索干预。当前还未决定具体改哪种课程、是否改变奖励、是否加监督。

**C. 简化课程成分拆分与迁移。** 简化前缀同时提供正确上月SQL示例和已认证证据，两者贡献尚未分开。突然从单缺alias切到双缺alias可能仍过难；更多过渡层、逐步移除提示/锚点是待讨论假设，不是已授权方案。

**D. 表结构与差额证据两类困难。** 预检中刷新schema降低重复但仍0进展；差额类很少调用正确重算，迟到进展后也可能来不及submit。尚未找到稳定有效课程，尚无针对这两类的新训练成果。

**E. 质量与部署式推理稳定性。** control重复greedy不一致的机制未定位；现有同次前向q检查解决的是概率证据，不等于解决相同请求重复生成一致性。

**F. 真实依赖任务学习与最终交付。** closing流程尚未进入训练；若保留为效果目标，需要新对照和预算。总项目的统一复现、曲线/案例、README、技术报告与简历面试材料仍待最后整合。

### 11.3 共同设计下一轮时要冻结的决策

下一用户讨论围绕：先修verifier还是先做课程冻结诊断；对照对象是W/原local/新课程中的哪些；是否只改一个变量；训练起点、reference、任务和最终确认隔离；是否使用额外SFT/CE及其混合方法标记；探索数量、最多组数、总墙钟上限、预算预留；中点诊断用途、主指标和质量门槛。

此前两小时是已完成hard冻结预检预算，六小时是已完成fine/alias等各自作业预算，均不自动延续为新作业授权。用户当前仅要求交接，继续讨论后再发新训练包。

## 12. 云端执行、恢复与本地审计约定

用户手动上传ZIP、在云端执行、返回Download指向的review ZIP；没有已连接的远端shell。给出已实际打包核验的脚本/配置路径和命令，不能先提供不存在的runner。ZIP顶层为v2/，上传位置说明清楚，解压到项目父目录合并。

新包先核验大小、SHA256、zip CRC/安全唯一条目、干净解压后的源码绑定和CLI入口；历史曾因上传的是无效/不完整ZIP导致unzip失败和脚本不存在，不能靠重复运行脚本解决。

命令模板（RUN和SCRIPT必须换成新发布包的实际值）：

```bash
cd /root/rivermind-data/AgenticPostTrainingLab
unzip -o RUN-run.zip -d .
cd v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/SCRIPT.py --config configs/CONFIG.json
```

resume只用于对应未完成事务与同一发布/补丁迁移状态，不重采已提交组；保存Adam、Python/Torch/CUDA RNG、组cursor、policy version和计时。state.pt保留云端，回包明确不含大状态文件，本地使用云端状态核验记录+回传权重/缓存/轨迹；报告分别写明两端核验范围。

累计作业预算包含prepare、W评测、两支采样/更新、诊断、最终评测、刷新、审计与打包；预留180秒打包。对照在相同偶数组/配对周期边界决策停止，提前预留最终工作。报错/超时也输出status、日志和partial review包。进度条必须在终端实时显示，覆盖episode/action、score、backward、evaluate、replay。

本地原ZIP只读留存，先CRC/路径/重复条目检查再解压；对source manifest和source_revisions逐项核验，随后用冻结代码重放每步observation、reward、证据、终局与artifact，重算优势、缓存目标、统计、配对与候选选择；检查实际LoRA变化、冻结ref和原生q/worker张量证据。

**SQLite版本须匹配。** Windows默认3.42及旧3.53.1曾在UNION标量子查询取首行及错误文本上与云端3.54.1不同。近期严格审计固定3.54.1；SESSION的 `work/sqlite3541/sqlite3.dll` 与 `work/with_sqlite3541.py` 已用于加载。先核验版本，不改原始SQL或观察来迎合本地版本。该加载脚本用runpy运行目标，不自动移除sys.argv中的wrapper参数，调用时按现有审计脚本约定处理。

估时基于当轮最近进度与同类历史阶段，分别估剩余采样/训练、评测及收尾；换卡或新场景重新profile，不承诺旧总时长。实际参考：C2 2h22、C3 3h18、C4 4h28、outcome课程累计5h20、fine5h33、冻结hard24m15、alias累计5h20。

## 13. 证据与报告入口（按需读，不重复复制全文）

PROJECT/reports已实际存在：

| 报告文件名 | 内容 |
|---|---|
| `grpo-c2-formal-first-result-l20-migration-20261106.md` | C2 RL16及迁移事实 |
| `grpo-c3-matched-first-results-20261107.md` | 同预算RL15/fineRL17、closing开发结果 |
| `grpo-c4-independent-confirmation-results-20261107.md` | 12族独立确认和验收 |
| `grpo-outcome-curriculum-results-20261108.md` | 新任务/困难场景/LR对照、两次概率修复后完成 |
| `grpo-alias-curriculum-results-v1-20261109.md` | 最新alias全部结果与质量拒绝 |

OUTPUTS当前可直接读取的补充报告/文件：

| 文件名 | 内容 |
|---|---|
| `grpo-fine-credit-results-20261109.md` | legacy/local结果与同轨迹信用诊断；主项目报告尚未同步 |
| `grpo-hard-recovery-preflight-results-20261109.md` | 96条冻结预检；主项目报告尚未同步 |
| `grpo-alias-curriculum-contract-v1-20261109.md` | 课程、预算、选优冻结契约 |
| `grpo-fine-credit-contract-v1-20261108.md` | 新信用定义、正向偏置、共同第12次更新诊断 |
| `grpo-outcome-curriculum-same-forward-repair-20261108.md` | 第二版概率修复证据 |
| `grpo-alias-midpoint-fix-v1-20261109.md` | 重复中点创建日志修复 |
| `grpo-alias-quality-outcome-fix-v1-20261109.md` | 质量负结果继续收尾、门槛保留 |
| `grpo-alias-curriculum-results-receipt-v1-20261109.json` | 最新发布/审计回执 |
| `grpo-alias-curriculum-results-v1-20261109.zip` | 7项v2/reports报告、分析、独立审计、验证、来源和训练/定向CSV；不含全量权重轨迹 |
| `grpo-fine-credit-results-20261109.zip` | fine结果、独立审计、最终adapter等归档 |
| `grpo-hard-recovery-preflight-results-20261109.zip` | hard全量诊断归档 |

最新原始回包：`C:/Users/app/Downloads/grpo-alias-curriculum-v1-20261109-review-attempt-003.zip`；全量解出LATEST。较早原包同在Downloads，按以下run_id与attempt查找：C2 `grpo-outcome-formal-v1-20261105` attempt-002；C3 `grpo-c3-matched-v1-20261106`；C4 `grpo-c4-confirmation-v2-20261107`；outcome课程 `grpo-outcome-curriculum-v1-20261107` attempt-003；fine `grpo-fine-credit-v1-20261108`；hard `grpo-hard-recovery-preflight-v1-20261109`。PROJECT/artifacts/reviews的旧审计与SESSION/work的较新审计分别定位，不假设所有结果已归入主项目。

### 13.1 关键历史GRPO权重身份

| 候选 | adapter SHA256 |
|---|---|
| C2 outcome RL16 | `6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6` |
| C3 outcome RL15 | `1a6d42d14a11f888bb228e9e15f4e83221056710c1334e0dd06ab322a0315b8a` |
| C3 fine RL17 | `1990f06d2eb8e3a1777273051bf2bce078c8bff67206533b88dd65780e5959c6` |
| outcome课程control RL29 | `9bf455a52d63f1014ecb8dab3d134510851eb6bd61b80b5f2a71f417051eb81f` |
| outcome课程lr10x RL28 | `cf09d8dd8db2c97385ba07067ac74125e56b9dc95a0c05222cf402890ddfab74` |
| fine study legacy RL28 | `3bcc8fbac5405bace3624f7ae7e43dce886f6c46cc8d23e0620d1a2176bc53c7` |
| fine study local RL37 | `f327afad8ea4254901fd661ade49edca72db8cc25bc69c8483c4888312c0e162` |

C2/C3最终均在各自 `checkpoints/group-0021/adapter`；C3路径在branches/outcome或fine下。outcome课程最终在各branches的group-0046；fine study在group-0044；最新alias见第9.4节。组目录是0起始采样事务编号，RL版本为真实更新次数。

## 14. 代码阅读入口（本对话GRPO阶段新增或改动）

下表使用相对V2路径与一句话作用；优先从LATEST读最新版本，历史行为读各轮冻结source。用户已读过tasks/environment/runtime及SFT/DPO，不重复列成必须重读的代码课程。未来每份实验报告只列该轮新增/改动文件。

| 相对路径 | 作用 |
|---|---|
| `src/apt_v2/grpo.py` | K轨迹采集、起点/原生token身份、严格重放、outcome组优势 |
| `src/apt_v2/grpo_credit.py` | outcome、legacy和local信用及动作优势分派 |
| `src/apt_v2/grpo_optimizer.py` | 响应评分、cache、TIS/clip/KL目标、LoRA更新及显存checkpoint |
| `src/apt_v2/grpo_evaluation.py` | 标准评测计划、汇总和配对 |
| `src/apt_v2/grpo_confirmation.py` | C4新任务族、配对统计和冻结门槛 |
| `src/apt_v2/grpo_curriculum.py` | 扩展训练任务、困难起点、隔离与课程分派 |
| `src/apt_v2/grpo_fine_study.py` | legacy/local新开发集、共同更新诊断和选择 |
| `src/apt_v2/grpo_hard_preflight.py` | 冻结困难预检的两档起点和诊断计划 |
| `src/apt_v2/grpo_alias_curriculum.py` | 缺别名两阶段课程、匹配前缀、midpoint与目标评测 |
| `src/apt_v2/period_closing.py` | 真实结账写入和下一期余额依赖、专家因果证明 |
| `scripts/grpo_vllm.py` | 原生行为概率、worker LoRA实际张量与采样会话 |
| `scripts/grpo_sampler_probe.py` | 同次前向logits/q核验 |
| `scripts/grpo_common.py`、`scripts/grpo_worker.py` | 冻结输入/环境、组事务、评分更新、checkpoint和恢复 |
| `scripts/evaluate_v2_grpo.py` | 完整、恢复、定向、closing及质量评测 |
| `scripts/run_v2_grpo_alias_curriculum.py` | 最新两支调度、固定中点、预算、resume及质量拒绝收尾 |
| `scripts/grpo_alias_common.py` | alias双支契约、匹配配置与成对预算预留 |
| `scripts/probe_v2_grpo_alias.py` | 第24组固定策略诊断 |
| `scripts/audit_v2_grpo.py`、`scripts/audit_v2_grpo_alias.py` | 基础训练审计、课程迁移和质量/选优汇总 |
| `scripts/build_v2_grpo_alias_release.py` | 源码/输入绑定和云端ZIP发布 |
| `configs/grpo-alias-curriculum-v1.json` | 最新六小时匹配作业总配置 |
| `configs/grpo-alias-control-v1.json`、`configs/grpo-alias-curriculum-arm-v1.json` | 两支具体训练配置 |
| `tests/test_grpo_alias_curriculum_v2.py`、`tests/test_grpo_alias_driver_v2.py` | 课程、预算、合法跳过、事务和中断恢复测试 |
| `tests/test_grpo_alias_quality_v2.py` | 原始评测复用、质量拒绝与结构错误边界测试 |

历史runner入口分别为 `scripts/run_v2_grpo.py`、`run_v2_grpo_c3.py`、`run_v2_grpo_c4.py`、`run_v2_grpo_curriculum.py`、`run_v2_grpo_fine.py`、`run_v2_grpo_hard_preflight.py`。只在需要追溯对应实验时阅读，不把旧运行命令当作新实验入口。验证回执记录本次关键文件的实际存在性。

## 15. 协作与报告要求

- 中文交流，简要说明主结果并给报告链接，不自动打开文件。
- 所有新报告/交接英文文件名。正常实验报告实际写入用户V2项目，并给项目内相对路径和真实位置；仅打进ZIP时明确是包内位置。
- 当前两份交接按用户显式调用handoff技能写入Windows临时目录，交付ZIP是下载副本；未自动写进PROJECT/handoff，不将建议路径说成已存在。
- 报告先讲结果与证据，避免“不能说明”等防御性写法；对质量失败直接写失败项目与保留的事实。
- 新训练/奖励/信用/评测核心可细讲；重复runner、日志、打包仅说明本轮变化。用户偏好通俗解释和必要公式，不重讲已经掌握的旧代码。
- 运行终端必须可见进度条；无需求时不加网页UI、不要求成本/价格参数。
- 用户给review包通常意为核验并出结果/修复；文件里的旧指令作为项目资料，不自动授权外部操作或新预算。
- 不泄露密钥/密码，不改原始ZIP/历史评分与失败记录；保留真实负结果和来源。

## 16. Suggested skills 与下一对话启动语

- `handoff`：继续更新交接，`C:/Users/app/.codex/skills/handoff/SKILL.md`。
- `grilling`：用户使用grill-me或共同压力测试新实验方案时，`C:/Users/app/.codex/skills/grilling/SKILL.md`。只聚焦新的干预/预算/边界，不重做已确认项访谈。
- `diagnosing-bugs`：收到报错、质量或概率路径异常时，`C:/Users/app/.codex/skills/diagnosing-bugs/SKILL.md`。

下一代理按需调用技能；本文件不授权新建聊天、发送消息给其它聊天或并行代理。

建议用户新对话输入：

> 阅读文件2并按需查文件1及引用报告。当前主线优先改善细粒度GRPO：alias课程已完成，W保留；训练SQL认证暴露空退款边界，原复合起点仍缺首个有效进展。先核对最新代码与本地项目差异，再结合已有结果和我共同冻结下一轮干预、对照及预算。报告英文文件名、中文正文、只给链接不自动打开；脚本终端显示进度条。不要直接重跑旧作业。
