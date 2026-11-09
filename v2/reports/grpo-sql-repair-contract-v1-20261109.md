# GRPO SQL repair study v1

本轮先修复 SQL 认证，再通过固定 W 诊断选择一条训练主路。目标是使最终候选在无提醒任务中超过 W，并改善原复合缺别名终局成功。

## 已确认的共同设置

- run_id：grpo-sql-repair-study-v1-20261109。
- 正式基线 W：SFT513 / DPO11 / RL0，adapter SHA256 为 f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。
- 新任务显式绑定 sql_semantics_v2，旧 Task 与历史回放继续使用 legacy_v1。
- 新 verifier 检查原库、两个金额扰动世界、退款跨月、本月无退款、上月无退款、两月均无退款。空 gross 集合作为本地回归。所有世界按整数语义执行判断，支持等价正确 SQL。
- 本轮 GPU 总墙钟最多21600秒，覆盖准备、诊断、训练、所有评测、刷新、审计和实际打包；打包预留180秒。
- 生成 vLLM BF16；评分与更新 HF/PEFT FP32；LoRA q_proj/v_proj，r8、alpha16、dropout0。
- K4、训练T1、top_p1、top_k=-1、min_p0；回复256、上下文8192；原生token/q和同次前向证据绑定。local 信用、GRPO LR1e-6、beta0.02、clip0.2、TIS cap2、grad clip1、epsilon1e-8。每组最多一次 optimizer.step，合法零信用组跳过且保留。

## 任务、提示与数据隔离

训练池沿用6个原train和6个扩展train。诊断从中冻结旧/新各4个，覆盖有/无临时查询故障。新开发集为 repair-development-000至003，seed83000至83003，2036/2037年份，月份1/4/8/12，和历史开发/C4身份隔离；本轮属于开发选择。合格候选之后另冻最终独立确认。

恢复前缀仍真实执行，总预算12步包含前缀。原复合与提醒组具有同一环境起点、动作数和剩余预算；提醒只在起点追加一次公开规则文本，明确两个别名与 verified_metrics，不提供SQL答案、金额、内部seed或gold。提示额外tokens单独记录。简化参考保留真实上月认证及其SQL示例。

冻结任务、96条计划、24个监督样本和SFT顺序位于 artifacts/inputs/grpo-sql-repair-20261109/。监督样本逐条真实重放，监督响应包含EOS，prompt labels为-100，拒绝train身份以外的数据。

## 96条冻结诊断与路由

8个train任务×K4×3条件：repair_compound、repair_reminded、repair_simplified；W固定、零更新、配对seed。每个条件32条、8组。

信号达标同时要求：首动作取得本月或上月新认证至少8/32；至少4/8组同时具备新认证与非零动作信用。

| 条件 | 路线 |
| --- | --- |
| 原复合达标 | online_plain：单个无提醒GRPO候选对W |
| 原复合未达标；提醒组达标且首认证净增至少4/32 | online_reminder：control与提醒渐退课程双支 |
| 其余情况 | warm_start：SFT B，再按门槛决定GRPO C |

路由只读取train诊断结果，不根据开发收益改变本轮路线。

## 在线训练

每支最多36个采样组，维护18组、目标18组，W初始化、冻结W reference、新AdamW。维护覆盖完整任务和标准恢复；目标为原复合缺别名。

提醒路线按3个固定阶段执行，每阶段12组，其中6个目标组；提醒目标组数量依次6、3、0，即100%→50%→0%。control始终无提醒。两支任务、seed、组数和阶段边界配对，阶段间保留Adam/RNG；实际更新数分别报告。无中点模型搜索，训练日志记录首次认证和有效组。

## SFT B与GRPO C

SFT数据为12个复合train起点首个正确修复动作，加12个维护动作；维护中query_sql、calculate_change、check_result、submit各3个。共24动作，4遍96例次，batch4、24次更新、LR1e-5、grad clip1；无DPO、无额外CE混入GRPO。只评最终B。逐步保存adapter、Adam/RNG与动作索引，恢复沿同一预定顺序继续。

B门槛：四项标准成功计数均不低于同轮W，全部质量检查通过；无提醒原复合sample首动作认证至少3/12且高于W。未通过则结束该路线并保留结果。

C从B初始化、冻结B reference、新建AdamW；source血缘SFT537 / DPO11 / RL0，之后单独累计RL更新。C最多36组、维护与目标各半，目标始终无提醒，采用共同local/GRPO超参数。B的Adam状态保留在SFT作业，C不继承。

## 评测与选择

每模型标准132条：完整greedy4、sample16；标准恢复greedy28、sample84。另做无提醒原复合greedy4、sample12，共148个episode；重复greedy七上下文各两次，14次质量动作单列。采样评测T0.8，完整每task4次、恢复每context3次；各模型计划与seed配对。

候选门槛：四项标准指标均不低于W；主指标严格增加；原复合sample终局至少比W增加2/12，greedy不下降；全部质量检查通过。若W完整sample达到16/16，主指标为标准恢复sample；否则主指标为完整sample。

C另需相对B四项标准指标不下降，并在标准指标或原复合终局成功中至少一项增加。B−W、C−B分别报告，B单独超过W时登记SFT成果。

多候选合格时，按主指标、复合sample、另一标准sample、复合greedy、完整greedy、恢复greedy顺序排序，完全相同时优先control。

## 预算、恢复与负结果

依据本轮W实际评测耗时，以1.25倍且至少2400秒/模型预留候选评测，另留刷新、审计与打包预算；配对周期开始前检查剩余预算。预算不足时在配对周期边界收尾。完成36组配方的模型进入正式候选选择，提前停止结果保存为预算诊断。

基础设施错误、中断或质量拒绝均保留状态、日志与review包。语义质量拒绝写为结果，结构/源码/token/概率/状态身份错误仍致命；审计失败撤销候选。resume核对发布源码、配置、GPU UUID、软件、SQLite及累计时间，复用已提交轨迹和更新，不重新抽取已提交组。

原始包保留.pt状态在云端，review排除.pt；回包包含真实LoRA、评分缓存、token/q、worker证明、轨迹、源码与审计。final_status.json在主体打包后追加，外部receipt记录归档哈希及包含打包的累计时间。旧driver累计时间先于打包的历史记录继续保留，新作业使用本轮计时。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| sql_verifier.py | 新版SQL认证世界与整数语义检查 |
| environment.py | 按任务显式选择认证版本 |
| grpo_repair_study.py | 新任务、公开提醒、路由、课程与验收 |
| grpo.py | 将新任务的完整起点提示绑定到重放身份 |
| grpo_curriculum.py | 新实验的任务、起点与评测分派 |
| grpo_common.py | 支持W/B来源计数与输入绑定 |
| grpo_worker.py | 新实验采样与C来源计数接入既有更新事务 |
| grpo_vllm.py | 新实验接入同次前向原生概率证明 |
| evaluate_v2_grpo.py | 新上下文重放、定向评测和质量负结果收尾 |
| grpo_repair_worker.py | 冻结诊断、SFT数据、24步热启动与B审计 |
| run_v2_grpo_repair.py | 六小时路由、配对训练、门控、恢复与打包 |
| audit_v2_grpo.py | 新实验合法零信号审计接入 |
| audit_v2_grpo_repair.py | 全局重放、SFT/GRPO与选优重算 |
| validate_v2_grpo_repair.py | 本地专家、token预算和监督掩码验证 |
| build_v2_grpo_repair_release.py | 云端包CRC、条目和逐文件哈希验证 |
| grpo-sql-repair-study-v1.json | 作业预算、路由与SFT配置 |
| grpo-sql-repair-branch-v1.json | 共用GRPO分支模板 |
| test_grpo_repair_study_v2.py | 认证、真实重放、路由与目标门槛测试 |
| test_grpo_repair_driver_v2.py | 原生token、LoRA/Adam恢复与调度测试 |
