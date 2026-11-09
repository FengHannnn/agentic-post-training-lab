# Fine GRPO Credit Comparison Results

日期：2026-11-09。运行：`grpo-fine-credit-v1-20261108`。

本轮正常完成，各分支训练 46 组，按预定预算规则成对停止，累计耗时 **5 小时 33 分 03 秒**。云端与本地独立审计均通过。**新 local 相比旧 legacy：采样完整任务净增 1 条、恢复任务净增 3 条；相对 W：完整净增 1 条、恢复净减 1 条，本轮合格候选为 0，W 继续作为正式基线。**

报告项目归档路径：`reports/grpo-fine-credit-results-20261109.md`。

项目归档包顶层为 `v2/`，包含本报告、独立审计、逐例分析和最终适配器。将归档包解压至项目父目录后，报告对应云端路径为 `/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-fine-credit-results-20261109.md`。原始 review ZIP 作为完整轨迹档案保留。

**比较模型与方法**

| 模型 | 初始化与信用定义 | 学习率 | 训练组数 / 实际更新 |
|---|---|---:|---:|
| W | 最佳加权 SFT，冻结参照 | — | 0 / 0 |
| legacy RL28 | 从 W 独立开始；`potential_rtg_depth_v1`，沿用 fine RL17 的信用规则 | 1e-6 | 46 / 28 |
| local RL37 | 从 W 独立开始；`outcome_local_verified_v2` | 1e-6 | 46 / 37 |

legacy 为本轮重新训练的模型。两支各自新建 AdamW，reference 固定为 W；基础模型、课程、任务顺序、采样种子、K=4、解码参数、LoRA、KL、clip 和 TIS 配置一致，分支配置仅 run_id 与 credit_method 不同。本轮额外 SFT/DPO 更新均为 0。

local 使用 `A[i,t] = A_outcome[i] + 0.2 × 本动作新验证指标数`。局部加成与前缀、先前动作去重，重复取证和无验证进展动作的局部部分为 0，终止时 potential 清零不扣局部分。该方法按约定记录为 outcome 与局部验证信号混合的自定义动作信用。

两支各完成 23 个完整任务组、23 个恢复组，实际覆盖预设 24 个训练任务中的 23 个；训练轨迹共 368 条，最后刷新另有 8 条。开发评测使用 4 个全新任务族，排除历史 C3、C4 和上一轮 curriculum 开发任务。

**最终开发评测**

| 指标 | W | legacy RL28 | local RL37 |
|---|---:|---:|---:|
| sample 完整 | 11/16，68.75% | 11/16，68.75% | **12/16，75.00%** |
| sample 恢复 | **60/84，71.43%** | 56/84，66.67% | 59/84，70.24% |
| greedy 完整 | 4/4 | 4/4 | 4/4 |
| greedy 恢复 | 20/28 | 20/28 | 20/28 |
| 新增困难恢复，greedy + sample | 0/24 | 0/24 | 0/24 |

W 的 sample 完整任务成功率尚未饱和，因此主指标为完整任务成功数。冻结规则同时要求采样完整、采样恢复和两项 greedy 成功数不低于 W，且主指标增加。legacy 主指标增量为 0、采样恢复减少 4 条；local 主指标增加 1 条、采样恢复减少 1 条。选优结果 `selected_candidate=null` 与本地复算一致。

配对比较按相同任务、场景、模式与 rollout 对齐：

| 比较 | sample 完整：改善 / 退步 | sample 恢复：改善 / 退步 | greedy 恢复：改善 / 退步 |
|---|---:|---:|---:|
| legacy 对 W | 0 / 0 | 1 / 5 | 1 / 1 |
| local 对 W | 1 / 0 | 1 / 2 | 1 / 1 |
| local 对 legacy | 1 / 0 | 3 / 0 | 1 / 1 |

local 对 legacy 的采样完整提升为 +6.25 个百分点，采样恢复提升为 +3.57 个百分点。本轮两项 greedy 总成功数相同，具体成功任务存在一进一退。

**具体改善与剩余回退**

local 相对 legacy 的四条采样改善均没有同类配对退步，rollout 使用 0 起始编号：

| 任务 | 场景 / rollout | 变化 |
|---|---|---|
| fine-development-000 | 完整 / 1 | legacy 未取得有效指标证据并耗尽 12 步；local 按有效查询、差额计算、检查、提交完成，共 7 步 |
| fine-development-001 | sql_unverified_requery / 1 | legacy 重查后重复失败检查；local 补做 calculate_change，得到差额 8640 和 c4 证据，4 个继续执行动作完成 |
| fine-development-002 | format_repair / 1 | legacy 持续查询后耗尽预算；local 取得有效证据并重算差额，完成检查与提交 |
| fine-development-003 | failed_check_repair / 0 | legacy 重复失败检查；local 修正后检查通过并提交 |

local 相对 W 的两条 sample 恢复回退是 `fine-development-000 / repeat_sql_error / rollout1` 与 `fine-development-003 / repeat_evidence_error / rollout1`。前者仍在查询、格式和无效证据计算之间徘徊；后者重复执行失败的 check_result。greedy 中 local 改善 `002 / repeat_sql_error`，同时退步于 `002 / failed_check_repair`，后者重复失败检查至预算耗尽。

**信用信号实际改变了什么**

| 训练统计，不含刷新 | legacy | local |
|---|---:|---:|
| 实际更新 / 训练组 | 28/46，60.87% | 37/46，80.43% |
| 跳过组数 | 18 | 9 |
| 非零动作优势数 | 871/1373 | 952/1370 |
| 完整任务组更新 | 20 | 22 |
| 恢复任务组更新 | 8 | 15 |
| 终局回报零方差组 | 20 | 20 |
| KL / clip 保护跳过 | 0 | 0 |

local 的更新增加集中在更多有验证进展的组。format_repair、unknown_tool_repair、stale_evidence_repair 各两个训练组均为 8/8 成功：legacy 因旧信用为零全部跳过，local 在这些组的新验证动作上得到局部加成，共执行 6 次更新。

为分离信用规则与采样路径差异，本地还在同一批已采集轨迹上同时重算两种规则：在 legacy 的 46 组原样轨迹上，旧规则得到 28 个有效组、871 个非零动作，新规则得到 37 个有效组、1005 个非零动作。在终局回报相同的组中，旧规则给 18 个没有新验证进展的动作分配正优势，新规则对这类动作的优势为 0；在 local 自身轨迹上，同一统计为 64 → 0。

实际 28 与 37 次更新的差额同时伴随两支采样路径变化。以上同轨迹复算单独呈现规则本身的变化，属于离线诊断。

**困难恢复仍缺少学习信号**

| 新增训练场景，每支 | 轨迹成功数 | 组数 | legacy 更新 | local 更新 / 局部加成 |
|---|---:|---:|---:|---:|
| missing_alias_pair | 0/8 | 2 | 0 | 0 / 0 |
| schema_query_repeat | 0/12 | 3 | 0 | 0 / 0 |
| requery_stale_change | 0/12 | 3 | 0 | 0 / 0 |
| 合计 | 0/32 | 8 | 0 | 0 / 0 |

三类场景每支均全组失败、终局回报相同，且继续执行动作没有新增验证进展，旧 fine 与 local 均跳过这些组。三模型最终专项评测均为 0/24。

这解释了本轮新信用的作用范围：它修正了部分信用方向，并增加了已有验证进展上的更新；最困难状态仍需要采样产生有效 SQL 取证、差额重算或成功修复轨迹。

**共同第 12 次更新诊断**

| 项目 | legacy | local |
|---|---:|---:|
| 首次达到 12 次更新时，已处理组数 | 20 | 16 |
| sample 完整诊断 | 3/4 | 4/4 |
| 困难恢复 sample 诊断 | 0/12 | 0/12 |

两支 checkpoint 均按实际第 12 次更新锁定，16 条诊断轨迹逐条重放。local 配对改善 1 条、退步 0 条。该节点完整任务样本数为 4；两支此前已处理组数分别为 20 和 16。该结果作为共同更新数的诊断记录，模型选择使用最终标准开发评测。

**预算与环境**

本轮 error=null、stop_reason=budget_reserve。处理 46 组后，预估下一配对周期约 654 秒，加最终工作预留约 6301 秒，合计约 6955 秒；当时剩余约 6774 秒，按事前规则在两支相同偶数组数停止。最终完整评测、共同节点诊断、刷新和审计全部完成，实际总耗时 19983.28 秒，距离 6 小时上限余约 26 分 57 秒。

两支实际采样组数和最终评测节点相同，最终标签均为 grpo_g0046。最后一组 0045 跳过，两份最终适配器均保存在各自 checkpoints/group-0044/adapter。

GPU 为 NVIDIA L40S，UUID `f672f938-9f65-a76d-089e-ab1402ca1b76`。388 份资源快照均指向该卡。Python 3.11.17、Torch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.57.6、PEFT 0.17.1、SQLite 3.54.1，两支身份一致。

| 模型 | adapter_model.safetensors SHA256 |
|---|---|
| W | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| legacy RL28 | 3bcc8fbac5405bace3624f7ae7e43dce886f6c46cc8d23e0620d1a2176bc53c7 |
| local RL37 | f327afad8ea4254901fd661ade49edca72db8cc25bc69c8483c4888312c0e162 |

**独立审计与归档**

原始回包 302787892 字节、2183 项，CRC、重复项及路径检查通过。回包 SHA256：`4bbb1a1b5b53921c5872c889f0115d1b070bb034272fd73e1803494747ef9c00`。发布包 SHA256：`c4ecf159a6f3711ba1b8fef19aaf3ecf041b77ddff5a6bd241f172920c689e91`。

根运行与两支各 236 个冻结源码文件均与已发布运行包一致，三个配置与发布内容一致。本地使用回包冻结源码和与云端相同的 SQLite 3.54.1 完成重放：376 条训练/刷新轨迹、468 条最终评测轨迹、32 条共同节点诊断轨迹，以及 42 次 greedy 质量动作；本地全部语义审计字段与云端一致。

65 次真实更新的目标函数和指标从不可变评分缓存重新计算，legacy 最大差约 4.77e-8，local 约 5.15e-8。94 个采样/刷新组采用 same_forward_sampler_v2，共 1369 个观察 token 的同次前向误差均为 0。额外 prefill 诊断最大差约 0.735，原生同次前向准入均通过；本轮另生成 94 个 prefill 探针 token。两支最大 sampled k3 分别约 7.19e-5、1.20e-4，clip fraction 均为 0。

云端审计核对保存的 Adam/RNG 状态及计数，state.pt 留在云端用于续跑记录；本地复核覆盖返回的权重、事务、评分缓存、原生 token 与环境轨迹。原始 ZIP 保持原样。

证据归档位置：`artifacts/reviews/grpo-fine-credit-v1-20261108/`。归档包包含本地独立审计、云端审计、信用与轨迹诊断、训练和评测曲线、配置、冻结源码哈希、共同节点结果和两支最终适配器。

**后续方向**

local 保留为后续信用方案对照，W 保持正式基线。下一轮优先做困难恢复课程预检：从单次错误逐步扩展到复合错误，固定 local 和学习率 1e-6，先记录每类场景的新验证进展率、成功轨迹和有效组比例，再确定正式训练规模。若引入额外监督训练，按约定另立混合方法记录。

**代码阅读顺序**

以下路径相对 v2 项目根目录：

| 相对路径 | 大致作用 |
|---|---|
| configs/grpo-fine-credit-v1.json | 双分支、6 小时上限与成对停止规则 |
| src/apt_v2/grpo_credit.py | legacy/local 优势计算与动作信用 |
| src/apt_v2/grpo_fine_study.py | 新开发任务、共同节点计划和选优规则 |
| src/apt_v2/grpo_curriculum.py | 训练日程、困难起点与隔离检查 |
| scripts/run_v2_grpo_fine.py | 调度、预算、续跑和打包 |
| scripts/grpo_worker.py | 采样、打分和更新事务 |
| src/apt_v2/grpo_optimizer.py | 概率评分、损失与 LoRA 反向传播 |
| scripts/probe_v2_grpo_fine.py | 共同第 12 次更新诊断 |
| scripts/evaluate_v2_grpo.py | 最终完整、恢复与专项评测 |
| scripts/audit_v2_grpo_fine.py | 原生重放、独立复算和配对汇总 |
