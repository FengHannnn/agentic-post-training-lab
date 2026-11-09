# GRPO Outcome Curriculum Results

本轮两支 outcome GRPO 均完成 48 组训练，实际更新分别为 control 29 次、lr10x 28 次，云端与本地独立审计通过。control 的标准采样恢复成功数由 W 的 62/84 提升到 64/84，完整任务采样仍为 7/16；lr10x 的完整任务采样提升至 8/16，但 greedy 恢复从 23/28 降为 22/28。按事前冻结的选优规则，本轮合格候选数为 0，当前正式基线继续使用 W。

项目内路径：reports/grpo-outcome-curriculum-results-20261108.md

云端同步位置：/root/rivermind-data/AgenticPostTrainingLab/v2/reports/grpo-outcome-curriculum-results-20261108.md

**模型与运行范围**

两支都从最佳加权 SFT W 独立开始，使用相同扩展课程、任务顺序、采样 seed 和 outcome 优势，唯一分支配置差异是学习率。各 24 个完整任务组、24 个恢复组，K=4，共 384 条训练轨迹；最终刷新另含 8 条轨迹。训练使用 12 个旧任务和 12 个新任务，开发评测使用另外 4 个新任务。原 C4 独立确认任务已排除。

| 模型 | 学习率 | 本轮组数 | 累计 RL 更新 | 最终权重 SHA256 |
|---|---:|---:|---:|---|
| W / weighted_sft | — | 0 | 0 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| control / outcome GRPO | 1e-6 | 48 | 29 | 9bf455a52d63f1014ecb8dab3d134510851eb6bd61b80b5f2a71f417051eb81f |
| lr10x / outcome GRPO | 1e-5 | 48 | 28 | cf09d8dd8db2c97385ba07067ac74125e56b9dc95a0c05222cf402890ddfab74 |

两支最终 adapter 均位于各自 artifacts/grpo-outcome-curriculum-v1-20261107/branches/<branch>/checkpoints/group-0046/adapter；最后一组 0047 因零方差跳过。评测标签 grpo_g0048 表示已处理 48 组，实际参数更新次数以表中 RL 更新为准。继承的 SFT/DPO 更新分别为 513/11；本轮额外 SFT=0、DPO=0，reference 始终为 W。

每个模型完成 132 条标准开发轨迹、24 条困难恢复轨迹和 14 次质量检查，总计 468 条评测轨迹、42 次质量检查。本轮 closing_evaluation 关闭。

**标准开发评测**

| 指标 | W | control | lr10x |
|---|---:|---:|---:|
| 完整任务 greedy | 3/4，75.00% | 4/4，100.00% | 3/4，75.00% |
| 完整任务 sample | 7/16，43.75% | 7/16，43.75% | 8/16，50.00% |
| 标准恢复 greedy | 23/28，82.14% | 23/28，82.14% | 22/28，78.57% |
| 标准恢复 sample | 62/84，73.81% | 64/84，76.19% | 63/84，75.00% |
| 新增困难恢复 greedy | 0/12 | 0/12 | 0/12 |
| 新增困难恢复 sample | 0/12 | 0/12 | 0/12 |

control 相对 W 的标准恢复采样：3 条由失败变成功，1 条由成功变失败，净增 2 条，即 +2.38 个百分点；完整任务采样逐例结果相同，greedy 完整任务净增 1 条。lr10x 相对 W：完整任务采样净增 1 条，即 +6.25 个百分点；标准恢复采样 6 条改善、5 条回退，净增 1 条；greedy 恢复回退发生在 curriculum-development-003 的 sql_unverified_requery。

本轮采用新开发任务实例，包含新的年份、月份、区域和客群。W 的完整任务采样为 7/16，主指标因此仍为完整任务成功数。冻结规则要求主指标增加、greedy 完整/恢复不回退、采样完整/恢复不回退。control 主指标增量为 0；lr10x 的 greedy 恢复回退 1 条。development_selection.selected_candidate=null 与本地重算一致。困难恢复评测用于诊断。

**困难恢复与训练信号**

| 新增训练恢复场景，每支 | 训练轨迹成功数 | 采样组数 | 实际更新 |
|---|---:|---:|---:|
| missing_alias_pair | 1/8 | 2 | 1 |
| schema_query_repeat | 0/12 | 3 | 0 |
| requery_stale_change | 0/12 | 3 | 0 |
| 合计 | 1/32 | 8 | 1 |

两支在这三类训练场景上的计数完全相同。三模型各 24 条困难恢复开发评测均失败。schema_query_repeat 和 requery_stale_change 的训练组全部以失败结束且用满 12 步，每条 outcome return 都是 −0.12，组内优势为 0，更新被跳过。control 的 29 次更新中，22 次来自完整任务、7 次来自恢复；lr10x 对应为 21 次和 7 次。

代表性错误：missing_alias_pair 中模型重复无 AS current_net_cents/prior_net_cents 的查询，虽然返回数值，verified_metrics 仍为空，随后 calculate_change 持续收到无效证据错误。完整任务失败记录还包含 status='PAID' 与真实 'paid' 不匹配、退款月份混用、缺失指标别名，以及读取数值后继续使用未认证 evidence ID。

本轮主要瓶颈是困难恢复组缺少成功探索带来的相对优势。学习率从 1e-6 增至 1e-5 后，两类全失败场景仍为 0 次更新。固定专家流程在全部 308 个训练/开发起点与场景组合中均能完成，原任务步数预算内存在可行解。

**第二版补丁与审计**

第三次回包正常结束，error=null、stop_reason=group_cap，所有第三次运行阶段退出码为 0。两支各 18 个原始组使用原 prefill 记录，各 31 个后续采样/刷新组使用 same_forward_sampler_v2。共 62 组、902 个观察 tokens 的同次前向概率最大误差均为 0，进入原生采样器的 logits 检查全部一致。两组 prefill 跨路径诊断仍超过原 0.25 阈值，同次前向准入检查通过。

审计逐项核对三份有效源码快照，各绑定 227 个文件；实际安装的第二版补丁与已发布 ZIP 完全一致。两支 96 个训练组及 2 个刷新组共 392 条轨迹完成原生 token 与环境回放，468 条评测轨迹完成回放和统计复算，42 次质量动作通过检查。57 次实际更新的损失与指标由不可变分数缓存重新计算，control 最大差为 5.364418043e-8，lr10x 为 3.725290310e-8。所有 reference 保持不变，same-path 评分差为 0。

云端审计同时核验 Adam/RNG 文件及更新游标；state.pt 保留在云端，回包保存哈希与核验结果。本地独立审计与云端审计的训练统计、目标函数、评测、逐例配对、困难恢复和选优结果一致。原 175 个既有组、事务、检查点与 W 评测文件保持原哈希，解压出的全部 3538 个原始回包文件经复核保持原字节。

本地初始 SQLite 3.42.0/3.53.1 在一条 UNION 标量子查询上得到 24064，云端 3.54.1 得到 21558；涉及两支各一条训练动作，结果均未认证为证据。使用官方、校验和匹配的 SQLite 3.54.1 后，原始动作回放全部一致。最小复现 SELECT 10-(SELECT SUM(x) FROM (SELECT 3 AS x) UNION SELECT 0) 在前两版本返回 10，在 3.54.1 返回 7。SQLite 的标量子查询取首行，未指定 ORDER BY 的多行结果顺序未定义；本地审计因此固定与云端同版本。[标量子查询文档](https://www.sqlite.org/lang_expr.html#subquery_expressions)、[结果排序文档](https://www.sqlite.org/lang_select.html#the_order_by_clause)、[官方运行库](https://www.sqlite.org/download.html)。

运行环境为 NVIDIA L40S，UUID 2ac022d0-8ede-c73c-a21e-ccb5f7a3c88e，Python 3.11.17、Torch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.57.6、PEFT 0.17.1、SQLite 3.54.1；两支 GPU 与软件身份一致。

**实际耗时**

| 范围 | 实际用时 |
|---|---:|
| 第二版补丁后的第三次续跑 | 3 小时 27 分 17 秒 |
| 其中剩余 60 组采样与更新 | 2 小时 10 分 18 秒 |
| 其中两个最终模型评测 | 1 小时 11 分 44 秒 |
| 整轮累计，含前两次尝试 | 5 小时 20 分 17 秒 |
| 原 6 小时预算剩余 | 39 分 43 秒 |

第二版补丁后的实际续跑耗时位于此前 3–3.5 小时估算范围内。

**下一步建议**

优先做困难恢复课程的小规模探索预检：训练从单次错误逐步增加到重复、复合错误，增加探索样本，先确认每类场景能产生成功/失败混合组，再扩展正式训练。学习率 1e-6 保留为对照，记录各场景有效组比例、成功轨迹和证据修复动作。标准开发指标和 greedy 保护条件继续用于选优；出现合格候选后，在结果揭晓前冻结一批全新独立确认任务。若后续引入额外监督或混合方法，另立方法记录。

**代码阅读顺序**

| 相对路径 | 大致作用 |
|---|---|
| configs/grpo-outcome-curriculum-v1.json | 两分支、48 组日程和 6 小时预算 |
| src/apt_v2/grpo_curriculum.py | 扩展任务、错误上下文、专家证明与困难恢复计划 |
| scripts/run_v2_grpo_curriculum.py | 分阶段运行、续跑与累计计时 |
| scripts/grpo_worker.py | 原生采样、分数缓存和 LoRA 更新 |
| scripts/grpo_sampler_probe.py | 同次前向概率观察与校验 |
| scripts/evaluate_v2_grpo.py | 标准评测、困难恢复与质量检查 |
| scripts/grpo_curriculum_common.py | 匹配分支配置与冻结选优规则 |
| scripts/audit_v2_grpo_curriculum.py | 全量回放、目标函数复算和配对比较 |

回包文件：grpo-outcome-curriculum-v1-20261107-review-attempt-003.zip

回包 SHA256：72ef147869cb93ff390b5573b1d254e2a7807f7ca2acf6936b11732b27149cb7

第二版补丁 ZIP SHA256：5c83b52dc686659d3a0bae97b4f04ce84447798ed9cc2c3627da1c05f1ad685b
