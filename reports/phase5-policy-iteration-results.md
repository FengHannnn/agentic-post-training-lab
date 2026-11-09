# Phase 5：候选刷新 rollout 与第二轮有限更新结果

日期：2026-10-28。本轮已完成“首轮候选 → 新采样 → 新 old → 继承 Adam 的有限更新 → 三方生成评测”，并完成本地原始记录复核与 CPU 权重/状态重建。工程闭环通过；训练诊断严格成功增加2题，开发严格成功仍为36/56，未证明新的开发收益，不晋升基线。

## 实际执行与策略身份

统一调度的 refresh、update 两阶段退出码均为0，未自动重试或提前停止。

- base：固定 SmolLM2-135M-Instruct revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`。
- sampler / 本轮 old / 初始 new：首轮 GRPO 最终候选，完整参数摘要 `853c4027c91b968faadba67d46e3e9f2f239f107ab7bf8e58f276b0a643fe55b`。
- reference：原始冻结 SFT，完整参数摘要 `9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb`。本轮 reference 与 old 明确不同。
- 用户选择延续父 Adam 动量，累计 step 从16到31；没有把重置优化器的分支混入本轮。
- 新候选完整参数摘要：`70804ab1c9344f7578396e4ccaa90dfe1ea374c69f96cb69ee3970d036eedf7d`。

输入、reward、串行训练路径、学习率与停止门遵循准备方案。新 checkpoint 输入 digest 为 `d381207a2a62d1e4ccbaaac9364249e4db3eada7cc0f6cfa612fe5bbe6584407`，本地已依据候选/reference、配置/源码、实际新 rollout 和规则生成的组批次独立重算。旧 rollout、旧 checkpoint digest 和冻结基线保持原样。

## 新采样与信号

144个数学训练输入，每题8条，共1152条新回答；全部保留。原生回答 token 共32485个，1150条 EOS、2条 length；严格 reward 总和583，语义诊断751，坏JSON37。

| 分组 | 数量 |
| --- | ---: |
| 全对 | 35 |
| 全错 | 50 |
| 混合奖励 | 59 |
| 混合组内回答 | 472 |

同批每组前4条可发现41个混合组，全部8条发现59组，多18组。新59组与旧62组来自不同策略和seed，不能只按组数判断学习进退。旧批严格572/1152、新批583/1152也不构成同seed的受控收益证明。

按问法，严格成功为自然103/384、顺序171/384、符号309/384。训练侧old/new无更新最大logp差为0，ratio=1。sampler/trainer最大token logp差仍为0.1411747932；TIS权重约0.86834—1.10260，无cap触发。新策略刷新并没有解决既有跨引擎概率差异。

## 有限更新、恢复与参数

按预先固定的 case ID 顺序使用全部59个混合组一次，472条候选回答，15个逻辑更新步（14批×4组、末批3组）。另有一次丢弃的第2步恢复探针，因此本轮实际 optimizer 调用16次、backward504次；候选累计步数为16+15=31，不能把额外探针算入候选训练轨迹。

Adam导入前后状态误差0。第1步 checkpoint 的本轮step为1、Adam累计step为17；最终checkpoint的本轮step为15、Adam累计step为31，120份状态的名称、shape、dtype、有限性与步数通过本地核对。

第2步恢复探针在云端记录的adapter误差0、Adam误差0，loss完全相同。最终重新加载后8条logp差0、8题greedy token一致。该数值恢复/前向检查是绑定可信源码的GPU运行证据，本地没有重新执行GPU恢复探针。

本地CPU加载父候选、第1步与最终adapter，独立重建完整参数摘要及冻结base摘要；120/120 LoRA张量相对父候选发生变化，最终最大元素变化约0.0001504663。冻结base摘要始终为 `e72fa38da3ed620a2f0879d8f11b0a6fe908e965f058b95b52738417bae1a4b6`。原SFT reference参数摘要也重新构建匹配。没有本地模型前向、反向或optimizer更新。

本轮没有提前停止：梯度范数约0.16917—0.55964；sampled k3最高0.00225897，ratio超区间比例最高约0.74074%。个别token的ratio可超出[0.8,1.2]，PPO clip不是对真实ratio的硬限制；停止门统计的是比例。

## 三方真实生成比较

同一L20、FP32/eager、相同模板和greedy参数，三模型各评测48个训练诊断与56个开发输入，共312条主预测；重载探针另计。

| 指标 | 原SFT | 首轮候选 | 本轮候选 |
| --- | ---: | ---: | ---: |
| 训练诊断严格 | 28/48 | 28/48 | 30/48 |
| 训练诊断语义 | 35/48 | 36/48 | 38/48 |
| 开发严格 | 36/56 | 36/56 | 36/56 |
| 开发语义 | 44/56 | 44/56 | 44/56 |
| 主预测length | 0 | 0 | 0 |

本轮相对首轮新增两道训练诊断严格成功，无严格退步：

1. `dpo-v2-train-14-55-26-symbolic-right_plain`：`14*(55+26)` → `14*55+26`。
2. `dpo-v2-train-23-91-53-symbolic-left_grouped`：`91+53*23` → `(91+53)*23`。

两题都在本轮59个直接更新组内。这是训练输入上的行为改善，不能当作新任务泛化。

开发只有一道题的输出文本相对首轮变化：`dpo-v2-development-30-103-61-natural-left_plain` 从 `(61*30)` 改为 `(61+30)*30`，前后都不满足目标表达式和结果。总分持平的同时，本地逐题核对也确认无新增JSON/schema/route退化；不存在被总分掩盖的严格成功互换。

工程检查和行为安全门均通过，`development_strict_net_gain_vs_previous=0`，`baseline_promoted=false`。开发集已在历史阶段使用，不是最终盲测；训练诊断也不是新增独立测试题。

## 本地复核与边界

输入ZIP：`grpo-policy-iteration-review-20261028.zip`；SHA256：`8b0096178ee2a2d5971411cdce45f7c50b16700229d7cc5a3830d8796a7e1f04`。

先检查178个成员的重复/大小写冲突、路径、符号链接、大小及CRC，再安全解压。重新核验新采样7个清单产物/71个源码输入，训练94个清单文件/71个源码输入；源码快照与本地可信源完全匹配。父产物、baseline/base缓存也重新验证。

本地完成：1152条新rollout的原生token/seed/解码/reward/advantage与评分张量重审；312条主预测重解码、严格规则重评分、三方逐题比较；15步history计数与loss关系/停止门/费用重算；1152份保存reference分数的对齐、shape与有限性检查；CPU权重和Adam状态重建。SFT与上一候选的208条主预测与首轮实验保存记录完全相同。

精确JSON比较在本地发现一处浮点末位差异：重算的 `correction_weights.min_weight=0.8683375189005751`，云端保存为 `0.868337518900575`，差约1.11e-16；其余summary字段全部完全一致。保留原summary与该差异记录，未改任何训练门限。该差异出现在跨本地/云端运行时的汇总计算中，未逐算子定位原因。

reference的GPU运行冻结、恢复探针和重载前向一致性依赖绑定源码与日志。本地对reference保存分数只做记录和张量核对，没有重新计算1152条reference前向，也没有GPU前向/反向重演。CPU加载对应权重能验证参数身份，不能替代这些GPU运行检查。

采样阶段478.13秒，训练/评测阶段467.88秒，合计946.01秒，约15分46秒；按2.18元/小时估计0.57286元。该数值排除阶段外预检/打包以及开机、空闲、存储和计费取整，不等于账户总账单。脚本未自动关机。

## 本轮结论与后续边界

本次预定的候选刷新和有限后续更新已完成，old/reference分离与Adam跨轮延续得到实际验证。现在有两个有限优化轮次及一次更新后采样刷新，仍不是长期自动在线循环、服务权重热更新或完整多轮工具Agent。

本轮有训练内严格改善，但共同开发集未显示净收益。按既定范围收尾，不自动增加第三轮、学习率搜索或采样量，不晋升基线。Phase 5可据此整理有限实验结论；是否继续收益实验或进入Phase 6 observation/final工具闭环，应另定明确问题和验收范围。

## 代码阅读顺序与重点

1. `configs/grpo_policy_iteration.json`：候选、固定reference、seed、继承Adam及停止门。结合实际59组/15步，理解最多36步不是必须训练36步。
2. `trainers/policy_iteration.py` → `verify_parent`、`select_mixed_batches`：先确认是谁产生新样本，再看为何混合组数量变化且顺序可复算。
3. 同模块 `import_parent_adam`：旧input digest与新input digest、动量无损导入、累计step与本轮逻辑step的不同含义。
4. `scripts/collect_grpo_iteration.py` → `generate`、`score`、`audit`；`scripts/iteration_common.py` → `validate_records`：原生token/q、新old/new评分、reward重审，以及1150 EOS/2 length如何保留并评分。
5. `scripts/train_grpo_iteration.py` → `context`、`train`：新rollout绑定、候选policy与SFT reference分别加载、reference_cache、恢复分支与504次实际反向的来源。
6. 同入口 `compare_three`：对照两题训练改善与开发文本仍错的实例，区分工程通过、安全通过、训练改善和开发收益。
7. `tests/integration/test_policy_iteration_adam.py`、`test_grpo_iteration_audit.py`，再看 `tests/unit/test_policy_iteration.py`：先读真实tiny动量继承/恢复，再读1152条合成记录重审和失败阻断。测试不是本次GPU实验的替代。

上文 `trainers/` 简写前缀为 `src/agentic_post_training/`。本地完整回传产物保存在原计划的 `artifacts/trajectories/grpo-refresh-20261028-01`、`artifacts/checkpoints/grpo-iteration-20261028-01`，独立审计位于 `artifacts/evaluations/grpo-iteration-review`。
