# C4 expanded independent confirmation contract

日期：2026-11-07。运行：`grpo-c4-confirmation-v2-20261107`。

用户授权本轮增加2–3小时，并确认v1尚未启动。实际运行使用v2：12个独立任务实例、三模型共1,188条episode，预算上限7小时。原v1运行包与报告保留为准备记录。

## 扩容目的与范围

优先增加独立数据实例与任务族覆盖，保持每任务采样次数、模型权重和生成参数。前8个任务与未启动的v1相同，新增4个任务补齐2027年的3、6、9、12月，合计覆盖全年12个月。seed91000–91011，独立family_id为c4-family-91000至c4-family-91011；6个任务注入首次查询临时故障。id/family_id/seed与原train/development/test及closing-development隔离，数据库分别生成。

任务沿用月度对账业务模板和SQLite规则。每个任务总预算仍为12步，恢复前缀计入预算；脚本专家验证12×8=96个完整/恢复起点在相同预算内可完成。本轮新增训练更新为0。

| 项目 | v1准备方案 | v2实际方案 |
|---|---:|---:|
| 独立任务/任务族标识 | 8 | 12 |
| 每模型sample完整 | 32 | 48 |
| 每模型sample恢复 | 168 | 252 |
| 每模型greedy完整 | 8 | 12 |
| 每模型greedy恢复 | 56 | 84 |
| 每模型episode | 264 | 396 |
| 三模型episode | 792 | 1,188 |
| 正常质量动作 | 42 | 42 |
| 最大总wall-time | 4小时 | 7小时 |
| 恢复成功最低净增 | 4/168 | 6/252 |

## 固定候选

| 标签 | 候选 | adapter SHA256 |
|---|---|---|
| weighted_sft | 最佳加权SFT W，RL0 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| outcome_rl16 | C2结果奖励GRPO，RL16 | 6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6 |
| fine_rl17 | C3细粒度GRPO，RL17 | 1990f06d2eb8e3a1777273051bf2bce078c8bff67206533b88dd65780e5959c6 |

三者均继承SFT513/DPO11，固定Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。sample temperature0.8、top_p1、top_k-1，响应上限256、上下文8192；相同任务/场景/mode/rollout使用相同seed。确认seed2026110700，逐动作加动作序号。

## 预先冻结的验收

每个GRPO候选分别与本轮同机W配对，需同时满足：

1. sample恢复成功至少多6/252，即+2.380952个百分点，与原4/168保持完全相同的提升幅度。
2. sample完整、greedy完整、greedy恢复成功数均不下降。
3. W sample完整若为48/48，则主指标为恢复；否则主指标为完整，完整成功数还需至少增加1。

任一GRPO候选满足全部条件即通过实用确认。审计通过与模型效果达标分别记录。所有候选结果均保留，并报告配对改善/退步、任务与场景级结果、动作与token成本。bootstrap按12个任务族聚类，10,000次，seed2026110711，95%区间用于描述统计。

任务、候选、规模和门槛在推理前冻结。本轮内按固定计划完成，不根据中间成功率追加样本或改参数。若以后用本批结果调参，则本批转为诊断数据，后续最终确认另冻新任务。

## 执行与审计

同一张L40S、三模型各自独立进程、单LoRA编号1；vLLM0.11.0 BF16，原生processed logprob、禁用prefix caching。准备时校验完整base、三份adapter和源码哈希，绑定本次GPU UUID、驱动与软件环境，作业内及续跑保持相同身份。

每session先做7个上下文各两次greedy质量检查和112个真实worker加载张量证明。每条完整episode提交前再次核对权重，绑定证明摘要，经原生token、环境重放和SQL退款日历验证后原子保存。中断保留已提交episode与未完成证据，续跑只生成剩余episode；新session重新做质量与权重检查并计数。

总预算25,200秒，含准备、全部推理和审计，预留120秒打包。预计约4.5–6小时，实际取决于轨迹长度。累计预算覆盖续跑，超时、失败及正常完成均输出review ZIP。终端显示5个总阶段及每模型410个episode/质量动作进度，工具动作中显示turn与响应token。

使用独立配置 `configs/grpo-c4-confirmation-v2.json`、release manifest `grpo_c4_v2_release_manifest.json` 和新run_id。v1原始ZIP SHA256保留为a45cdf78b6215437b8b9799b4a36f5a5d7a8aa842a3b229eeef52ad9a4235f47。

## 代码阅读顺序

以下路径相对v2项目根目录：

1. `configs/grpo-c4-confirmation-v2.json`：12任务、7小时和验收门槛。
2. `src/apt_v2/grpo_confirmation.py`：任务生成、配对统计与验收。
3. `scripts/run_v2_grpo_c4.py`：调度、累计预算和返回包。
4. `scripts/grpo_c4_worker.py`：预检、单模型推理和逐episode续跑。
5. `scripts/grpo_c4_common.py`：输入绑定、原生重放和独立审计。
6. `scripts/build_v2_grpo_c4_release.py`：冻结发布包构建。
