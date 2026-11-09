# C4 independent confirmation contract

日期：2026-11-07。运行：`grpo-c4-confirmation-v1-20261107`。用户已确认实用门槛与本轮安排；单张 L40S，最多 4 小时，新增训练更新为 0。

## 冻结候选

| 标签 | 候选 | 新增 RL 更新 | adapter SHA256 |
|---|---|---:|---|
| weighted_sft | 最佳困难恢复加权 SFT，W | 0 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| outcome_rl16 | C2 结果奖励 GRPO 最终候选 | 16 | 6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6 |
| fine_rl17 | C3 细粒度 GRPO 最终候选 | 17 | 1990f06d2eb8e3a1777273051bf2bce078c8bff67206533b88dd65780e5959c6 |

三者均继承 SFT513/DPO11，底座 Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。本轮预先选定三者，不使用 C4 结果调整权重、checkpoint、任务或生成参数。

## 独立任务与评测规模

8 个全新实例与任务族标识，seed91000–91007，family_id 为 c4-family-91000 至 c4-family-91007；与原 train/development/test 及 closing-development 的 id、family_id、seed 隔离。保留既有月度对账业务、SQLite 结构和金额生成规则，采用新的 2027 年账期及区域/segment 组合，含 2027-01/2026-12 跨年案例。四个实例注入首次查询临时故障。

这里的新任务族指新 family 标识和数据实例，业务模板沿用原月度对账。C3 结转扩展结果保留在独立报告；本轮确认既有 GRPO 候选在全新月度实例上的效果。

每模型：完整 greedy8 + sample32；7 类完整恢复 greedy56 + sample168，共264条episode。三者合计792条episode。每模型7类恢复上下文各做两次greedy首动作质量检查，正常共42次质量动作。

工具预算保持12步，错误注入前缀消耗总预算；恢复运行至真正成功或预算终止。64个完整/恢复起点先以脚本专家验证相同预算可完成。sample temperature0.8、top_p1、top_k-1，响应上限256，模型上下文8192；同任务、场景、mode、rollout使用相同种子。确认seed2026110700，逐动作seed为episode seed加动作序号。

## 预先固定的实用验收

每个 GRPO 候选分别与同机 W 配对。候选达标需同时满足：

1. sample 恢复成功至少比 W 多4/168，即 +2.38 个百分点。
2. sample 完整、greedy 完整、greedy 恢复成功数均不低于 W。
3. 沿用先前的主指标顺序：若 W sample 完整为32/32，则主指标为恢复；否则主指标为完整，完整成功数还需至少增加1。

两支有任意一支达标，即本轮“至少一个 GRPO 超过 W”实用确认通过。作业成功运行、审计通过和模型是否达到效果门槛分别记录。若均未达标，保留该结果用于下一轮实验设计。若后续据此调参，本批任务转为诊断数据，后续最终确认另冻新任务。

同时报告配对改善/退步、任务级与场景级成功数、动作和token开销。以8个任务族为聚类单位进行10,000次配对bootstrap，seed2026110711，95%区间作为描述统计；实用验收使用上述固定计数。保留两项候选比较全部结果，不按本轮表现删掉未达标候选。

## 运行、重放与预算

三模型分别独立进程，只加载一个LoRA、编号1，max_loras=1，禁用prefix caching，沿用vLLM0.11.0 BF16推理与原生processed logprob。准备阶段核对完整base文件、三份adapter及源码哈希，采集本次L40S UUID、驱动和软件身份；新作业可使用另一张L40S，作业内和续跑保持同一身份。

每个推理session先完成重复greedy质量检查和真实worker加载112个LoRA张量证明，再采集episode；session结束再次核对张量。每条完整episode提交前再次验证worker加载张量，并绑定该证明摘要；经原生token、环境和独立SQL退款日历校验后原子提交。中断保留已提交episode和未完成轨迹证据，续跑验证后只生成剩余episode；新增session重新执行质量和加载证明，额外质量动作单独计数。

总wall-time14,400秒，含准备、三模型评测、审计，预留120秒打包。续跑累计已耗时间；异常强制中断后，活动阶段的未记录时间也计入预算。终端显示总阶段进度和每模型278个episode/质量动作进度，并在每次工具动作显示当前turn与token数。超时或失败生成review ZIP；完整成功时归档冻结源码、配置、权重、逐条轨迹、质量检查、审计、配对案例和CSV汇总。

## 代码阅读顺序

以下路径相对v2根目录：

1. `configs/grpo-c4-confirmation-v1.json`：候选、规模、预算和验收约束。
2. `src/apt_v2/grpo_confirmation.py`：新任务、配对统计和验收判断。
3. `scripts/run_v2_grpo_c4.py`：作业调度、累计预算和失败打包。
4. `scripts/grpo_c4_worker.py`：身份预检、单模型推理与逐episode恢复。
5. `scripts/grpo_c4_common.py`：原生轨迹、权重证明及独立审计。
6. `scripts/build_v2_grpo_c4_release.py`：发布冻结源码与三份候选权重。
