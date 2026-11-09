# GRPO alias curriculum training contract v1

本轮在 L40S 上比较两支固定 local 信用的 GRPO，检验“先补齐单月缺别名，再恢复双月缺别名”的课程能否提高原复合起点的恢复成功率。整个作业累计最多 6 小时，每支最多 48 组，K=4。

## 实验依据

上一轮冻结 W、仅采样的困难恢复预检耗时 24 分 15 秒。缺别名简化起点成功 11/16、出现新增认证指标 14/16、4/4 组存在 local 信号；原复合起点成功 0/16、进度 0/16。因此，本轮优先检验缺别名课程。

## 比较模型与固定设置

| 模型 | 初始化与目标组起点 | 信用、学习率 |
| --- | --- | --- |
| W | 最佳加权 SFT/DPO 基线，SFT 513、DPO 11、RL 0 | 冻结评测 |
| control | 从 W 独立初始化，所有目标组均为原复合起点 | local，1e-6 |
| curriculum | 从 W 独立初始化，前半目标组为简化起点，后半转回原复合起点 | local，1e-6 |

W adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。

基础模型 Qwen2.5-1.5B-Instruct，revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306。两支各自新建 AdamW、参考模型固定为 W；阶段切换连续保留策略、Adam 和 RNG 状态。继续训练 q/v LoRA，额外 SFT/DPO 更新为 0。

local 仍为“组归一化 outcome advantage＋0.2 × 当前动作新增认证指标数”；前缀已有的上月指标排除在新增奖励之外。K=4、temperature=1、beta=0.02、clip=0.2、TIS cap=2、梯度裁剪 1、采样 K3 上限 0.05、clip fraction 上限 0.5 保持固定。总步数 12 包含注入错误的前缀。

两支共用任务、rollout seed、组序和维护组，只改变目标组起点课程。12 个训练任务为原任务 6 个＋此前课程任务 6 个，覆盖有/无 transient_query；两阶段使用相同 12 个任务。新开发集使用 seed 81000–81003，排除历史开发、测试和 C4 确认实例。

## 固定训练日程

| 提交组数（每支，含跳过事务） | 维护组 | 目标组：control | 目标组：curriculum |
| --- | --- | --- | --- |
| 1–24 | 12 组 | 原复合 12 组 | 简化 12 组 |
| 25–48 | 12 组 | 原复合 12 组 | 原复合 12 组 |

每两个组依次为维护、目标。每支满额 24 个维护组，其中完整任务 12 组、原 7 类标准恢复共 12 组；第二阶段交换各任务在维护组中的完整/恢复用途，使每个训练任务各覆盖一次完整和一次标准恢复。每支目标组共 24 组。满额共 384 条训练轨迹，最终再各采 4 条冻结策略刷新轨迹。

简化起点只为上月 SQL 恢复 AS prior_net_cents；本月仍缺别名，后续错误计算保留。与原复合起点的前缀长度、剩余步数相同。原复合起点保留双月缺别名、未认证查询和无效证据计算。

两支完成第 24 个组事务后，分别固定该策略进行 16 条诊断：4 开发任务 × 2 缺别名起点 × 2 次采样。两支诊断均完成后进入第二阶段。课程切换时点固定，中点不用于选模型、调参或改变课程。

全零信用组提交 skip 事务，保留原策略，不拒绝采样、不补组。actual_updates 单独统计；训练剂量按采样并提交的组数对齐。整支全零时，审计要求所有训练组均无有效信用、所有事务均合法跳过、最终权重严格保持 W，并完成最终策略采样证明。

## 评测与判断

W 与两支最终策略各评测 132 条标准轨迹：完整 greedy 4、完整 sample 16、标准恢复 greedy 28、标准恢复 sample 84；另外各做 7×2 次 greedy 一致性生成。

每个模型再做 32 条缺别名定向评测：4 开发任务 × 2 起点 ×（greedy 1＋sample 3）。三模型共 492 条最终评测，加两支中点 32 条，共 524 条评测轨迹。

课程机制主要观察原复合起点采样成功数：curriculum 对 control、对 W 各自的增减，并列出成对 improved/regressed、简化起点表现和新增认证指标。中点与最终结果帮助观察简化能力向原复合起点的迁移。

项目候选仍按标准任务选择：W 完整采样未饱和时以完整成功数为主，饱和后以标准恢复成功数为主；两项采样成功数和两项 greedy 成功数均要求不低于 W，主指标需有正增益。同分按标准恢复成功数，再优先 control。定向诊断与中点不参与项目候选选择。开发集候选在后续新任务确认中复核。

## 预算与续跑

6 小时为整个匹配作业累计预算，含准备、W 评测、两支训练、中点、最终评测、刷新、审计与打包。参考上一轮 fine 作业的训练和评测时间，本轮预计约 5.5–6 小时；48 组为上限，实际规模由测得的耗时和剩余预算决定。

每完成一对“维护＋目标”后，在两支 cursor 齐平且没有待提交采样组时，按近期组耗时预估下一周期，并预留两次最终评测、刷新、尚未完成的中点、审计及打包时间。可以在相同偶数组数提前停止，并报告每阶段维护/目标的实际暴露量。若在第 24 组或之前停止，第二阶段目标暴露记录为 0，迁移阶段尚待运行。

任一阶段报错，自动打包返回已提交轨迹、权重、日志、配置和源码；云端保留 Adam/RNG 的 state.pt，返回包不包含该文件。使用 --resume 续跑相同作业，已提交组只校验，累计时间接续计算。首轮可换另一张 L40S；同一作业及续跑固定 GPU UUID、软件和驱动。

终端显示总阶段、采样 episode/action、训练 scoring/backward、评测和回放进度。报告仅提供链接，不自动打开。

## 本地验证

本地全量 GRPO 回归 260 项通过，新增总审计汇总测试 1 项通过；实际 ZIP 干净解压后的 28 项新增测试全部通过，276 个绑定文件、全部源码语法和启动入口核验通过。验证结果随运行包写入 reports/grpo-alias-curriculum-validation-v1-20261109.json。真实 Qwen 分词器核验覆盖 160 个任务起点：最大前缀 1,345 token、专家路径最大输入 1,968 token、最大回复 142 token；均满足输入 8,192、单次回复 256 的预算。全部专家路径均在总步数 12 内成功。

## 运行方式

将 grpo-alias-curriculum-v1-20261109-run.zip 上传到 /root/rivermind-data/AgenticPostTrainingLab，再执行以下命令。包内顶层 v2/ 合并到现有 v2/，包含本轮脚本、配置、W 和 profile 输入；基础模型沿用现有 .cache。

    cd /root/rivermind-data/AgenticPostTrainingLab
    unzip -o grpo-alias-curriculum-v1-20261109-run.zip -d .
    cd v2
    /root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_alias_curriculum.py --config configs/grpo-alias-curriculum-v1.json

中断后续跑：

    /root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_alias_curriculum.py --config configs/grpo-alias-curriculum-v1.json --resume

首轮返回包路径：/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-alias-curriculum-v1-20261109-review.zip；续跑使用 attempt-002 等后缀。请返回终端 Download 指向的文件。

## 代码阅读（仅本轮新增或改动）

| 文件相对路径 | 大致作用 |
| --- | --- |
| src/apt_v2/grpo_alias_curriculum.py | 定义两支课程、配对上下文、任务隔离、评测计划和阶段暴露量。 |
| src/apt_v2/grpo_curriculum.py | 将本轮任务、上下文、冻结输入与定向评测分派到新课程模块。 |
| scripts/grpo_common.py | 接入本轮配置合同和零更新结果的合同边界。 |
| scripts/grpo_worker.py | 为新课程启用训练时同次采样前向概率捕获。 |
| scripts/grpo_vllm.py | 为新课程使用原生采样概率证据。 |
| scripts/grpo_alias_common.py | 核验两支匹配配置及六小时的成对预算预留。 |
| scripts/run_v2_grpo_alias_curriculum.py | 执行准备、训练、中点、最终评测、刷新、审计和续跑。 |
| scripts/probe_v2_grpo_alias.py | 在提交第 24 组的策略上执行配对中点诊断。 |
| scripts/audit_v2_grpo.py | 核验全零信用且权重保持 W 的合法跳过证据，检查原生概率证明。 |
| scripts/audit_v2_grpo_alias.py | 回放训练与评测，重算损失，汇总课程迁移和标准能力。 |
| scripts/build_v2_grpo_alias_release.py | 打包运行文件并核验 ZIP、源码与输入哈希。 |
| configs/grpo-alias-curriculum-v1.json | 固定总预算、两支配置和诊断规则。 |
| configs/grpo-alias-control-v1.json | 固定对照支原复合起点训练。 |
| configs/grpo-alias-curriculum-arm-v1.json | 固定课程支简化到复合的阶段安排。 |
| tests/test_grpo_alias_curriculum_v2.py | 验证配对任务、上下文、信用、预算和零信号审计。 |
| tests/test_grpo_alias_driver_v2.py | 验证完整作业、阶段顺序、原生诊断和失败续跑。 |

云端报告路径：v2/reports/grpo-alias-curriculum-contract-v1-20261109.md。
