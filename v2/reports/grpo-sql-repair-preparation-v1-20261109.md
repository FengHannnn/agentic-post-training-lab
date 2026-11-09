# GRPO SQL repair preparation v1

新版SQL认证、96条冻结W诊断和六小时门控作业已实现。运行后按预检信号选择无提醒GRPO、提醒渐退对照，或少量SFT热启动后GRPO。正式基线保留W，云端作业待上传执行。

## 实现结果

- sql_semantics_v2显式绑定新任务，历史Task和旧回放保留legacy_v1。
- 认证包含金额扰动、强制跨月退款以及本月/上月/双月空退款；空gross列入CPU回归，按整数执行语义判断等价SQL。
- 8个train任务×K4×3条件，逐轨迹提交、绑定原生token/q与worker真实LoRA；中断后验证并复用已提交轨迹。
- 在线最多36组/支，维护与目标各半；提醒路线按100%→50%→0%渐退，最终评测均无提醒。
- SFT固定24动作、24步、96例次，逐步保存adapter和Adam/RNG；B来源为SFT537/DPO11，C以B为冻结reference并新建Adam。
- 质量拒绝保存为结果，完整性错误保留失败包；最终选优独立重算信用、缓存目标、实际权重变化、B门槛和C增量。
- 预算累计包含实际打包，外部receipt记录归档哈希及总时间；resume继续累计该回执中的时间。

完整方法与阈值见 reports/grpo-sql-repair-contract-v1-20261109.md。冻结任务和监督动作位于 artifacts/inputs/grpo-sql-repair-20261109/。

## 本地验证

回归覆盖512项：历史478项通过，新实验34项在冻结源码后独立复跑通过。首次全量进程载入早期测试夹具产生的4项失败均已解决，日志与重跑记录保留。

最终验证结果写入 reports/grpo-sql-repair-validation-v1-20261109.json，测试日志保存在 .local/grpo-sql-repair-20261109/。发布包干净解压后再次通过34项测试；290项文件哈希、291个ZIP条目、CRC、源码编译及5个命令入口全部通过。最终包沿用该轮测试的源码、配置和冻结输入，更新验证记录。

已完成原生token验证：176个专家起点在原预算内成功；最大提示2057tokens，最大响应142tokens，上限8192/256。24个监督动作及掩码通过真实执行和EOS检查。

本地LoRA/Adam验证使用微型Qwen2模型；云端作业记录本轮1.5B模型的采样、更新、性能和GPU身份。

## 云端运行

将 grpo-sql-repair-study-v1-20261109-run.zip 上传到云端包含v2目录的项目父目录，在该目录执行：

~~~bash
unzip -o grpo-sql-repair-study-v1-20261109-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_repair.py --config configs/grpo-sql-repair-study-v1.json
~~~

对应作业中断后，在同一v2目录执行：

~~~bash
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_repair.py --config configs/grpo-sql-repair-study-v1.json --resume
~~~

已完成的作业保留结果；resume核验同一发布、配置、GPU UUID和累计预算。输出终端Download指向的review ZIP及Receipt JSON。二者一起返回，用于核验最终打包后的累计时间。state.pt留在云端。

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
