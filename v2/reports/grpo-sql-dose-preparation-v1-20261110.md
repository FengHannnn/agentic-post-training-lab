# Target-dose warm-up study v1 实现与运行说明

已实现目标重复4/8/16次的预登记搜索。固定学习率2e-6及48条维护动作，候选从W独立训练，先做train首动作筛选，再执行完整B验收，保存首个通过者。新增GPU作业上限6小时。

方法与阈值见 reports/grpo-sql-dose-contract-v1-20261110.md，上一轮诊断见 reports/grpo-warmup-probe-results-v1-20261110.md。冻结数据和顺序位于 artifacts/inputs/grpo-sql-dose-20261110/。

## 实现与验证

- 60条原始动作内容不变；三档总例次为96/144/240、更新24/36/60步，维护各48例次。
- 新来源计数与source_config同时校验，区分r4与历史B24；每个候选的新Adam、实际更新、保存和恢复沿用已验证事务。
- train筛选要求两次认证均至少9/12并高于W，维护及重复一致性不低于W，有效BF16投影变化；完整B门槛沿用原协议。
- 固定r4→r8→r16顺序，首个完整通过后收尾；每个候选的未通过结果、实际成本和来源分别保留。
- root audit重新计算数据顺序、SFT检查点、探针、完整门槛与选择；ready-B.json仅标记下一阶段warm-up起点，正式基线W保留。

本地数据执行与原生token验证通过：最长提示1358tokens、响应142tokens，均在8192/256预算内。全部26个GRPO模块367项回归通过；独立解压后93项定向复测通过，三档候选导出25项检查通过。四个新CLI入口、361个安全唯一条目与360项文件哈希核验通过。最终包保留干净解压测试的代码、配置和冻结输入，只补充验证记录。本地真实LoRA/Adam更新与恢复由微型Qwen回归覆盖，云端作业记录1.5B模型的真实效果。验证记录写入 reports/grpo-sql-dose-validation-v1-20261110.json，日志位于 .local/sql-dose-20261110/。

## 云端执行

上传 grpo-sql-dose-study-v1-20261110-run.zip 至云端项目父目录，在该目录执行：

~~~bash
(
set -e
unzip -o grpo-sql-dose-study-v1-20261110-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_dose.py --config configs/grpo-sql-dose-study-v1.json
)
~~~

中断后在同一v2目录给原命令增加--resume。已完成作业保留结果。返回终端Download指向的review ZIP和Receipt JSON；Adam/RNG state.pt继续保留云端。

本轮新增最多21600秒，与此前9927.179秒分开登记，累计最大31527.179秒。作业完成或达到预设停止条件即收尾。本轮取得合格B后进入下一阶段信用设计，RL更新为0。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo_dose_study.py | 剂量及选择规则 |
| grpo_repair_study.py | 剂量分支来源计数 |
| grpo_common.py | 新B配方与来源验证 |
| grpo_repair_worker.py | SFT重复数据与恢复 |
| grpo_warmup_probe_worker.py | 可复用原生诊断接口 |
| grpo_dose_worker.py | warm-up、探针和独立审计 |
| run_v2_grpo_repair.py | 预算超限撤销选择 |
| run_v2_grpo_dose.py | 有序搜索及新增预算调度 |
| validate_v2_grpo_dose.py | 数据、token和频数验证 |
| build_v2_grpo_dose_release.py | 云端包核验 |
| grpo-sql-dose-study-v1.json | 主配置 |
| grpo-sql-dose-branch-v1.json | 评测分支配置 |
| test_grpo_dose_v2.py | 配方和事务测试 |
