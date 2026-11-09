# V2 项目交接：2026-11-04

用户要求：按原因逐项排查加权DPO退化；C1表现验收保持未通过，GRPO暂缓。每轮复核后独立冻结下一项。报告在对话简要列结果并提供链接，不自动打开；采用直接、基于结果的表述。

## E1 已完成

运行 dpo-regression-checkpoints-v1-20261103。8模型、169对偏好、238专家探针、384条轨迹，零新训练更新。86步随机完整任务13/16、greedy完整恢复21/28，258步0/16、6/28。训练stale chosen CE从父模型约0.004793升至最终1.065367，约222倍；局部损伤先出现，再随更新扩大。

本地重算评分，SQLite3.53.1严格重放160完整任务、224完整恢复，共2,692个生成动作；固定恢复前缀另行复原。报告reports/dpo-regression-checkpoints-v1-review-20261104.md，证据artifacts/reviews/dpo-regression-checkpoints-v1-20261103/。

## E2 已完成并通过补审计

运行 dpo-uniform-control-v1-20261104，均匀分支从chosen-only父模型独立训练258次，全部样本权重1。最终adapter SHA256 6f6de631322216e24a824f03d3ae4a90929accd2ecbc08624908275384dfe11a。

| 模型 | sample完整任务 | sample完整恢复 |
|---|---:|---:|
| parent | 8/16 | 46/84 |
| weighted SFT | 16/16 | 60/84 |
| weighted DPO | 1/16 | 15/84 |
| uniform DPO | 2/16 | 33/84 |

原审计因跨次reference最大token差1.31845e-4超过绝对门槛1e-4停止；本次fresh评分与本次训练缓存差为0。身份与对齐一致。新增联合数值校验保留绝对容差并加入相对容差，约束平均/序列/DPO logit误差；本次DPO loss影响上界2.09966e-5。

补审计32.83秒、exit0、error=null，audit.passed=true；208份原返回文件逐一一致，新增训练和生成0。原run_status.json保留第一次错误，新完成状态以audit_repair_status.json及audit.json为准。四模型80完整任务、448完整恢复、448首动作，共3,202个生成动作已严格重放，固定前缀另行复原。cloud真实Adam state再次审计通过。

原包证据artifacts/reviews/dpo-uniform-control-v1-20261104/；补审计证据artifacts/reviews/dpo-uniform-control-v1-20261104-audit-repaired/extracted/dpo-uniform-control-v1-20261104/。补审计ZIP SHA256 1392f4bce45f14b74d407ace78f4bdf854d59906116647aa13f9c8c44d9794b4。报告reports/dpo-uniform-control-v1-review-20261104.md。

均匀权重分支恢复损伤减轻，但整体仍退化。当前GPU记录L40S，前期登记L20；同硬件比较在下一轮补齐。候选均未晋升。

## 当前待云端执行：E3

运行dpo-chosen-ce-control-v1-20261104，入口scripts/run_v2_dpo_chosen_ce.py，配置configs/dpo-chosen-ce-control-v1.json。准备说明reports/dpo-chosen-ce-control-v1-preparation-20261104.md。

同NVIDIA L40S顺序训练weighted_control与dpo_ce：同父模型078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55，同169对，同reference字节，6遍/258更新/支，fresh AdamW、seed2026110301、LR5e-6、beta0.1、batch4、maxnorm1。困难原始权重4、全数据均值归一。

目标mean(w×DPO + λ×chosen CE)，CE按动作内token平均、样本间均匀平均。λ控制0、混合1，不做系数搜索；同次反向和step联合更新。总新增516个optimizer更新；混合支auxiliary_chosen_ce_updates=258单列，历史SFT255保持，累计DPO269。

Reference来自E2保存的父模型评分，SHA256 448b6c8af6fb18ae35fcad02f904167fea1f75d293fff0c53a70968c280b2a65。参考策略仍是chosen-only父模型。两个分支逐字节复用；每支首个更新前核验fresh父评分并记录。训练前严格比较GPU/库版本/精度设置；推理后校验硬件和运行库身份。

同轮自由生成weighted_sft、uniform_dpo、weighted_control、dpo_ce四模型，共528条轨迹。FP32评分另加父模型，共5×169对、5×238探针。配对主比较weighted_control→dpo_ce；uniform_dpo→weighted_control补权重对照。固定final258评测，中间43/86/129/172/215状态留存。

整个作业14,400秒，实时日志、资源快照、失败归档，保护已有run目录。回传dpo-chosen-ce-control-v1-20261104-review.zip。基础模型权重、中间adapter及Adam/RNG状态保留云端；review回传两个final adapter和完整元数据/评分/轨迹。云端执行绝对解释器路径/root/rivermind-data/envs/apt-vllm/bin/python，无需依赖终端激活状态。

本地全套163项测试通过，另1项真实非零CE更新通过，精度检查范围调整后对应测试复测通过；编译、CLI与真实输入预检通过。实际CUDA环境和完整base权重由云端prepare核对。完整回执handoff/dpo-chosen-ce-local-validation-20261104.json。

## 后续 E4 与项目范围

E3复核后，E4固定DPO部分、CE系数与CE动作曝光预算，将CE数据从原169对chosen换为train专家完整恢复续跑，补submit覆盖，另列token计算量。当前E3包只执行E3及其必要控制。

C1关闭后再接结果奖励GRPO与细粒度GRPO，二者原定共同起点仍是hard weighted SFT f8fb95…；当前保持暂缓。约10天收尾与原轻量SQLite依赖扩展范围沿用既有计划，避免无限调参。完整项目历史、代码地图、模型路径与奖励口径见handoff/PROJECT_HANDOFF_V2_2026-11-03.md；其旧状态条目以本文件为准。

本地v2为C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2，Python为其父目录.venv/Scripts/python.exe。云端v2为/root/rivermind-data/AgenticPostTrainingLab/v2。审阅按各包冻结source，保留原ZIP与失败记录；逐步核验模型、数据、tokenizer、状态、原生token与配对汇总。历史SQLite错误文本严格重放使用已安装3.53.1 DLL，不放宽错误字段比较。
