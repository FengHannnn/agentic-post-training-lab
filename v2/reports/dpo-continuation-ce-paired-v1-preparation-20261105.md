# E4 GPU UUID 变化诊断与同机成对修复

2026-11-05，Asia/Shanghai。

故障已定位：E3 与当前运行环境只有 gpu_uuid 不同。原 E4 在 prepare 阶段把跨作业 UUID 也作为精确相等条件，因此停止；本次优化器更新为 0。新作业在当前 GPU 上顺序新训练 chosen CE 控制与专家续跑 CE 两支，形成同一作业内的主比较。已完成本地验证，云端结果等待执行。

## 原因与复现

| 字段 | E3 | 当前 |
|---|---|---|
| gpu_uuid | 086e2842-344c-9528-cbae-b8fcba665e1c | 92e14e85-628d-e5a4-af16-0db4defb2afb |
| GPU 名称、显存、计算能力 | NVIDIA L40S / 原记录 | 与 E3 一致 |
| Python、torch、CUDA、cuDNN、transformers、peft、vLLM | 原记录 | 与 E3 一致 |
| matmul/TF32 精度设置 | 原记录 | 与 E3 一致 |

依据为用户执行完整 runtime_identity 字段比较后的输出。用该差异重放原 check_runtime，稳定得到 ValueError: paired runtime identity differs；最小变化字段仅为 gpu_uuid。已有完整差异证据和确定性复现，直接修正跨作业与作业内的比较策略。

原实现将两个范围合在一次比较：沿用 E3 冻结模型做主控制时要求同一物理 GPU；而云平台当前提供了另一个 GPU UUID。我在原 E4 包中同时固定 UUID 且报错未列明字段，造成此次预检中止与定位往返。此次修复补齐运行环境记录与具体字段差异。

## 新作业与实验匹配

新运行 ID：dpo-continuation-ce-paired-v1-20261105。入口 scripts/run_v2_dpo_continuation_ce_paired.py。

| 分支 | 初始化 | CE 数据 | 新更新 | CE 动作曝光 |
|---|---|---|---:|---:|
| chosen_control | 原 chosen-only 父模型 | 原169个chosen，沿用E3 CE内核 | 258 | 1014 |
| continuation_ce | 同一父模型 | train专家完整恢复续跑288动作 | 258 | 1014 |

两支都 fresh AdamW、6遍、batch4、LR5e-6、beta0.1、hard4权重、grad clip1、DPO seed2026110301、CE系数1，精确复用父模型reference缓存448b6c…；每支累计SFT255/DPO269，辅助CE联合更新258单列。总计516次新更新，整项作业时限仍为14400秒。

DPO部分和CE动作曝光保持相同；主比较为同机新训练 chosen_control→continuation_ce。CE续跑池保持原E4固定方案：train12任务×7上下文，288动作，固定CE seed2026110504，抽样1014次，每动作3–4次。查询167、计算256、校验296、提交295次。test保留。

推理评测4个模型：加权SFT、历史E3 chosen CE、新chosen_control、新continuation_ce。共80条完整任务、448条完整恢复；再对包含parent的5模型做169对偏好与238个固定探针评分。历史E3 CE用于观察跨作业差异；主要CE覆盖效应来自本次两支的配对比较。

| 每支计算预算 | chosen_control | continuation_ce |
|---|---:|---:|
| DPO偏好曝光 | 1014 | 1014 |
| DPO输入token | 1941912 | 1941912 |
| CE动作曝光 | 1014 | 1014 |
| CE响应token | 105486 | 85203 |
| CE新增forward | 0 | 1014 |
| CE新增forward输入token | 0 | 1202952 |

控制分支复用 DPO chosen forward 计算 CE，续跑分支独立 forward。两种CE都按动作内部response token均值（含EOS）、动作之间均匀平均。主指标为完整任务成功、核心辅助为完整恢复成功，并记录有效首动作后的成功转换、分场景和计算/校验/提交概率。

## 运行环境检查

跨作业允许有效 GPU UUID 变化，其余型号、计算能力、显存、软件版本与训练精度继续匹配 E3。该策略所在固定配置要求两支 fresh 控制同时存在。

prepare 先保存 execution_environment.json 和 runtime_precheck.json，再执行身份检查；错误会附具体 expected/actual 字段。通过后，以本作业实际 runtime 作为基准，两次训练开始前都严格核对相同 UUID、软件和精度。推理后检查仅按原协议略过引擎可改变的三项精度全局开关，硬件/软件身份保持核对；训练前精度严格匹配。

旧 dpo-continuation-ce-control-v1-20261105 失败目录和review ZIP保留，新运行名独立。新包不会调用旧失败输出。根目录历史源码文件与原发布包保留，新脚本、配置和发布清单独立命名。

## 验证

原错误复现成功；对应回归测试先在原精确UUID比较逻辑上失败，再在新策略上通过。11项新主验证通过，另3项检查通过（1项强制成对控制契约与2项既有真实LoRA内核回归），覆盖12个新增测试与2个内核回归。

覆盖仅UUID变化可启动新匹配作业、作业内UUID变化拒绝、软件/硬件/训练精度变化拒绝、错误字段与实际环境先行保存、原生完整轨迹与五模型评分篡改检测、第一支完成而第二支失败时准确保留更新计数和归档。源码编译、5个CLI入口通过。

本地使用已审3份真实冻结adapter和tokenizer核验169对reference绑定、288个CE动作、固定1014抽样、238探针、27份provenance和两支token预算。输入预检注入用户给出的当前runtime差异；实际GPU与完整base权重检查由云端在更新前执行。回执见handoff/dpo-continuation-ce-paired-local-validation-20261105.json。

E3已审结果保持：chosen CE随机完整任务10/16、恢复37/84，对加权SFT16/16、59/84。C1表现验收未通过，GRPO暂缓。

## 云端执行

上传 dpo-continuation-ce-paired-v1-20261105-run.zip 至云端 v2 后：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-continuation-ce-paired-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_continuation_ce_paired.py
~~~

回传 dpo-continuation-ce-paired-v1-20261105-review.zip。保留E2已修复审计与E3完整artifacts供输入来源验证。失败或超时仍打包日志；两份新最终adapter随返回包，中间权重和state.pt保留云端。
