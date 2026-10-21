# Phase 5：144×8 训练信号批次实际结果与验收

日期：2026-10-21。Phase 5 数据与学习信号准备完成；真实模型反向、优化器更新、reference/KL 和行为对照仍未执行。training_admitted=false。

## 产物身份与复核

用户回传 phase5-rollout-review-20261021.zip，SHA256=c008077b529a70127268b449a9f8cc28950c20acd7aa1424255df8b45c12b071。共 32 个文件、解压约 8.91 MB，归档于 artifacts/trajectories/phase5-signal-20261021-01。

本地验证 7 份产物哈希、23 份源码快照哈希及与当前可信源码的一致性，并在确认路径不越界后解包。复用真实冻结 tokenizer 和审计实现，重新核对全部 1152 条 token、模型/输入身份、解码、seed、严格评分、8 条分组 advantage、old/new 数值统计以及打包分数/mask。weights_only=True 读取张量，完整重算 summary 与云端结果一致。

本地/云端依赖版本不同，摘要统计浮点比较用相对 1e-6、绝对 1e-8 容差；没有修改原始概率门或任何原分数。本地只读取已保存概率，未加载模型、重跑 GPU 或独立复现云端运行时梯度图。冻结基线 74 份快照与 8 份 cache 验证通过，不伪称在 Windows 重现原 Linux cloud bundle 路径清单。local_audit.json 位于 artifacts/evaluations/phase5-signal-review。

## 实际结果

| 项目 | 结果 |
| --- | ---: |
| GPU | NVIDIA L20 |
| 训练输入 / 回答 | 144 / 1152 |
| EOS / length | 1152 / 0 |
| 回答 token | 32296 |
| 严格成功 | 572/1152（49.65%） |
| 语义成功（仅诊断） | 731/1152（63.45%） |
| 非法 JSON | 46 |
| 全对 / 全错 / 混合组 | 31 / 51 / 62 |
| 零方差率 | 56.94% |
| 非零 advantage 回答数 | 496 |
| old/new logp 最大差 / ratio 最大偏离 | 均为 0 |
| sampler/trainer 最大逐 token logp 差 | 0.1388685703 |
| token TIS 权重范围 | 0.9085090004–1.1489730809 |
| 权重上限触发比例 | 0 |
| 校正策略 loss 诊断值 | -0.0008627223 |

旧跨引擎门继续为 false。训练侧内部同路径检查通过，参数摘要前后保持原始 SFT 身份，云端记录 model_backward_calls=0、optimizer_steps=0。负 loss 只是已保存分数上的策略目标诊断，不表示完成训练或改进。token TIS 仍是有偏条件概率近似，不声称已纠正完整序列或组联合分布。

## 增加到八条的作用

对同一批 144 组，只看各组前 4 条有 47 个混合组；看完整 8 条有 62 个。额外 4 条揭示了 15 个混合组，即混合组绝对数量增加约 31.9%。该比较不额外采样、不修改 reward，说明这批数据里扩大组内采样有实际价值。

此前 48×4 只有 10 个混合组、40 条非零 advantage 回答；本批为 62 组、496 条回答。这里同时改变输入覆盖、组大小与 seed，不能将全部差异单独归因于组大小，也不能将严格成功率从约 50.52% 到 49.65% 解释成训练退步，因为根本没有更新，评估输入也不同。

## 分问法与失败类型

| 问法 | 严格成功 | 全对组 | 全错组 | 混合组 |
| --- | ---: | ---: | ---: | ---: |
| natural | 108/384（28.13%） | 2 | 25 | 21 |
| sequence | 159/384（41.41%） | 7 | 24 | 17 |
| symbolic | 305/384（79.43%） | 22 | 2 | 24 |

580 条严格失败中，wrong_arguments 507、invalid_json 46、invalid_expression 12、invalid_schema 15。wrong_arguments 包括表达式不符合严格目标的情形，不能直接等同于全部算错。语义成功比严格成功多 159 条，提示部分表达式结果可对但严格结构不符；不因此放宽冻结 reward。

自然语言与顺序问法仍是主要行为弱项；已有 62 个混合组足以支持一个有边界的小规模真实更新实验设计，不意味着已经具备稳定大规模 RL 收益。51 个全错组不能从当前组相对策略项获得信号，但不应为了报表好看在原始数据中删除。未来可显式采用有效组调度并披露选择规则，不将其冒充原始分布指标。

## 时间与费用

生成 277.20 秒，约 116.51 response token/s；全部两遍 FP32 评分 53.42 秒；作业总时长 358.47 秒，约 5 分 58 秒。按记录时价 2.18 元/时估算作业费用约 0.2171 元，算术已复核。不是包含开机、空闲、存储和取整的完整平台账单。三个阶段退出码均为 0，无重试，无自动实例关机。

## 完成状态与下一步

本批数据准备验收通过；原始记录、概率、严格 reward、mask、分组 advantage 和新契约均已核对。无需为了继续追求更高混合比例再盲目扩大一轮。下一步应冻结有限真实 GRPO 更新方案，明确旧策略与 reference 的角色、校正与 KL、有效组调度、梯度/clip/长度监控、保存恢复和独立行为对照。

不能把这批固定 SFT 回答无限作为后续更新策略的最新 on-policy 数据；真实循环需要按策略版本生成并刷新 rollout。首次更新前先验证真实模型 LoRA 梯度与基础参数冻结、损失尺度和参考策略身份，设置有限步数和停止规则。该实现与云端更新尚未开始，不在本次记录中伪称已完成。

本轮未改运行源码，未重复已经通过的 289 项测试；只执行完整产物重审、统计与文档归档。

## 代码阅读顺序

1. configs/phase5_rollout_signal.json：144 个固定训练输入、每题 8 条、新 seed 与分块。
2. src/agentic_post_training/rollouts/phase5_signal.py → signal_summary：前4/全8对照、混合组与非零 advantage 回答数的定义。
3. scripts/collect_phase5_signal.py → generate/score：原生 token/q、固定 FP32 old/new、无更新的边界。
4. 同入口 audit：原生数据、奖励与 advantage、打包 mask、逐 token 分数及 summary 的一致性。
5. src/agentic_post_training/algorithms/rollout_correction.py → corrected_grpo_policy_loss：为什么当前非零 loss 不代表模型已训练。
6. tests/unit/test_phase5_signal_audit.py：合成完整1152条产物与 mask 篡改拒收；结合本报告区分测试构造数据和实际云端数据。
