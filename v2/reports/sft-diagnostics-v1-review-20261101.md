# V2 SFT 执行诊断复核

日期：2026-11-01（Asia/Shanghai）。硬件：NVIDIA L20。输入：sft-diagnostics-v1-20261101-review.zip。

SFT 已提升协议和专家上下文中的动作预测；当前瓶颈集中在 SQL 自由生成。12 个已见训练任务成功 0/12，有效 SQL 为 0。固定输入下，HF BF16 与 vLLM 的 16 条 greedy 输出全部一致，模型分支开关均产生可观测影响。下一轮优先加强 SQL 构造、两个月份的区分，以及无有效指标后的补查修复。

## 审计错误与修复

GPU 评分、推理阶段均完成。最后 audit 报 probe set differs，原因是记录构造中的字典合并：

```python
{'probe_id': ..., 'mode': mode, 'score': score, **asdict(sample)}
```

Generation 自带 mode='greedy'，覆盖了外层用于区分 base/sft 的字段，16 条记录因此共享 8 个错误身份。两个分支的侧录文件因按 base/sft 过滤而为空。

代码已将模型分支 mode 和生成方式 sampling_mode 分开，并加入重复、缺失和未知 probe 身份检查。回归测试先复现失败，修复后 V2 全部 60 项测试通过。

本轮恢复依据是冻结 collector 的 base→sft 循环顺序，以及每个分支内固定的 8 个 probe ID 和 prompt token 顺序。原始 ZIP 和原始记录保留；恢复副本仅调整分支元数据，记录 branch_recovery，全部 token、概率和文本保持原值。随后使用冻结审计逻辑在本地完成重放和对照。GPU 作业保持原来的 audit 失败记录，本地补审计结果独立保存。

## 已见训练任务的完整执行

| 指标 | 原始模型 | SFT 后 |
|---|---:|---:|
| 任务成功 | 0/12 | 0/12 |
| 最终故障恢复 | 0/6 | 0/6 |
| 严格格式通过 | 78/144（54.2%） | 144/144（100%） |
| 工具错误 | 108 | 81 |
| 有效月度 SQL | 0 | 0 |
| 已验证指标事件 | 0 | 0 |
| 生成 token | 8,358 | 6,020 |

SFT 后错误包括：无效月度证据引用 75 次、预设临时查询故障 6 次。SQL 可执行但业务校验未通过的结果，另外体现在 verified_metrics=[] 中。工具错误计数与业务验证指标需要结合阅读。

## 正确专家上下文中的评分

对同一组 252 个训练动作进行 HF FP32 评分；正确回答的历史 token 均由示范提供。

| 指标 | 原始模型 | SFT 后 |
|---|---:|---:|
| token 平均 NLL | 0.09202 | 0.03743 |
| 动作平均 NLL | 0.15725 | 0.02102 |
| token 准确率 | 97.72% | 99.22% |
| 全 token argmax 匹配的动作 | 134/252 | 225/252 |
| SQL 动作全 token 匹配 | 30/72 | 45/72 |
| 其他动作全 token 匹配 | 104/180 | 180/180 |

SFT 后剩余 27 个不完全匹配的动作全部属于 query_sql。此处逐 token 预测以正确前缀为条件；完整任务评测则使用模型自身生成的历史。

## 首处分歧

固定 probe-02（train-000 首次月度查询）在三后端都走向同一个错误 JOIN 聚合 SQL。正确答案在 SELECT 后需要进入独立标量子查询，teacher-forced 首处分歧位于 response 第 16 个 token：正确 token 是“ (”，模型首选“ orders”。随后自由生成沿错误查询结构继续。

固定 probe-06（上月查询的 SQL 错误修复）已采用正确子查询结构，但将上月 2026-02 写成本月 2026-03，并仍命名为 prior_net_cents。三后端生成一致。

其余 6 个固定 probe——规则读取、schema、正确证据上下文中的计算、检查、提交及证据错误修复——SFT greedy 输出均与对应专家 response token 完全一致。

## 跨后端对照

| 对照 | base greedy 一致 | SFT greedy 一致 | 最大逐 token logprob 差 |
|---|---:|---:|---:|
| HF FP32 ↔ HF BF16 | 7/8 | 8/8 | base 0.1452；SFT 0.1279 |
| HF BF16 ↔ vLLM BF16 | 8/8 | 8/8 | base 0.3414；SFT 0.1187 |

adapter 开关使固定 probe 的自由输出发生变化：HF FP32 为 3/8，HF BF16 为 4/8，vLLM 为 4/8。评分也有变化。固定样本支持将本轮工作重点放在 SQL 学习；跨后端概率差异保留在原始审计数值中。

## 本地复核范围

- ZIP CRC 与 24 个冻结源码文件摘要通过。
- 10 个输入文件摘要与已保存的 SFT pilot 回传包一致；6 个 tokenizer/config 文件与模型身份摘要一致。
- 24 条训练任务轨迹逐轮重放，输入模板、token 解码、动作、观察、奖励、验证进展及终止状态全部匹配。
- 504 个 teacher-forced 评分记录重新汇总，与原评分摘要一致。
- 恢复 16 条 probe 分支标签，并完成 32 组跨后端配对比较。
- 本次诊断与本地复核 optimizer_updates=0。
- 原始 GPU 作业耗时 493.65 秒，原始阶段状态和失败日志保留。

ZIP SHA-256：1c307756a13a509f599e263301f060234b7741c4321d4fe3b3c39b45ce5acc61。

## 后续安排

进入 SQL 专项 SFT：先提高两个独立月度净额查询的动作掌握率，关注 SELECT 后的结构选择、paid 过滤、退款月份与 prior/current 参数绑定。以当前 12 个训练任务的自由执行成功和有效 SQL 作为第一层反馈，同时保留既有 development 对照。

训练数据应增加“SQL 有结果但 verified_metrics 为空”后的正确修复，以及缺少另一月份证据时的补查。训练预算、数据扩充与动作权重分别记录，按控制变量方式安排下一轮。

文件保存在项目 v2：reports/sft-diagnostics-v1-review-20261101.md；artifacts/reviews/sft-diagnostics-v1-20261101/；修复实现为 src/apt_v2/sft_diagnostics.py 和 diagnostics/diagnose_sft_v1.py。本地复核入口为 diagnostics/review_sft_diagnostics_v1.py。
