# SFT 独立 test 结果与共同基线冻结

日期：2026-11-02（Asia/Shanghai）。硬件：NVIDIA L20。运行：sft-independent-test-v1-20261102。

126-step SQL 加权 SFT 候选在独立 test 上成功 4/4、故障恢复 2/2，预设验收项全部通过。本地复核完成后，已将其保存为共同 SFT 冻结基线 sft-sql-weighted-v1-20261102，供后续 DPO 与 GRPO 从同一权重出发。

## 本轮独立 test 结果

| 指标 | 63-step 父模型 | 126-step SFT |
|---|---:|---:|
| 任务成功 | 0/4 | **4/4** |
| 故障恢复 | 0/2 | **2/2** |
| 有效月度 SQL | 0 | **8** |
| 已验证指标事件 | 0 | **12** |
| 有效检查 / 提交 | 0 / 0 | **4 / 4** |
| 严格动作格式 | 48/48 | **30/30** |
| 工具错误 | 36 | **2** |
| 总动作轮数 | 48 | **30** |
| 生成 token | 1,750 | **2,344** |

SFT 的 2 次工具错误均为预设的临时查询故障，后续重试并成功提交。普通任务 7 轮、故障任务 8 轮，所有任务均完成读规则、查 schema、两个月度查询、差额计算、检查与提交。

| 任务 | 本月 cents | 上月 cents | 变化 cents | 轮数 | 最终恢复 |
|---|---:|---:|---:|---:|---|
| test-000 | 20142 | 28669 | -8527 | 7 | 无预设故障 |
| test-001 | 17487 | 17040 | 447 | 8 | 成功 |
| test-002 | 15966 | 16783 | -817 | 7 | 无预设故障 |
| test-003 | 29673 | 21522 | 8151 | 8 | 成功 |

本轮只做推理，optimizer_updates=0，总耗时 171.24 秒。prepare、evaluate、audit 三阶段均成功。

## 当前 SFT 固定划分表现

同一 adapter 身份对应的最近评测为：train 12/12、development 4/4、test 4/4；故障恢复为 6/6、2/2、2/2。

范围为当前合成业务对账任务族的固定 12/4/4 划分、greedy 单次运行。独立 test 的首次模型评测现已完成，后续比较保留该曝光记录；更大规模泛化评测采用扩充后冻结的划分。

## 本地复核

- ZIP CRC 通过，33 个冻结源码/配置文件摘要一致，且均匹配本地发布版本。
- 63-step 父 adapter 与已归档首轮包一致；126-step 候选 adapter 及 complete 身份匹配 SQL 加权回传包。
- 候选训练数据摘要匹配；test 任务 ID、family 与训练样本隔离，4 题任务清单与冻结生成器一致。
- 6 个 tokenizer/config 文件匹配模型身份摘要。
- 本地重放全部 8 条轨迹、78 个动作，prompt、token 解码、工具观察、奖励、终止与恢复状态一致。
- 本地重算 test 汇总、验收和审计与云端结果完全一致；78 次生成均以 EOS 结束。

原始 ZIP SHA-256：e81ac810633028cebfaf989611ec535c4af04f5d268e8225fb732a7ebeeba4a3。

## 冻结基线

本地目录：artifacts/baselines/sft-sql-weighted-v1-20261102/。

adapter SHA-256：e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0。

baseline_manifest.json SHA-256：225f430fa263b72b0a9e0ee39613504cc3714ba98b0b87fd99934a2d52c0d318。

冻结副本包含 adapter、Adam 状态、原 checkpoint 身份、模型与训练配置、tokenizer、训练统计和 test 复核证据，共 20 个文件逐项摘要核验。原候选、训练运行和所有回传包继续保留。candidate_identity.json 已更新为 promoted_to_frozen_sft_baseline。

后续 DPO、GRPO 在独立运行目录加载该权重，reference 使用固定 SFT 权重并冻结梯度。保存的 Adam 用于复现及明确约定的续训；各新算法的优化器起点在对应实验配置中单独确定。

云端现有权重位置仍为 /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/sql-sft-weighted-v1-20261101/final/。本次独立冻结副本已建立在本地项目，后续云端训练包会按同一摘要绑定起点。

## 接续

SFT 基础执行阶段完成。下一阶段可从冻结 SFT 采集随机训练轨迹，检查成功/失败分布，再构建有效 DPO 偏好对；GRPO 也从同一 SFT 权重独立出发。当前固定任务已全部成功，后续数据采集同时关注随机失败、复杂恢复和更广任务覆盖。

原始回传与本地审计：artifacts/reviews/sft-independent-test-v1-20261102/。
冻结凭据：artifacts/baselines/sft-sql-weighted-v1-20261102-receipt.json。
