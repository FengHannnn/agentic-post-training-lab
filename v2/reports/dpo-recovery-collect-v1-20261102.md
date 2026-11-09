# DPO 恢复偏好扩充：数据采集准备

日期：2026-11-02。运行 ID：dpo-recovery-collect-v1-20261102。

本轮扩充训练任务中的格式修复、证据恢复、SQL 未验证后重新查询，以及避免重复错误调用的偏好。先完成数据生成、采样与审计，复核后固定训练配置。

## 已完成的本地数据

使用真实 Qwen tokenizer，本地构建并验证 84 对规则恢复偏好，覆盖 12 个训练任务、7 类恢复上下文。完整保留原 44 对 SQL 偏好，共 128 对；最长 prompt+response 为 1376 token，低于 2048 限制。参数更新数为 0。

数据目录：artifacts/datasets/dpo-recovery-scripted-v1-20261102。包含 contexts.jsonl、scripted_pairs.jsonl、anchor_pairs.jsonl、pairs.jsonl、preflight_summary.json。

| 恢复场景 | 上下文与 rejected | chosen | 本地新增对数 |
|---|---|---|---:|
| format_repair | 之前输出带说明前缀的无效动作，继续重复 | 合法 JSON 的 calculate_change | 12 |
| unknown_tool_repair | 调用了不存在的工具，继续重复 | calculate_change | 12 |
| sql_unverified_requery | 本月 SQL 执行成功但未验证，仍计算差额 | 重新查询本月净额 | 12 |
| repeat_evidence_error | 引用未验证证据失败两次，继续重复 | 重新查询缺失指标 | 12 |
| stale_evidence_repair | 已有正确月度证据，但引用错误编号 | 使用实际证据编号计算差额 | 12 |
| repeat_sql_error | 错误 SQL 已失败两次，继续重试原 SQL | 正确 SQL 查询 | 12 |
| failed_check_repair | 结果检查失败，重复提交同一错误值 | 修正数值并重新 check_result | 12 |

本地 128 对的 chosen 分布：query_sql=80、calculate_change=36、check_result=12。原 44 对均为 SQL；新增数据将训练目标扩展到后续动作的正确选择与参数。

本地预检 pairs.jsonl SHA256：5894d8b1c06dea3b5082cbdd3717bfbcab5ae9f6e16458ecd3c436d403e756a2。

## 云端采样计划

冻结已复核的 11 步 DPO 适配器，在 84 个规则构造的恢复上下文中，各进行 3 次单动作随机采样，共 252 次模型生成。temperature=0.8、top_p=1、top_k=-1、基础 seed=2026110241，单动作最多 256 token。

这些是受控恢复上下文中的单动作采样。每次生成保存原生 prompt/response token、raw log probability、EOS/结束原因及 seed。模型动作在同一动作前状态中执行；失败动作或无新增进展的动作，与经过验证且有实际进展的专家 chosen 配对。输出被截断的采样保留记录，偏好筛选要求完整 EOS。

最终选择保留：原 44 对 + 新规则偏好 84 对 + 每恢复上下文最多 1 对去重后的模型失败偏好（0～84 对），共 128～212 对。若模型在某上下文采样全部正确，该上下文仍有规则构造样本，实际采样情况按原始结果统计。

## 来源与验证

三种 rejected 来源独立标注：frozen_sft_sample（原 44 对）、scripted_failure（新规则负例）、frozen_dpo_sample（新原生模型负例）。规则负例 token 通过 chat template 编码，原生模型负例保留采样 token 与概率；两者的 token 来源分别写入 rejected_token_origin。

所有恢复前缀通过真实 Environment.step 产生观察，包含完整错误反馈。chosen 根据当前公共进度与证据生成，使用独立重放的相同状态验证。SQL 未验证时优先重新获取缺失指标；指标已验证但引用编号错误时使用现有证据编号。

数据仅来自 train 的 12 个任务。准备阶段核对冻结 DPO、原 SFT 来源、原 44 对偏好身份、train task/family；采集前后固定文件指纹。独立 test 轨迹保留在评测流程。

最终 audit 重新生成 84 个上下文与规则偏好，核对 252 次采样的任务、场景、seed、原生 token 和评分，重建模型候选及最终混合数据，并逐项比较保存结果。

## 代码与本地验证

新增 src/apt_v2/dpo_recovery.py、scripts/collect_v2_dpo_recovery.py、scripts/run_v2_dpo_recovery.py、configs/dpo-recovery-collect-v1.json 和 tests/test_dpo_recovery_v2.py。

encode_recovery_pair() 同时支持新恢复偏好和原 44 对的原生 token 格式，为后续训练提供统一 prompt/chosen/rejected token 接口。

93 项测试通过，Python 编译检查通过。测试覆盖全部 84 个场景、专家进展、未验证 SQL 后重新查询、错误证据后的恢复、规则/原生来源区分、完整 252 次模拟采样的混合数据构建与审计、去重上限和失败回传。

## 云端执行

上传 v2-dpo-recovery-collect-v1-20261102.zip 至 /root/rivermind-data/：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
unzip -o v2-dpo-recovery-collect-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_dpo_recovery.py
```

全流程依次 prepare → collect → build → audit，时间预算 5400 秒，优化器更新数为 0。结束后回传：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-recovery-collect-v1-20261102-review.zip。回传包省略模型权重和优化器文件，保留上下文、采样、偏好、日志、源码及审计。

## 下一步

复核真实模型新增负例的数量、来源和类型分布，固定最终数据指纹，再制定恢复专项 DPO 的训练与开发集评测配置。本轮继续使用现有验证环境；月份关系覆盖增强作为独立分支记录。