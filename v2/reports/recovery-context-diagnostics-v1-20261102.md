# 恢复诊断：原因分析与训练／开发上下文对照

日期：2026-11-02。诊断任务：recovery-context-diagnostics-v1-20261102。

开发集已有的 336 次恢复动作已在本地逐条重放，并完成工具、证据 ID 和 SQL 分项归因。新的 GPU 任务将补齐 84 个原样训练上下文，并在同一轮重新生成 28 个开发上下文的动作，比较起点 DPO11、恢复 DPO140、chosen-only SFT 三个模型。该任务只推理，不更新模型参数。

## 我认为结果形成的原因

首先，当前瓶颈是识别环境状态后改变计划。模型经常把“查询已执行、有 qN”当成“该月指标已验证”，继续 calculate_change。实际环境中 qN 可以对应空的验证结果，此时正确恢复是重新查询缺失指标。已有正确月度证据时，模型选择 ID 的表现明显更好。

其次，teacher forcing 的训练分数与自由生成考验的环节不同。计算训练损失时，每个位置都能看到正确动作的前缀；生成时，工具名或关键参数的一次偏离就改变了整段动作。当前 SFT 按整段 response 的 token 平均求损失，工具名和少量 SQL 条件没有额外权重。因此整体 CE 降低与关键决策仍失败可以同时发生。“关键 token 权重不足”作为后续可检验的优化假设保留。

第三，本次增加的是同一批数据的使用次数。169 行只含 123 组不同的上下文与 chosen 组合；恢复上下文来自 12 个训练任务、7 类固定错误场景。三轮续训增加了熟悉样本的梯度次数。训练上下文的自由生成表现将帮助判断当前更偏向拟合不足，还是跨任务迁移不足。

DPO 与 SFT 的差别也体现在实际分数上：DPO 的 chosen 平均 token logp 从 -0.098198 提高到 -0.066612，rejected 从 -0.224551 降到 -0.352743；它同时降低了错误示范的概率。SFT 直接提高 chosen，平均 CE 降到 0.058988。两种目标对同一状态中的竞争动作施加不同约束，行为收益也落在不同位置。本轮 DPO 的重复动作较少，SFT 的格式和少量检查修复更好。

以上判断按已重放的行为与实际损失构成排序；新增对照将直接测量训练上下文上的自由生成，检验拟合与迁移这两种解释。

## 开发集已完成的分项统计

来源：已审阅 chosen-only-sft-v1-20261102 返回包。下表使用随机恢复探针：4 个开发任务 × 7 类上下文 × 每上下文 3 次采样。每个分支 84 次动作。

| 指标 | 起点 DPO11 | 恢复 DPO140 | chosen-only SFT |
|---|---:|---:|---:|
| 有效恢复动作 | 29/84 | 35/84 | 37/84 |
| 与专家下一步工具一致 | 43/84 | 44/84 | 53/84 |
| 工具一致且需要证据时，全部 ID 有效 | 29/31 | 35/36 | 40/41 |
| 全部证据槽位中，指向正确已验证指标 | 91/134 | 105/141 | 114/146 |
| 需要查询的上下文中，实际选择 query_sql | 12/36 | 8/36 | 12/36 |
| 需要查询的上下文中，目标指标验证成功 | 0/36 | 0/36 | 0/36 |
| 全部 SQL 尝试 | 12 | 9 | 13 |
| SQL 执行错误 | 12 | 7 | 10 |
| SQL 执行成功但未验证指标 | 0 | 1 | 2 |
| SQL 验证出任意指标 | 0 | 1 | 1 |

最后一行 DPO/SFT 各 1 次查询发生在检查失败场景，重新查询已经验证过的指标，环境判为 no_progress。目标指标验证成功率按“确实需要查询”的上下文另行统计。

| 证据槽位分类 | 起点 | DPO | SFT |
|---|---:|---:|---:|
| 正确引用已验证指标 | 91 | 105 | 114 |
| ID 存在，但该证据未验证任何指标 | 24 | 23 | 23 |
| ID 存在，但指向另一指标 | 15 | 9 | 4 |
| ID 不存在 | 4 | 4 | 5 |

证据槽位按实际生成的 calculate_change、check_result、submit 所需字段统计。工具一致条件下的 ID 比例另列，便于把上游工具决策错误与参数选择错误区分开。

SFT 在两类最关键状态上的表现：

| 场景 | 应选工具 | 12 次实际选择 |
|---|---|---|
| SQL 未验证后重查 | query_sql | calculate_change 11 次，query_sql 1 次 |
| 重复无效证据错误后恢复 | query_sql | calculate_change 11 次，check_result 1 次 |
| 连续 SQL 错误后重查 | query_sql | query_sql 11 次，格式错误 1 次 |

连续 SQL 错误场景中的 11 次 SFT 查询，有 6 次继续访问不存在的 metrics 表。这里既有工具选择问题，也有选择 query_sql 后未修正 SQL 的问题。

在 failed_check_repair 中，SFT 5 次选了 check_result，5 次证据 ID 都有效；其中 2 次修正 change_cents 后通过，另 3 次仍保留错误数值。因此 ID 有效与数值正确也有独立统计价值。

## 新对照的设计与统计口径

| 项目 | 设置 |
|---|---|
| 模型 | DPO11、恢复 DPO140、chosen-only SFT；固定现有 adapter SHA256 |
| 训练上下文 | 12 个训练任务 × 7 类 = 84 个，与偏好数据 messages 完全一致 |
| 开发上下文 | 4 个开发任务 × 7 类 = 28 个，均未出现在训练 pairs 的 messages 中 |
| 生成 | 每上下文 greedy 1 次、temperature=0.8 采样 3 次 |
| 总动作数 | 3 × 112 × 4 = 1,344 |
| 生成设置 | BF16 vLLM，max_new_tokens256，上下文上限8192，top_p1，top_k=-1；关闭 prefix cache，固定种子 |
| 随机种子 | 开发集沿用前轮种子公式；训练集另加 1,000,000，避免两种划分的种子重叠 |
| 运行时限 | 14,400 秒，即 4 小时 |
| 进度 | 终端实时进度条，显示模型、划分、场景及完成动作数 |
| 训练更新 | 0 |

每种模型、划分、greedy／sample 模式和场景分别输出：

1. 工具选择：专家工具与模型工具分布、工具一致次数、有效动作数和精确重复上一动作次数。工具一致按固定专家计划衡量，其余可推进动作另外计入有效动作。
2. 证据 ID：逐槽位判定正确已验证指标、ID 存在但未验证、ID 指向其他指标、ID 不存在、字段缺失、类型错误；同时记录该指标在当前状态是否已有可用证据，并统计工具一致条件下的 ID 正确率。
3. SQL 验证：查询次数、执行错误、执行成功但未验证、验证了其他指标、验证了目标指标；分别以需要查询的上下文数与实际查询数为分母。
4. 训练／开发差值：有效动作率、工具一致率、工具一致条件下 ID 有效率、应查询时选择查询的比例及目标验证率。没有适用样本的比例存为 null。

诊断使用现有 Environment 的只读执行和原始／扰动数据库验证规则，逐条重放固定前缀，再执行模型动作。

## 已完成的本地验证

- 已重放旧开发集 336 次动作，结果与原有审计记录一致。
- 84 个训练上下文全部逐字命中已审阅 pairs.messages；28 个开发上下文均未命中。
- 112 个上下文的专家下一步全部能推进验证或通过检查。
- 最长 prompt1282 tokens，加256生成上限为1538，配置上下文上限8192。
- 新增 6 项测试通过；完整正式测试目录 113 项通过，compileall 通过。
- 审计测试覆盖两种划分、三模型、原生 token、证据分类、SQL 目标验证、汇总分母以及记录篡改检测。

可重复故障探针：diagnostics/work/probe_recovery_tool_choice.py。实际运行输出 expected_tool=query_sql、actual_tool=calculate_change、provide verified monthly evidence IDs，并按预设断言失败，锁定了需要分析的行为。该探针保留在诊断目录供回归比较。

首次全目录 pytest 收集到了 artifacts 中的历史源码快照；随后限定正式 tests 目录，113 项全部通过。

## 运行与回传

将 v2-recovery-context-diagnostics-v1-20261102.zip 上传到 /root/rivermind-data/AgenticPostTrainingLab，然后执行：

```bash
cd /root/rivermind-data/AgenticPostTrainingLab
unzip -o v2-recovery-context-diagnostics-v1-20261102.zip
cd v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_recovery_context_diagnostics.py
```

输出：
/root/rivermind-data/AgenticPostTrainingLab/v2/recovery-context-diagnostics-v1-20261102-review.zip

返回包包含 contexts.jsonl、context_membership.jsonl、三个模型的逐动作原生生成记录、diagnostic_summary.json、diagnostic_counts.csv、train_development_comparison.json、audit.json 和阶段日志。

训练上下文的模型生成结果由这次 L20 任务补齐。收到返回包后，将以同轮三个分支的训练／开发差值判断拟合与迁移，并给出下一轮训练方向。

本报告原件：v2/reports/recovery-context-diagnostics-v1-20261102.md。
本地已有记录的分项结果：v2/artifacts/diagnostics/recovery-context-diagnostics-v1-20261102/。
