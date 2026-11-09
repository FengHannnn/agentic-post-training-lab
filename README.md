# Agentic Post-Training Lab

面向工具调用 Agent 的 SFT、DPO、GRPO、vLLM rollout 与评测项目。

主体框架、阶段路线与截至2026-10-27的进度见 [完整交接](PROJECT_HANDOFF_2026-10-27.md)；2026-10-28新增进展见[候选刷新与第二轮有限更新结果](reports/phase5-policy-iteration-results.md)。[2026-10-18 交接](PROJECT_HANDOFF_2026-10-18.md)和[原始逐步记录](PROJECT_HANDOFF.md)保留历史状态；旧的“下一步/尚未开始”应结合最新结果阅读。

最新结论：Phase 4 工程已收尾；Phase 5 已完成首轮16步更新，以及从首轮候选刷新144×8 rollout后的第二轮15步更新，延续Adam累计到31步。训练诊断严格28/48→30/48，开发严格仍为36/56，基线未晋升。用户选择继续Phase 5收益排查，暂不进入Phase 6；已完成[现有记录和CPU固定概率诊断](reports/phase5-grpo-effectiveness-diagnostic.md)。选项5的[24题受控采样探索已执行并复核](reports/phase5-sampling-exploration-results.md)：固定31步候选，三个温度共2304条；1.0下每题8→32条仍未覆盖16道重点难题，高温仅覆盖同1题且格式退化。未更新模型；现已实现[离线分级奖励并重评分4920条回答](reports/phase5-graded-reward-rescore.md)，温度1.0的16道重点题有差异组从严格0→等价二元8→分级16，其中后8题仍无满分答案。原奖励与旧训练器保留，开发集没有新增训练收益；[同起点两奖励训练对照已执行并复核](reports/phase5-reward-comparison-results.md)：两组各6步，开发严格36/56、新验证正确44/56均未变化，分级组训练诊断严格30→29；未晋升基线，下一步建议补齐24道实际训练题评测；[原准备方案](reports/phase5-sampling-exploration-preparation.md)保留历史状态。正式训练保留串行；微批8的6.46倍局部加速及失败logp门仍按[原结果](reports/phase5-grpo-microbatch-results.md)保留。最新实验准备见下方记录，测试数以新准备报告为准。下方按日期保留阶段推进记录。

## 运行环境

- Python：3.11
- 本地虚拟环境：`.venv`
- MiniServe 的虚拟环境和依赖不会被复用。

Phase 1 已实现并验证 SFT、DPO、GRPO 基础损失。Phase 2 已接入真实 tokenizer、CPU LoRA 训练与 adapter 保存重载，并增加独立工具调用评测。

## 阶段文档

最新进展（2026-10-21）：Phase 4 工程主线已按新契约验收，保留原跨引擎数值门失败和调度 logp 差异限制。L20 的 48×4 最终包已取回重审：192 条 EOS，严格成功 97，19 组全对、19 组全错、10 组混合；训练侧无更新 ratio=1，单次串行/批量测量约 3.59 倍。Phase 5 真实 GRPO 更新尚未开始，建议先用全部 144 个算术训练输入×8 做有限信号扩展。见下方最终验收报告。

Phase 5 数据准备已完成：L20 的 144×8 实际包取回并重审通过，62 个混合奖励组、496 条非零 advantage 回答；同批前4条有47个混合组，增至8条多发现15组。全套已有289项测试通过；真实 GRPO 更新、KL、监控与行为对照尚待实现。

2026-10-27：首次有限真实GRPO更新代码已实现，加入冻结reference、sampled k3、最多16步、LoRA反向、Adam恢复与同条件生成对照；全套305项测试通过。135M模型的云端更新尚待执行，不提前宣称训练有效。

2026-10-27实际结果：首轮真实16步更新已完成并独立复核，120个LoRA张量变化，基础权重不变，恢复/重载通过。严格通过数训练诊断28→28、开发36→36，无新增结构退化，未证明严格收益、不晋升基线；下一步若继续，需更新策略后刷新rollout，不能重复旧数据无限训练。

- [Phase 1：后训练目标函数基础](reports/phase1-foundations.md)
- [Phase 2：SFT 数据流水线与 LoRA 训练](reports/phase2-sft-data.md)
- [Phase 2：接入真实 tokenizer](reports/phase2-real-tokenizer.md)
- [Phase 2：LoRA 最小训练闭环](reports/phase2-lora-training.md)
- [Phase 2：独立工具调用评测](reports/phase2-tool-evaluation.md)
- [Phase 2：扩充格式与路由数据](reports/phase2-expanded-data.md)
- [Phase 2：乘法与括号组合数据 v2](reports/phase2-bracket-data.md)
- [Phase 2：括号与文字问法诊断](reports/phase2-bracket-diagnostic.md)
- [Phase 3：冻结 v2 SFT 基线](reports/phase3-dpo-baseline.md)
- [Phase 3：小规模偏好数据](reports/phase3-preference-data.md)
- [Phase 3：DPO 训练、保存与重载](reports/phase3-dpo-training.md)
- [Phase 3：SFT 与 DPO 行为比较](reports/phase3-behavior-comparison.md)
- [Phase 3 后续：偏好概率诊断与受控对照设计](reports/phase3-preference-diagnostic.md)
- [Phase 3 后续：chosen-only 继续 SFT 对照](reports/phase3-sft-control.md)
- [Phase 3 后续：均衡偏好数据 v2 与规模对照](reports/phase3-balanced-preferences.md)
- [Phase 3 后续：v2 DPO 规模对照训练](reports/phase3-dpo-scaling-training.md)
- [Phase 3 后续：v2 DPO 规模对照的实际生成评测](reports/phase3-dpo-scaling-behavior.md)
- [Phase 3 后续：定位并修复 DPO 生成退化](reports/phase3-dpo-repair.md)
- [Phase 3 后续：采集冻结 SFT 的训练错误](reports/phase3-sft-training-errors.md)
- [Phase 3 回顾与38同题三组对照方案](reports/phase3-matched-control-plan.md)
- [Phase 3：38同题三组对照执行与结果](reports/phase3-matched-control-results.md)
- [Phase 4：CPU 随机采样与 rollout 数据接口](reports/phase4-rollout-contract.md)
- [Phase 4：CPU reward 契约与离线审计](reports/phase4-cpu-reward-contract.md)
- [Phase 4：智川云资源与接入方案（总预算 200 元）](reports/phase4-zhichuan-cloud-plan.md)
- [Phase 4：云端环境检查与固定产物迁移](reports/phase4-cloud-transfer.md)
- [Phase 4：云端固定 SFT 最小模型加载](reports/phase4-cloud-model-smoke.md)
- [Phase 4：云端单序列完整前向概率重算](reports/phase4-cloud-probability-replay.md)
- [Phase 4：GPU 4×4 随机 rollout 与概率、奖励审计](reports/phase4-vllm-random-rollout.md)
- [Phase 4：随机 rollout 数值差异定位与有限路径诊断](reports/phase4-rollout-numeric-diagnosis.md)
- [Phase 4：训练侧概率契约与无更新 ratio 检查](reports/phase4-trainer-probability-contract.md)
- [Phase 4：采样分布校正契约与策略损失接入](reports/phase4-rollout-correction-contract.md)
- [Phase 4：一次性收尾安排与执行入口](reports/phase4-closeout-plan.md)
- [Phase 4：最终验收与后续数据规模建议](reports/phase4-closeout-results.md)
- [Phase 5：144×8 随机 rollout 与训练信号准备](reports/phase5-rollout-signal-preparation.md)
- [Phase 5：144×8 训练信号批次实际结果与验收](reports/phase5-rollout-signal-results.md)
- [Phase 5：有限真实GRPO更新与保存恢复方案](reports/phase5-first-grpo-update.md)
- [Phase 5：首次真实GRPO更新结果与独立复核](reports/phase5-first-grpo-results.md)
- [Phase 5：候选策略刷新与第二轮有限更新准备](reports/phase5-policy-iteration-preparation.md)
- [Phase 5：候选刷新rollout与第二轮有限更新结果](reports/phase5-policy-iteration-results.md)
- [Phase 5：GRPO效果停滞的现有记录与固定概率诊断](reports/phase5-grpo-effectiveness-diagnostic.md)

## 运行测试与数据检查

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py
```

## 2026-10-28: Correct Demonstration Warm Start

按用户选择，继续Phase 5。已准备[少量正确示范SFT热启动、重新采样与有限GRPO](reports/phase5-demonstration-warmstart-preparation.md)：原31步候选A、热启动B、GRPO后C分别评测，B−A与C−B分别归因；24条chosen示范、18步SFT、B重新采样768条、仅混合正确性二元组最多6步GRPO。GRPO沿用原Adam累计31步动量，SFT使用独立优化器；云端本轮尚未执行，未宣称提升、不晋升基线。

## 2026-10-28: Demonstration Warm Start Results

[正确示范热启动结果已复核](reports/phase5-demonstration-warmstart-results.md)：18步SFT工程通过，实际24题验证正确13→14；原48题38→35，开发56题44→39。全部8个正确转错集中在必要括号丢失，行为验收失败后停止，新采样0条、GRPO0步。24条示范无left_grouped且无括号开头答案；下一轮建议仅调整示范覆盖、保持18步与72例次。保留原31步候选，不晋升本轮模型。

## 2026-10-29: Balanced Demonstration Warm Start

按用户选择，已实现[调整示范组成的受控热启动](reports/phase5-balanced-warmstart-preparation.md)：保留24条、加入12条训练集left_grouped示范，36条各两轮，仍72例次18步和5e-5学习率，新增12题单列评测。使用新脚本和目录，从原31步候选开始；旧失败结果保留。云端本轮尚未执行。

## 2026-10-29: Balanced Warm Start Results

[示范组成调整结果已复核](reports/phase5-balanced-warmstart-results.md)：上轮8道括号退化题在本轮B/C全部正确；18步SFT后原24题验证正确13→15，原48题38→37，开发44→44。B新采样768条，17个混合组完成5步GRPO，开发验证正确44→45、严格36→37；原24/48题C相对B不变。8道natural难题贪心仍0/8，B随机采样仅1/256正确。未达到显著提升、未晋升基线；下一步建议从C刷新采样，先核对概率与信号，不自动追加训练。
