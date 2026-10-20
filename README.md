# Agentic Post-Training Lab

面向工具调用 Agent 的 SFT、DPO、GRPO、vLLM rollout 与评测项目。

当前完整进度、阶段路线、结果与接手续作见 [2026-10-18 完整交接](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/PROJECT_HANDOFF_2026-10-18.md)；原始逐步记录保留在 [历史交接文档](C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/PROJECT_HANDOFF.md)。已实现内容与未来计划分别标注。

## 运行环境

- Python：3.11
- 本地虚拟环境：`.venv`
- MiniServe 的虚拟环境和依赖不会被复用。

Phase 1 已实现并验证 SFT、DPO、GRPO 基础损失。Phase 2 已接入真实 tokenizer、CPU LoRA 训练与 adapter 保存重载，并增加独立工具调用评测。

## 阶段文档

最新进展（2026-10-18）：当前仍处于 Phase 4。按用户选择，在云 GPU 接入前完成 CPU reward 契约与离线审计，作为后续训练的接口准备。严格算术奖励第一版已实现，现有 3 个算术组均为零方差；GPU 引擎接入仍待进行，Phase 5 真实 GRPO 闭环尚未开始。全套 160 项测试通过。详见下方 Phase 4 reward 报告。

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

## 运行测试与数据检查

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py
```
