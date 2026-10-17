# Agentic Post-Training Lab

面向工具调用 Agent 的 SFT、DPO、GRPO、vLLM rollout 与评测项目。

完整进度、阶段路线、实验结果和接手续作说明见 [项目交接文档](PROJECT_HANDOFF.md)。其中已实现内容与未来计划分别标注。

## 运行环境

- Python：3.11
- 本地虚拟环境：`.venv`
- MiniServe 的虚拟环境和依赖不会被复用。

Phase 1 已实现并验证 SFT、DPO、GRPO 基础损失。Phase 2 已接入真实 tokenizer、CPU LoRA 训练与 adapter 保存重载，并增加独立工具调用评测。

## 阶段文档

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

## 运行测试与数据检查

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\inspect_sft_data.py
```
