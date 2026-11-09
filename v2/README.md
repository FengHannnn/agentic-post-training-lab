# Agentic Post-Training Lab V2

> 最新接续（2026-11-05）：保守短程 DPO 已复核达标，第 12 步作为首个通过节点自动停止。采样完整任务与同轮加权 SFT 均 16/16；采样恢复 59/84 对 62/84，贪心恢复 21/28 对 20/28，满足预定收尾容差。报告：reports/dpo-sft-start-short-v1-review-20261105.md。

Current handoff: [GRPO continuation handoff](handoff/PROJECT_HANDOFF_V2_GRPO_2026-11-05.md). The next conversation starts C2; no GRPO GPU job has been prepared or launched yet.

## Current: DPO phase closeout criteria met

Reviewed run `dpo-sft-start-short-v1-20261105` selected checkpoint-0012 after12 new DPO updates and48 pair exposures, initialized from the best weighted SFT with that same SFT as a fresh frozen reference. Uniform pair weights, LR5e-7, beta0.1, one exactly resumed Adam/RNG trajectory. Checkpoint4 missed the greedy recovery gate; checkpoint12 passed all four practical criteria, so checkpoint24 was not run.

Selected adapter SHA256: `77c737965c25dfd3110177bb0856938a651cb387bc7153b588b3ce56adf6fc7e`. Ancestry: SFT513 / DPO23. Sampled complete tasks16/16 equal SFT16/16; sampled recovery59/84 versus62/84; greedy complete4/4 both; greedy recovery21/28 versus20/28. Local review replayed396 episodes and1,919 generated actions and recomputed the selection. Candidate registry: `artifacts/candidates/dpo-sft-start-short-v1-20261105/review.json`.

DPO supplementary comparison can close under the fixed development criteria. Next return to the original outcome-reward GRPO and fine-grained-credit GRPO plan; both GRPO branches retain the previously confirmed weighted SFT initialization. E1–E4 diagnosis is preserved in the comprehensive GRPO handoff appendix.

Main model: Qwen2.5-1.5B-Instruct, pinned revision 989aa79.

Initial task: multi-turn business-data reconciliation with SQL evidence, validation and error recovery.
Research track: verifiable turn-level potential rewards with terminal outcome verification.

## Local checks

Run from v2: python scripts/prepare_v2.py
Tests: python -m pytest tests -q

## Cloud baseline

python scripts/run_v2_baseline.py

Use the verified local model cache. Add --download-model only when initially preparing an uncached model.
The job has a 30-minute time limit; no hourly-price input or monetary estimate is used.

This first cloud job evaluates four development episodes and records native action tokens, tool observations, rewards and final artifacts. The LoRA update kernel is tested locally; model training follows the baseline review.

See PROJECT_REQUIREMENTS.md and reports/v2-initial-implementation.md for the confirmed scope and artifact reading order.
