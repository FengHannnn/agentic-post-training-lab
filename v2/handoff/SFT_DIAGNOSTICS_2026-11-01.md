# SFT diagnostics result

L20 run finished GPU scoring and inference. The audit mode-field collision is fixed in code; frozen returned data was reviewed offline. See reports/sft-diagnostics-v1-review-20261101.md.

Training-task success remains 0/12; format is 144/144; teacher-forced mismatches are confined to SQL (45/72 exact, other actions 180/180). Fixed SFT greedy outputs match across HF FP32, HF BF16 and vLLM on 8 probes. Next: bounded SQL-focused SFT, with explicit month binding and verified_metrics-empty repair data. Preserve pilot and diagnostic outputs. Full local tests: 60 passed.
