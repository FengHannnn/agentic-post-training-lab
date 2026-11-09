# DPO training ready

User selected DPO only; no chosen-only SFT control this round. Reviewed 44 same-context pairs, SHA 508464a0314982c7e6ade95c57afa3be081afc80f237e4fb20c8abe2079ffb78. From frozen 126-step SFT, fresh AdamW lr5e-6, beta.1, one epoch, accumulation4, 11 DPO updates. Fixed input-bound detached reference cache. Run scripts/run_v2_dpo.py on L20. Paired greedy train/dev (32 episodes) plus new-seed T=.8 train sampling (96 episodes), audit all 128. Total budget5400s. Local tests82 passed.

See reports/dpo-collect-v1-review-20261102.md and reports/dpo-train-v1-20261102.md. Current common SFT stays frozen; DPO is an independent candidate pending cloud result and local review.
