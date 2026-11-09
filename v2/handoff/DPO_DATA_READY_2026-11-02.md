# DPO data ready

Reviewed 48 frozen-SFT stochastic train episodes (T=.8): 23 successes, 12/24 fault recoveries, 473 turns. All 12 task groups mixed. 50 counterfactual candidates, 44 deduplicated/capped valid same-context pairs across all 12 tasks; max length 1199. Chosen all scripted verified SQL; rejected actual model outputs (40 SQL, 2 malformed JSON, 2 calculate_change). Local regeneration matches cloud exactly. Pair SHA-256 508464a0314982c7e6ade95c57afa3be081afc80f237e4fb20c8abe2079ffb78.

Next small DPO plan: fresh optimizer from frozen 126-step SFT, 44 pairs, one epoch, accumulation4, 11 updates, lr5e-6, beta.1; confirm optional chosen-only control preference. New-seed paired stochastic evaluation plus greedy train/dev. Report: reports/dpo-collect-v1-review-20261102.md.
