# SQL-weighted SFT result

Reviewed L20 run sql-sft-weighted-v1-20261101: 63 extra Adam updates, total 126, unchanged examples. Train 12/12, development 4/4, recovery 6/6 and 2/2; valid monthly SQL 24 and 8. Local replay of 32 episodes/312 turns matches cloud audit. Report: reports/sql-sft-weighted-v1-review-20261101.md.

Candidate identity: artifacts/candidates/sql-sft-weighted-v1-20261101/candidate_identity.json. Model test split has not been evaluated. Suggested next step: independent test evaluation, then freeze a common SFT starting point for DPO/GRPO. Optional future controls: equal-step uniform-weight continuation; broader SQL recovery/task families. User requests links to reports in project v2/reports and no automatic opening.
