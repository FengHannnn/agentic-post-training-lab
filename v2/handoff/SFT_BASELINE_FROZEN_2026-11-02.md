# Frozen V2 SFT baseline

Independent model test reviewed: 4/4 successful, 2/2 recovered, valid SQL 8, valid checks/submits 4 each. All 8 paired episodes/78 actions replayed; no training updates. SFT common parent is frozen locally at artifacts/baselines/sft-sql-weighted-v1-20261102. See artifacts/baselines/current_sft_baseline.json and reports/sft-independent-test-v1-review-20261102.md.

Adapter SHA-256 e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0, total training updates 126. Train/development/test 12/12, 4/4, 4/4 on the current synthetic family. Test exposure is now recorded. Next: stochastic train trajectory collection and DPO preference construction; GRPO starts separately from the same frozen SFT, not from DPO by default. Define optimizer starts per experiment. Current cloud final path is artifacts/sql-sft-weighted-v1-20261101/final; the independent freeze directory was created locally and is not yet copied to cloud.
