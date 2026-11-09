# Chosen-only SFT reviewed — 2026-11-02

User returned chosen-only-sft-v1-20261102-review.zip. Review completed; report: reports/chosen-only-sft-v1-review-20261102.md.

Local audit: 96 episodes, 792 turns, 336 recovery actions; cloud audit matches. Chosen169 rows, unique123 contexts/responses, 3 epochs, 129 steps; initial DPO11, fresh Adam. Final adapter SHA256 078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55.

Same-run DPO11 / DPO140 / chosen SFT: random dev11/16 each; greedy train12/12 dev4/4 each. Recovery random29/84,35/84,37/84; greedy12/28 each. SFT two new actions correct failed-check change_cents (-3257 to -3258;8651 to8650), check passes, submitted false. SQL unverified/repeated evidence/repeated SQL all0/12. Random complete evidence errors21/7/25; repeated recovery actions15/10/15. SFT format errors0 in full random episodes.

Report uses current three-model run; DPO140 recovery varied from prior34 to35. Keep candidates, no baseline promotion. Recommend train/development same-context recovery diagnostic before choosing targeted weighted SFT or additional preferences. No next GPU run prepared. User wants reports stored in v2 and links only, no auto-open; direct commands for GPU execution, generous time budget and streamed progress.
