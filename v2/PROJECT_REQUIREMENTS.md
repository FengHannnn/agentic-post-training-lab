# Project V2 Requirements

Initial confirmation: 2026-10-29. Latest confirmed closeout scope: 2026-11-03.

- Goal: internship-ready project in about 10–11 days, averaging 6 hours/day; prioritize LLM post-training / RL algorithm internships, then Agent development.
- Main model: Qwen/Qwen2.5-1.5B-Instruct. Current observed cloud GPU (2026-11-04): NVIDIA L40S.
- Main approach: real multi-turn SQLite tool interactions, verifiable evidence, error recovery, and fine-grained turn-level credit assignment.
- Required closeout: one bounded DPO supplementary comparison, outcome-reward GRPO, one fine-grained-credit GRPO variant, one small genuine dependency-horizon extension, and reproducible results/materials.
- Current experiment status (2026-11-05): conservative short DPO from weighted SFT completed and passed local review. Run dpo-sft-start-short-v1-20261105 stopped at the first passing checkpoint12: sampled complete16/16 versus SFT16/16; sampled recovery59/84 versus62/84; greedy complete4/4 both; greedy recovery21/28 versus20/28. The fixed practical closeout criteria are met. Candidate checkpoint-0012 SHA77c737965c25dfd3110177bb0856938a651cb387bc7153b588b3ce56adf6fc7e has ancestry SFT513/DPO23 and48 pair exposures from12 new DPO updates. DPO supplementary comparison is ready for phase closeout.
- Latest user decision: the user authorized the conservative SFT-start short DPO round; its first passing checkpoint12 is now reviewed and archived. End this supplementary DPO search under the predeclared criteria and return to the original GRPO plan next. The current turn reviewed the returned job and archived results; no new GPU job was launched. E1–E4 mechanism work is now included as extension material in the GRPO handoff; the late-model four-branch inference confirmation stays outside the main path.
- Latest DPO goal: better than or close to the best weighted SFT; checkpoint12 met the fixed practical development criteria. GRPO should strive to improve over the existing candidates.
- Both new GRPO branches start from hard-recovery-sft-weighted-v1-20261103, SHA256 f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f, ancestry SFT513 / DPO11. Preserve the historical pure-SFT baseline separately.
- GDPval, broad multi-file office tooling, a web UI and extra algorithm branches are optional. Multi-teacher token routing / OPD is excluded.
- Primary metric: final task success; core secondary: complete recovery success. Track tool/evidence/SQL failures, no-progress repetition, tokens and time.
- Review progress and freeze new features around day 7; reserve the final 3 days for evaluation, explanation and deliverables.
- Overall training-run cap is not fixed. Discuss material budget/scope changes; each run has finite steps/time and stopping criteria, progress bars and failure packaging.
- Deliver code/config/model identities, curves/tables, recovery cases, README, technical report, resume bullets and interview explanations; demo may be a report, terminal or recording.
- Read new training/probability/credit/reward/evaluation core code closely; summarize repetitive runners, logging and packaging.
- Reports lead with results and evidence, use concise Chinese and avoid defensive repetition. Briefly summarize report results in chat and provide links without automatically opening. English filenames; store originals under project v2.
- Cloud jobs are run by the user from uploaded ZIPs; give direct commands, default upload location is cloud v2, ZIP layout contains v2/. No price parameters or cost estimates.
- Keep train/development/test provenance explicit. Existing four test tasks are historical regression cases; freeze new task families/seeds for final confirmation. If GDPval is added, isolate its tasks, source files and gold materials from training.
- Preserve V1 archive, prior run outputs, model/optimizer/RNG identities and frozen source snapshots.

Canonical closeout plan: reports/project-roadmap-v2-20261103.md.
Latest comprehensive handoff: handoff/PROJECT_HANDOFF_V2_GRPO_2026-11-05.md.
