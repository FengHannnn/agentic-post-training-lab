# Hard recovery weighted SFT ready — 2026-11-03

User authorized: 进行困难恢复定向加权sft. Implemented and packaged GPU job; no cloud execution from local session. User runs commands and returns review ZIP.

Config configs/hard-recovery-sft-weighted-v1.json, run hard-recovery-sft-weighted-v1-20261103. Initial current chosen-only-sft-v1-20261102 final adapter SHA078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55, history SFT255+DPO11. Load adapter, freshAdamW lr5e-6 weightdecay0,FP32,LoRAq_proj/v_proj r8alpha16 dropout0,1089536params. Same169pairs SHA c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329, samechosenexamples andencoded fingerprints. Source diagnosticrun recovery-context-diagnostics-v1-20261102 bindsinitialmodel.

Hard4 scenarios: sql_unverified_requery24,repeat_evidence_error12,repeat_sql_error24,failed_check_repair24 =>84rawweight4. Others originalSQL44,format16,unknown12,stale13 =>85weight1. Normalizeglobalmean421/169; hard1.6057007,other.40142518,hard totalshare79.81%. Sampleweights only; no tokenweights/no newdata,no DPO/rejected/reference loss. All169rows same promptmask-EOS. Eachresponse tokenmeanCE*weight thenbatchmean. 6epochs258steps (43/epoch),1014exposures,105486toksupervisions,totalSFT513,DPO11. Saves43,86,129,172,215 plusfinal258,allwithAdamstatecloud.

Newfiles: src/apt_v2/hard_recovery_sft.py; train/evaluate/audit/run_v2_hard_recovery_sft.py scripts; tests/test_hard_recovery_sft_v2.py7tests. Existingcodeunchanged. Testsfull120passed1existingPEFTwarning166.18sec;preflightrealdata passed,all112expertcontextsproductive,84trainmessages matchpairs,28devnotinpairsmessages,longestsample1376. scripts compiled packaging. No subagentsused.

Pipeline train→checkpoints→recovery→greedy→sample→audit. 4htime14400sec,live258step and420/896actionbars. Fiveintermediatecheckpoints greedytrain84each=420actions. Before/final bothsplits112contexts*(greedy1+sample3)*2=896actions. Same recoveryseedformula/trainoffset1e6/dev0. Aftertrainingcheckpointprobes retrospective,notadaptive selection. Finalfixedstep258 model. Completebefore/final greedytasks train12dev4 plusdevrandom16each=64episodes. NativeBF16vLLM8192ctx256newtok,seedspreserved;maxLoRAcapacity powerof2intermediate8. Sharedrecoverydiagnostics audit andnative/token/prefix/environmentchecks reused. AuditorchecksparentinitialchosenFP32 logps againstparentafterchosenscores1e-5;weights,encodedhash,Adam perparam258,checkpointschedule/statehash,taskreplays,final896probes,mid420probes. Outputcheckpoint_progress.json,train_devcomparison,CSV,originalrows. FinaladapterincludedreviewZIP;intermediateadapters/state.pt remaincloud auditedhashes.

Release: artifacts/releases/hard-recovery-sft-weighted-v1-20261103/v2-hard-recovery-sft-weighted-v1-20261103.zip,188584bytes91boundfiles,SHA7d77215580c8de3e8f4475468c56dff353295e6d1e8c3bb921d784c860eb1bbd. Manifest hard_recovery_sft_release_manifest.json. Report reports/hard-recovery-sft-weighted-v1-20261103.md,preflight artifacts/datasets/hard-recovery-sft-weighted-v1-20261103. ZIP/receipt/report mirrored chatoutputs. Userprefs noautoopen,projectv2storage,commandsdirect,plainChinese avoiddefensive phrasing.

UploadZIP CLOUDv2 (userpreviouspathconfusion):
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/v2-hard-recovery-sft-weighted-v1-20261103.zip -d /root/rivermind-data/AgenticPostTrainingLab &&
cd /root/rivermind-data/AgenticPostTrainingLab/v2 &&
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_hard_recovery_sft.py

Return /root/rivermind-data/AgenticPostTrainingLab/v2/hard-recovery-sft-weighted-v1-20261103-review.zip.

Uponreturn: CRC/traversal safeextract; sourcefilescurrent/snapshot/provenance compare toalreadyreviewed parent/data/diagnostic; tokenizerandencoding/weights/membership; verify_weighted_updates withoutstate(localstatecloudonly), history,scores/parentprobabilities; audit_task_evaluations+sharedaudit_diagnostics+audit_checkpoint_actions runlocally frozenverifiedsource; recomputecheckpoint_progress bymapping parent pathlocally (ROOTdiff) orderivefromcompletemetadata; intermediateweights notreturned. Cloudauditorcheckedmetadatahash. Review when criticalqueryfirsttoolprob/greedy flips byepoch,SQLtargetverification,check_result wrongvalues,normalflows andsamplerate. Keepcandidateuntilreviewed. No claimGPUtrainingperformedlocal.
