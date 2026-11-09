# Hard recovery weighted SFT reviewed — 2026-11-03

User returned hard-recovery-sft-weighted-v1-20261103-review.zip. Complete review, report reports/hard-recovery-sft-weighted-v1-review-20261103.md, mirrored outputs;countsCSV mirrored outputs. Review source artifacts/reviews/run/extracted/run. ZIP SHA e385144bb1bee41961e77ecc27e6cd9b3faf6aecd93821ed5af0f896644d3e7b,9,922,466bytes155members.

All6stages0,4509.1sec75.2min total,train389.6sec6.5min,peak6.416GiB. 6epochs258updates1014exposures,169samepairs,hard84 raw4other85raw1 normalizedmean1share79.8%,sampleweights only no tokenweight. ParentchosenSFT255+DPO11 adapter078182...,finalSFT513+DPO11 adapterSHA f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f. Candidate artifacts/candidates/run/review.json status reviewed_hard_recovery_gain_ready_for_independent_test,baseline_promotedfalse.

Localreview verified76sourcefiles current/frozensnapshot,12provenancefiles,sameparent/data/tokenizer/examplesencoded/weights/membership,initialparentFP32scoresmaxdiff0,258history/checkpointmetadata/finaladapter112changedfinite. Replayed64episodes508turns +896initial/finalrecovery +420midcheckpointactions=1316allstop. CSV/checkpointprogress/train-devcomparison/evaluation/audit allrecomputedmatch. state.pt/intermediateadapters omittedreview;cloudauditchecksstatehashesandperparam258;localreviewhistory/meta/finalweightsonly.

Mainwithinrunbefore/after:
- Fullgreedytrain12/12&dev4/4both.
- Devfullrandom11/16→15/16;paired4improve0regress11bothsuccess1bothfail. Turns144→124,toolerrors46→15 (temp8both),evidenceerrors25→7. Allnew4successes havecorrectfirsttwoSQLqueries then7stepnormalplan;notin-episodeerrorrepair.
- Trainrecoverygreedy36/84→82/84;dev13/28→26/28. Notecurrentbeforedev13 vsprior12 samefrozenadaptercrossrunvariation,usethiswithinrunbaseline.
- Trainrecoverysample111/252→215/252;dev37/84→74/84.
- GreedySQLtargettrain0/36→36/36;dev0/12→12/12. SampleSQLtargettrain0/108→92/108,dev0/36→33/36.
- Greedytoolall84/84train28/28dev after;matchedtoolIDgood48/48train16/16dev.
- Sampletooltrain165/252→247/252,dev52/84→83/84;matchedtoolID123/126→142/143train,40/41→47/48dev.
- Sampleeasyscenarios stable;hardfinaltrain(36each):unverified32,repeatevidence31,repeatSQL29,failedcheck17;dev12each12,9,12,5.
- Checkpointtrain effective0:36,43:37,86:59,129:70,172:76,215:77,258:82 /84.
- SQLtarget0,0,21,31,36,36,36 /36. Failedcheck0,1,3,3,4,5,10 /12.
- Remaininggreedytrain001/004 failedcheck copywrong+1difference;dev000/001 likewise failedcheck2fails. AllIDsvalidbutwrongchange.
- Remainingfulltask dev001rollout3:temporaryfaultthenbothmonthlySQLexecuteunverified;refundjoin incorrectly filters orders by metricmonth,omitscrossmonthrefunds;7calculate_changeerrors. Existing sql_unverified_requery/repeat_evidence_error traincontexts missingcurrentwithpriorvalid;bothqueriesexecutedunverified combinationcanbenewdatafocus. repeatSQLerror also bothmetricsmissing butdifferentobservations(invalidtable),so do notclaimnoexistingbothmissingcontexts.

Tokenanalysisusingcurrentbefore/after169FP32chosen scores/84distincttraincontexts alignednativepretoolprefix: expectedquerysqlunverified2.87→89.57%,repeatevidence10.35→88.18%;repeatSQLquery99.84→99.997%;failedcheckcheck44.82→98.19%. All4toolsgreedy12/12 after. SampleweightedCE .050648→.015039(70.3%drop),unweighted.058988→.029675(49.7%). Thisroundcombinesweightand258updates,describeoveralltrainingeffectnotisolatedweightcausality.

Localwork scripts: inspect_hard_recovery_sft_return.py,review_hard_recovery_sft.py,analyze_hard_recovery_sft.py. Localreview summary,audit,behavior_analysis.json,tool_token_analysis.json underreviewdir. Candidate registeredreadyindependenttest. Recommend next independenttest complete tasks +recovery,focus numericcorrection,bothqueriesunverified/crossmonthrefunds. No newGPUjobpreparedthisblankattachmentreview. build_context currentlyonlytrain/development,independenttestrecoveryneeds explicit test-evaluation support ensuringtrainpairvalidationstilltrainonly. Usershouldauthorize next. Cloudartifactrunfullpathremainsavailable.

Userpref directcommands,uploadZIPcloudv2,absoluteunzipinputv2/file and -d projectparent(zipcontainsv2),4hbudgetliveprogress,noautoopenreports,projectv2storage,Chineseplainwritingavoiddefensivedisclaimers. Finaldeliverablelinks onlychatoutputs mirroredfromproject.
