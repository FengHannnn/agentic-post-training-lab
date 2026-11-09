# C1 匹配困难加权 DPO：运行准备

2026-11-03，Asia/Shanghai。状态：代码与本地真实输入预检已完成；新 GPU 训练尚未执行。

## 当前配置

第一份开发配置已冻结，run_id 为 hard-recovery-dpo-matched-v1-20261103。使用父模型 chosen-only-sft-v1-20261102（078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55），其履历为 SFT255 / DPO11。冻结 reference 直接由该父模型在更新前重新评分缓存；新建 AdamW，weight_decay=0。

与已完成的 hard-recovery-sft-weighted-v1-20261103（f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f）匹配以下条件：169 对数据、原始困难权重4/其余1、全数据均值归一、6 epochs、batch4、seed2026110301、学习率5e-6、梯度裁剪1。新 DPO 的 beta=0.1，共258次更新、1014次曝光；完成后履历应为 SFT255 / DPO269。

配置选择遵循已确认收尾计划：开发配置最多两份，本包只含第一份；固定最终检查点，不在中间 checkpoint 中择优。先复核本次结果，再决定是否值得第二份配置；本阶段结束后进入共同起点 f8fb95… 的 GRPO。

## 新增核心

- src/apt_v2/matched_dpo.py：每对仍使用 sequence-sum DPO；loss乘场景权重后按实际batch大小平均，包括每遍最后单样本batch。它同时提供开发集完整恢复执行和原生token严格重放。
- scripts/train_v2_matched_dpo.py：核对父模型履历与reference、重新缓存评分，保存每43步checkpoint、Adam/RNG、chosen/rejected概率和相对margin。
- scripts/matched_dpo_common.py：在训练前核对已有加权SFT的配置、权重、真实更新履历及两个模型和数据身份，冻结开发任务与恢复前缀。
- scripts/evaluate_v2_matched_dpo.py：父模型、加权SFT、新DPO共享任务与种子，完整续跑后再提取首动作诊断。
- scripts/audit_v2_matched_dpo.py：重放所有任务与恢复轨迹，核对原生prompt/response、EOS/长度、首动作绑定、种子、统计、reference、更新计数和优化器状态，并复查跨月SQL。
- 原 independent_recovery_test.py 的跨月审计仅增加显式 allowed_split='development' 入口，默认仍限test；训练样本准入保持train。

核心阅读顺序：matched_dpo.py 的 optimize_weighted_dpo → train_v2_matched_dpo.py 的reference与save → matched_dpo.py 的run_recovery/replay_recovery → audit_v2_matched_dpo.py 的audit_results。runner按预检、训练、完整任务、完整恢复、审计和打包顺序执行。

## 评测验收

只使用4个development任务。三个模型各评贪心4条、随机16条，共60条完整任务轨迹；28个恢复上下文（4任务×7场景）各跑贪心1次、随机3次，共336条完整恢复轨迹。随机温度0.8，top_p=1、top_k=-1。所有首动作均来自这些完整恢复的第一步，不另行采样替代。

主指标为最终任务成功；核心辅助为错误上下文后的完整恢复成功。首动作进展、工具/证据/SQL、错误与token为解释指标。完整恢复沿用原任务12轮总预算，包含已注入前缀；不同场景剩余轮数会有差异。按任务与场景分别汇总，保留父模型→DPO和加权SFT→DPO的配对改善/退化。

已有test-000～003本次不参与开发配置选择；最终新任务族仍待GRPO阶段按计划冻结。跨月挑战只重执行SQL并检查最终证据链，不重新让模型在重排数据库生成轨迹；环境主成功与calendar一致成功分别保存。

## 本地证据

真实169对数据SHA256为c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。来源44 frozen_sft_sample + 84 scripted_failure + 41 frozen_dpo_sample，困难84行/其余85行；困难归一权重占79.809976%。两个真实adapter、原权重文件、tokenizer和chosen监督编码已核对，28个开发恢复上下文与训练messages无重叠。

最长训练序列1376 tokens。每epoch chosen响应17,581 tokens、rejected响应10,359 tokens。DPO训练序列token总计323,652/epoch，对照SFT为165,437/epoch；这些是序列计数，不是FLOPs或耗时。DPO还需reference缓存和前后评分。因此本实验匹配样本曝光与更新安排，报告同时列计算量差异。

全套回归136项通过（348.02秒）；Python源码编译通过。随包 local-validation.json 与 test-results.txt 保留证据。真实GPU训练与vLLM集成尚由云端作业验证，完整基础模型权重将在云端prepare按model_identity校验。本地没有远端shell。

## 云端运行

将 hard-recovery-dpo-matched-v1-20261103-run.zip 上传到云端v2目录后执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/hard-recovery-dpo-matched-v1-20261103-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_matched_dpo.py
~~~

回传 hard-recovery-dpo-matched-v1-20261103-review.zip。作业总时限14,400秒，实时显示训练和评测进度；已有同名输出时拒绝覆盖。超时或失败也回传日志与已完成阶段。review省略Adam文件和中间adapter以控制体积，云端保留完整状态；包括最终adapter、冻结源码、输入身份和评测轨迹。本地复核时使用包内冻结源码，再解释chosen/rejected概率、margin、任务成功与完整恢复差异。
