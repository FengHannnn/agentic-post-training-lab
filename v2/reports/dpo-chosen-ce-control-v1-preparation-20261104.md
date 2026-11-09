# E3：chosen CE 保真与同硬件加权控制

2026-11-04，Asia/Shanghai。运行 ID：dpo-chosen-ce-control-v1-20261104。

E2 补审计已完成：32.83 秒，208 份原返回文件一致；新增训练和生成均为 0。E2 均匀 DPO 的随机完整任务 2/16、完整恢复 33/84，原加权 DPO 同轮为 1/16、15/84。C1 保持未通过，GRPO 暂缓。

## 本轮问题与目标

检验在原加权 DPO 上加入 chosen CE 后，正确动作似然及完整成功如何变化。当前记录的 GPU 是 L40S，前期登记为 L20，因此本轮同时重新训练一个纯加权控制，固定同一硬件、运行库、精度设置与 reference 字节内容。

两个训练分支从同一 chosen-only 父模型独立开始，顺序执行，每支各 6 遍、258 个 fresh Adam 更新。唯一的分支差异是 chosen CE 系数：控制 0，混合分支 1。新增训练总量为 516 次更新；CE 与 DPO 在同一次反向累积及 optimizer.step 中联合优化。

每个 batch 的混合目标：

~~~text
L = (1/B) × Σ[ w_i × DPO_i + λ × CE_chosen_i ]
CE_chosen_i = -mean(该 chosen 动作所有 response token 的 logp)
λ_control = 0
λ_dpo_ce  = 1
~~~

DPO 继续使用 chosen/rejected 动作的 sequence-sum logp，beta=0.1。困难场景 raw weight=4，其余为1，全数据均值归一到1。CE 按169对样本均匀平均，动作内按token平均，包含EOS；上下文只作为条件。CE 项使用同一次 chosen forward 的分数。

系数1是本轮预先固定的一次诊断：沿用已有 token-mean CE 定义，直接以单位系数相加；本轮不做系数搜索。DPO 优化相对偏好差距，CE 项直接鼓励已有正确动作的 token 似然。E4 将固定该系数与 CE 动作曝光预算，再替换 CE 数据以检验续跑覆盖。

## 精确固定输入

| 项目 | 固定值 |
|---|---|
| 父模型 | chosen-only-sft-v1-20261102，078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55 |
| 每支起点 | SFT255 / DPO11 |
| 数据 | 相同169对，仅train；pairs SHA256 c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329 |
| Reference | E2 保存的父模型评分缓存，SHA256 448b6c8af6fb18ae35fcad02f904167fea1f75d293fff0c53a70968c280b2a65 |
| 优化器 | 每支 fresh AdamW，LR5e-6，weight_decay0，max_grad_norm1 |
| 批次与顺序 | 每次更新4对、尾批1对；seed2026110301，每遍seed+epoch |
| 训练量 | 每支6遍/258更新/1,014对曝光；CE分支另记录1,014次chosen CE动作曝光 |
| 硬件 | NVIDIA L40S；同轮记录GPU UUID、CUDA/PyTorch/transformers/PEFT/vLLM版本与精度设置 |
| 检查点 | 每支43/86/129/172/215及final258，固定final评测 |
| 整体上限 | 14,400秒，实时日志，失败/超时也归档 |

reference_run 名称是 E2 作业目录；缓存的参考策略仍是父模型078182…，并非均匀DPO最终模型。prepare直接复制已审阅缓存；两个分支逐字节复用，同一SHA贯穿输入绑定、状态和最终审计。每个分支在首个更新前重新评分父模型并执行已审核的联合数值检查，记录 fresh_scores 和检查结果。

## 对照与评测

同轮自由生成评测四个模型：

- weighted_sft：已完成的加权SFT，作为表现对照。
- uniform_dpo：E2均匀DPO最终模型，6f6de631322216e24a824f03d3ae4a90929accd2ecbc08624908275384dfe11a。
- weighted_control：本轮重新训练的纯加权DPO。
- dpo_ce：本轮加权DPO+chosen CE。

主要配对是 weighted_control→dpo_ce（保真项作用）；uniform_dpo→weighted_control 补同硬件权重对照；weighted_sft→dpo_ce 列表现差距。每模型完整任务 greedy4+sample16，完整恢复 greedy28+sample84；合计80条完整任务、448条完整恢复，共528条。种子、温度、场景及原12轮预算沿用已冻结协议，test未使用。

FP32评分另包含父模型，共5个模型，各评分相同169对偏好与238个专家探针。探针含32个submit动作；它们用于测量，不进入本轮训练。现有169条chosen动作含query_sql104、calculate_change41、check_result24，CE训练覆盖保持该数据原貌。

重点报告：最终成功、完整恢复、配对改善/退化、stale chosen CE、计算/检查/提交CE、首动作有效到最终成功的转换、重复查询与提交次数。训练日志分别保存联合loss、weighted DPO loss、chosen CE与梯度范数。history.loss记录联合目标，诊断pair scores中的loss继续记录DPO单项。维持258步用于隔离目标项作用，E1的早期检查点曲线单独保留。

混合分支标记 weighted_dpo_chosen_ce，记录 auxiliary_chosen_ce_updates=258。两支每个模型的优化器更新仍各258次；历史SFT计数保持255，累计DPO计数269，联合CE更新单列。

## 云端执行

上传 dpo-chosen-ce-control-v1-20261104-run.zip 到云端 v2 后运行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-chosen-ce-control-v1-20261104-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_chosen_ce.py
~~~

绝对解释器路径指定 apt-vllm 环境，终端显示(base)也可使用。预期回传 dpo-chosen-ce-control-v1-20261104-review.zip。

阶段顺序：prepare → weighted-control → dpo-ce → scores → complete → recovery → audit。保留E2已补审计的artifacts目录及旧模型、数据、基础模型缓存。新run目录保护已有结果；中间adapter与Adam/RNG状态保留云端，review回传两个新final adapter和完整元数据/评分/轨迹/日志。GPU和主机资源按阶段只读采样。

## 本地验证范围

真实输入预检通过：三个冻结adapter、24份溯源文件、169对/reference token对齐、缓存精确复制、238探针（最长1527 tokens）及28个恢复上下文。完整基础模型权重和真实CUDA环境由云端prepare核对。

测试覆盖真实LoRA零CE控制与原加权DPO参数/Adam状态完全等价、CE梯度的动作内token平均及样本均匀权重、联合更新计数、硬件/运行库变更拒绝、五模型原始评分重算与篡改拒绝、四模型原生完整轨迹重放、失败归档。全套163项测试通过；另1项真实非零CE更新通过，训练精度检查范围调整后对应测试复测通过；编译、CLI与输入预检通过。完整回执见随包 handoff/dpo-chosen-ce-local-validation-20261104.json。GPU新实验待执行。
