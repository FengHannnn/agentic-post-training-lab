# 早期 DPO86 与加权 SFT：阶段收尾直接对照

2026-11-05，Asia/Shanghai。运行ID：dpo-early86-vs-weighted-sft-v1-20261105。

本轮目标：判断现有早期DPO检查点与加权SFT相比，完整任务和恢复表现是否更好或足够接近，作为DPO阶段收尾依据。固定比较两个已有模型，训练更新0。此前围绕末期CE模型的4分支推理确认不在本轮执行。

## 固定候选

| 模型 | 路径 | 本阶段新增训练 | 累计履历 |
|---|---|---:|---|
| 加权SFT | hard-recovery-sft-weighted-v1-20261103/final | SFT258次 | SFT513 / DPO11 |
| 早期加权DPO86 | hard-recovery-dpo-matched-v1-20261103/checkpoint-0086 | DPO86次，2遍/338对曝光 | SFT255 / DPO97 |

两者从相同chosen-only父模型出发。此次选择已有86步模型直接比较实际效果，原训练量各自保留记录。DPO候选在本轮执行前固定，运行中不选择其他检查点。

DPO86 adapter SHA256：43100acde91f178de59ac7a65f29867862a02a7f1a6cbcd134b7d81a7318f46c。complete SHA256：d2e5716ee3955e457eaeb7727781e65f05a4d56fa9442d762ee2f06f60573c1c。

加权SFT adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。

选择依据为E1已审同轮记录：86步随机完整任务13/16，对SFT16/16；greedy完整恢复均21/28。此次固定同轮重新评测并补齐随机完整恢复。

## 对照内容

沿用4个development任务、原7种错误上下文及冻结种子和温度。每模型完整任务greedy4条、随机16条；完整恢复greedy28条、随机84条，共132条。两模型合计40条完整任务+224条完整恢复=264条轨迹。

主指标为随机完整任务成功；辅助报告greedy完整任务、随机/greedy完整恢复、逐任务结果、配对改善/退化、token/轮数/工具错误与首动作后的成功转换。performance_comparison.json直接给出两模型成功数、DPO减SFT的成功数与百分点差距；接近程度根据本轮实际结果用于用户阶段验收。

使用已有development任务做阶段对照；候选已由历史development表现选定。test未使用，本轮不新增任务族、模型分支或训练参数搜索。

## 必要评测检查

每模型独立进程，只加载一份LoRA，固定编号1、名称fixed_adapter、max_loras1；私有复制配置仅规范化target_modules列表顺序，原权重哈希保持一致。vLLM BF16、eager、max_num_seqs1、无prefix cache、原生token/logp，max_new_tokens256、max_model_len8192。

两模型同作业核对GPU型号、软件、精度与实际UUID。跨作业有效UUID变化记录在runtime_precheck，不要求回到历史物理GPU。

每模型附7个固定错误上下文的首动作各生成2遍，共14次小规模调用；两模型总28次。检查原生prompt/response token绑定、有限logp、EOS/预算与相同输入greedy响应一致性，结果写入quality_checks.json，随本轮评测一并完成。

完整轨迹由环境严格重放，独立检查calendar与证据链，重算配对统计。通过输入和记录检查后，依据性能差距讨论阶段收尾。

## 来源与云端检查

prepare固定已审E1的audit/config/run_status SHA及原matched DPO config/input_binding SHA，核对86步、2遍、338曝光及累计SFT255/DPO97；加权SFT核对258次新增SFT及累计SFT513/DPO11。实际模型tensor在推理前逐个核对SHA，原文件在作业前后复查。

86步真实权重按此前约定保留云端；本地已核验其完成记录、adapter配置及E1对真实权重的绑定。云端须保留原hard-recovery-dpo-matched-v1-20261103/checkpoint-0086。返回review包会带回86步真实adapter权重，便于后续封存。

## 执行边界与验证

训练更新0，固定两个模型，无新训练。总时限14400秒、实时阶段日志、同名输出保护；失败/超时仍打包。输出包括原生轨迹/种子、质量检查、直接性能差距、逐任务/场景及配对结果、来源哈希、冻结源码、审计和86步adapter。

5项本地检查通过，覆盖两模型原生完整重放、单LoRA入口和小规模质量检查、配对性能差距公式、固定86步候选/履历校验、零更新失败归档。验证回执见handoff/dpo-early86-local-validation-20261105.json。实际GPU评测等待执行。

## 云端执行

上传dpo-early86-vs-weighted-sft-v1-20261105-run.zip至云端v2后：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-early86-vs-weighted-sft-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_early86.py
~~~

回传dpo-early86-vs-weighted-sft-v1-20261105-review.zip。返回后先简要报告完整任务/恢复差距，再据结果判断是否结束DPO阶段。
