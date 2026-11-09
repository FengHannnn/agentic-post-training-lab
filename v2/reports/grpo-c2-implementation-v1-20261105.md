# C2 GRPO 本地实现与 GPU 预检交付

日期：2026-11-05，Asia/Shanghai。

## 本轮结果

C2 结果奖励 GRPO 的本地核心实现、有限组 runner、恢复状态与审计入口已完成。104 项测试通过，耗时83.85秒。发布包通过 ZIP CRC、191个归档条目的覆盖检查、190份文件逐项 SHA256 核对、路径检查和干净目录命令入口检查。W 实际112个张量、1,089,536个参数全部有限，权重哈希匹配已确认起点。

当前交付是最长60分钟的 GPU 预检包。下一步由云端执行实际采样、HF评分更新、checkpoint保存、推理刷新及再次采样，并返回显存、吞吐、概率差异和有效组比例。正式训练日程将据此冻结。

## 实现内容

| 入口 | 内容 |
|---|---|
| src/apt_v2/grpo.py | 正常/恢复起点、完整续跑、任务/状态/版本绑定、原生token准入、组内优势、严格重放 |
| src/apt_v2/grpo_optimizer.py | 响应后缀logits评分、old/ref/q缓存、TIS与clipped surrogate、k3 KL、轨迹均衡损失和跳过更新 |
| scripts/grpo_vllm.py | processed行为概率、同token概率探针、单LoRA推理与worker实际加载张量核对 |
| scripts/grpo_worker.py | GPU准备、采样、FP32 LoRA更新、reference冻结、adapter导出、Adam/RNG保存与恢复 |
| scripts/run_v2_grpo.py | 总时限、事务恢复、实时终端进度、日志、资源快照和失败回传ZIP |
| scripts/audit_v2_grpo.py | 轨迹与优势重算、实际权重差异、Adam计数、缓存和最新策略再次采样审计 |
| scripts/build_v2_grpo_release.py | 冻结源码与W输入、构建发布包、逐文件校验 |

所有新增实现位于项目 v2。原有 DPO 训练代码及历史实验产物保留。

## 首版配方

W 同时作为策略起点和冻结 reference；新增 RL 更新另记。train 正常/恢复起点按组1:1混合，K=4，恢复前缀占原12步预算且不参与梯度。

R = success - 0.01 × 总动作数；A = (R - 组均值)/(组总体标准差 + 1e-8)。每条轨迹的所有动作共享 A。完全相同回报的组跳过并记录；全成功但动作数不同的组保留有效优势。

temperature=1、top_p=1、top_k=-1；最大响应256 token、上下文8192。q 直接使用 processed_logprobs。vLLM 将该字段定义为经过采样处理后的概率：[v0.11官方说明](https://docs.vllm.ai/en/v0.11.0/configuration/engine_args.html#logprobs-mode)。云端对首个动作的最多16个真实token进行同权重teacher-forced概率探针。

FP32 HF侧重算old/ref。LoRA AdamW初始学习率1e-6，clip0.2、KL系数0.02、TIS上限2、梯度裁剪1；sampled k3>0.05或clip_fraction>0.5时跳过。动作内token平均、轨迹内动作平均、轨迹间平均。每组最多一次更新，更新后以新进程加载不可变checkpoint再次采样。

每次采样进程向worker查询实际已激活的LoRA张量，并与目标checkpoint通过vLLM相同加载路径所得张量逐项比较。导出的真实adapter还由本地测试完成重新加载及评分一致性验证。

## 验证记录

本地环境：Python3.11.5、torch2.14.0+cpu、Transformers4.57.6、PEFT0.17.1、SQLite3.42.0。

测试覆盖：组均值/总体标准差/零方差；完整失败轨迹准入；七种真实SQLite恢复前缀；剩余步数与严格重放；EOS和响应长度；原生prompt/response/q绑定；后缀logits与完整前向对齐；实际LoRA梯度和更新；基座及reference冻结；KL跳过且Adam计数不增加；轨迹均衡损失；Adam/RNG恢复后下一步逐参数一致；生产adapter导出与重新加载；过期worker权重检测；真实子进程终端进度转发；失败打包；driver从已提交组恢复。

测试套件：test_grpo_v2.py、test_agentic_v2.py、test_hard_recovery_independent_v2.py、test_dpo_sft_start_v2.py，共104项全部通过。较长的原有多分支审计测试完成了SQLite重建与重放。

## 云端预检内容与额度

- 最长3600秒，预留120秒打包；最多4个训练组、2次实际更新。
- 工程通过条件：至少一次真实LoRA更新，保存状态可核对，最新保存版本完成再次采样，全部轨迹/优势/身份审计通过。
- 最后的刷新采样独立登记，计入生成token、动作和耗时。
- 终端显示总阶段、rollout、评分/反向和重放进度条，全部输出同步进入日志。
- 失败或超时同样回传review ZIP，包含status、日志、部分轨迹和已完成adapter。state.pt保留云端，回传哈希与核验记录。
- 本轮只运行预检，C2首批三个正式作业及12 GPU小时额度仍保留。

云端基座沿用 .cache/qwen2.5-1.5b-instruct；prepare核对完整权重及tokenizer哈希，采集当次GPU/软件版本，并在作业内保持同一身份。发布包附带W，约4.49MB。

## 上传与运行

将 grpo-outcome-preflight-v1-20261105-run.zip 上传至云端 v2 目录，执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-preflight-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo.py
~~~

运行结束后返回：

~~~text
/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-preflight-v1-20261105-review.zip
~~~

已完成prepare并有已提交组的作业，可使用相同命令追加 --resume，继续使用该作业的剩余时限。恢复回传文件名为 grpo-outcome-preflight-v1-20261105-review-attempt-002.zip 等，终端会打印准确路径。

发布ZIP SHA256：b296441f2729b0c48edb86f83ac24bdc803313a5e1ba40a9689ddf6a5a0061eb。

## 后续推进

收到review ZIP后，按包内冻结源码本地复核，给出GPU profile和有效组统计；据此选择正式组数、评测节点与时限，再执行同轮W/GRPO完整任务及完整恢复配对评测。

性能目标保持为至少一条GRPO分支超过W。提升幅度和回归容差继续按用户决定保持待定。已有dev和历史test用于开发/回归，最终确认任务独立冻结。细粒度分支及真实依赖扩展在C2闭环稳定后推进。
