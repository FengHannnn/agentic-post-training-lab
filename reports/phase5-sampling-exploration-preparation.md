# Phase 5：受控采样探索准备

本轮执行选项5：检查严格正样本是否可以通过增加采样数量、提高温度被发现。固定累计31步的GRPO候选，使用训练集24题，温度1.0、1.3、1.6各题各采32条，共2304条。严格奖励和模型权重保持不变，优化器更新为0。仍处于Phase 5。

截至本报告，本地输入预检通过，新增13项测试通过，项目`python -m pytest tests -q`共365项通过，保留1条已有的tiny PEFT配置警告；云端采样尚未执行，不能提前声称高温有效或GRPO效果提升。测试限定在项目`tests/`，避免将冻结快照内的历史测试副本一同收集。

## 为什么做这个实验

上一轮诊断发现：两批历史采样都没有严格正样本的题目集中在自然语言和顺序表达模板。如果同组全部为0，组内标准化后的advantage为0，继续使用这一组无法提供当前GRPO的学习信号。本轮先回答“更宽的采样能否产生有用的混合奖励组”，再决定是否值得更改正式采样方案。

两批历史采样分别来自SFT和首轮16步候选，不能当作同一个分布下的16次独立抽样。本轮使用新的31步候选，并重新采一组温度1.0对照；历史全零只是选题依据，不能预设当前模型仍然全零。

## 固定题目与比较方法

从训练集选四个算术family，每个family包含六种问法/结构组合：

| 分组 | 模板 | 题数 |
| --- | --- | ---: |
| 主要观察组 | natural/left_plain、natural/right_plain、sequence/left_plain、sequence/right_grouped | 16 |
| 近邻对照 | natural/right_grouped | 4 |
| 简单对照 | symbolic/right_plain | 4 |

四个主要模板在两批历史记录都全零的共同family有11个。按family ID排序，取索引`round(i*(N-1)/3)`，i为0至3，得到`v2-arithmetic-12-47-20`、`v2-arithmetic-16-63-32`、`v2-arithmetic-20-79-44`、`v2-arithmetic-23-91-53`。24个case ID在运行前冻结在计划文件中。开发集题目不参与采样。

温度比较：同一题在1.0、1.3、1.6下各采32条，比较相同样本预算。样本数量比较：使用同一温度同一批32条的前8、16、32条，不额外生成一批来替换样本。它们是嵌套前缀，不能当成三个独立实验。

同一case和sample index在三个温度共享seed；每组seed独立于历史采样。每8条为一个区块，轮换温度执行次序。共享seed方便对照，但不代表三个温度会产生相同回答，也不保证不同运行的GPU结果逐位相同。36次引擎调用，每次64个请求，实际并发最多4条。

固定原有vLLM 0.11.0、L20、FP16、512上下文、最多64个回答token、无top-k/top-p截断、无惩罚、无prefix cache。所有无效JSON及达到长度上限的回答保留，严格奖励继续按原契约计分，长度截断奖励为0。

候选：`artifacts/checkpoints/grpo-iteration-20261028-01`，完整参数摘要`70804ab1c9344f7578396e4ccaa90dfe1ea374c69f96cb69ee3970d036eedf7d`。运行前重建冻结的FP32模型并验证完整参数摘要；随后退出该进程，再由vLLM加载同一个adapter采样。

## 概率记录的边界

沿用已有的`raw_logprobs`路径。vLLM的此模式返回温度及其他logits处理之前的模型logp；`processed_logprobs`才涉及处理后的分布，见[vLLM 0.11.0官方模型配置源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/config/model.py)。

温度1.0且无其他过滤或惩罚时，原始模型logp可记录为实际采样的behavior logp。温度1.3、1.6的原始logp保存在`sampler_raw_model_token_log_probs`，但`behavior_token_log_probs`明确为null。不能简单把raw logp除以温度当作归一化后的behavior logp，因为还缺完整词表的归一化项。

本轮使用独立探索记录格式，`training_admitted=false`；不计算TIS、不接正式trainer、不更新Adam或reference。将来若决定采用高温训练，需要另行实现和验证真实behavior概率及采样修正，当前文件不能直接作为训练就绪数据。

## 如何读结果并决定下一步

结果`summary.json`按温度、样本前缀、主要观察组/对照组及六种模板分别汇总：严格成功回答数、至少有一条严格正样本的题数、混合奖励组数、每1000条样本的严格成功数、语义正确但严格失败数、无效JSON、截断数。混合组中的回答数只表示“如果按该大小组队时可有非零advantage”，不代表已训练或已获得性能收益。

优先看以下三个问题：

1. 温度1.0从8条增到32条，16道主要观察题中新覆盖了哪些题？如果有新增混合组，说明采样数量可能限制了信号，但要一起看单位样本的正样本密度和成本。
2. 相同32条预算下，1.3或1.6比1.0新增覆盖哪些题、失去哪些题？报告逐题列出new/lost，避免只看总数掩盖退步；重点看主要观察组，而不是被简单对照的高成功率主导。
3. 新增混合组是否伴随更多无效JSON或长度截断？若高温只增加噪声，或者仍然采不到严格正样本，就不应直接扩大高温正式训练。

这是有针对性的24题探索，4个family之间也存在关联，不能据此宣称开发集提升或推广到全部训练分布。不自动选温度、不自动训练、不晋升基线。取回结果后，我们共同决定继续扩大验证、尝试新的训练采样配置，还是转向其他原因排查。

## 云端执行

增量包为`grpo-sampling-exploration-20261028.zip`。它依赖云端原项目和累计31步候选仍然保留；包内manifest会核对已有源文件及新增文件，输入预检还会核对基线、模型缓存、历史记录和候选文件。已有文件不符时停止，不覆盖历史结果。没有新增src或顶层configs文件，从而避免改变旧训练身份扫描范围。

上传ZIP到`/root/rivermind-data`后，在云端终端执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e grpo-sampling-exploration-20261028.zip .
cd AgenticPostTrainingLab
python scripts/explore_grpo_sampling.py --hourly-price 2.18
```

最后一项是实际GPU小时价格，若当前价格不同应修改。仅运行一次。有限阶段按“30分钟与10元对应时长的较小值，再预留10秒清理”设超时，2.18元/小时对应阶段预算1790秒。该预算不包含启动、预检、打包、闲置、存储或平台计费取整；脚本不会自动关闭实例。运行结束后请停止实例，避免闲置计费。

成功或进入有限阶段后失败都会保留证据，项目根目录生成`grpo-exploration-review-20261028.zip`。下载并上传该结果包给当前对话。若在输入预检阶段停止，终端错误即是需要核查的输入问题，此时尚未启动采样，可能没有结果包。禁止盲目删目录或自动重试；已有输出会阻止重复运行。

主要输出在`artifacts/trajectories/grpo-exploration-20261028-01`：`run_manifest.json`、`source/`、`model_check.json`、`rollouts.jsonl`、`generation_checks.json`、`summary.json`、`cost_and_run.json`。失败时仅包含已经实际产生的文件，不能把不完整输出当作成功。

## 代码阅读顺序

1. `configs/experiments/grpo_sampling_exploration.json`：先读三个温度、8/16/32前缀、候选摘要与24个case ID，理解本轮固定边界。
2. `scripts/explore_grpo_sampling.py::select_cases`：看如何用两批历史全零组选择共同family，再加入两类对照；`setup`会重新算选题并与冻结计划比较。
3. `seed_for`和`requests`：看同题同序号如何共享seed，以及每8条轮换温度次序；`sampling`固定除温度和seed以外的采样设置。
4. `native_record`和`validate_records`：重点看原生token/logp对齐、EOS/长度、mask、重新解码与重新计算奖励，以及高温behavior logp必须为空。
5. `summarize`：看前缀比较、分层统计、严格正样本密度、new/lost题目；不要把潜在混合组数当成训练效果。
6. `model_check`、`generate`、`audit`、`run`：看冻结参数验证、vLLM原生采样、全量记录复算、阶段超时、失败保留及单次执行限制。
7. `tests/unit/test_grpo_sampling_exploration.py`：13项测试覆盖完整2304请求、选题和隔离、raw/behavior区别、完整记录篡改、前缀与温度统计、前序失败阻止后续采样并打包证据。

本地预检最大prompt为64个token，所有题目的prompt+64均未超过512上下文；新实验未在本地执行模型前向、反向或GPU采样。实际vLLM运行与采样结果仍需云端验证。
