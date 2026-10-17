# Phase 4 第一步：CPU 随机采样与 rollout 数据接口

## 本步完成了什么

Phase 3有限对照已收尾，本步开始准备rollout引擎与后续在线采样。已用冻结v2 SFT完成4题×4回答的随机采样，保存原始token、两类逐token概率、评分和响应掩码，验证完整前向重算及一次采样复现。模型参数未更新，没有新训练。

本机核验：Windows 10、Python3.11.5、torch2.14.0+cpu、CUDA不可用，vLLM/SGLang未安装。当前使用Transformers CPU引擎，不能称已接入vLLM、GPU或GRPO trainer。GPU引擎和资源方案仍待推进。

vLLM官方GPU安装路径以Linux为要求，并说明不原生支持Windows；未来需要独立规划环境，不向当前CPU虚拟环境直接安装引擎依赖。[官方环境要求](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)（查阅日期：2026-10-17，实际接入时还须锁定版本与兼容组合。）

## 与后续训练的连接

当前流程：

```text
4个偏好训练输入
  → 冻结SFT、每题4个独立种子的随机回答
  → 原生response token IDs + policy/behavior逐token logp
  → 既有JSON/路由/算式/执行评分
  → [prompt, generation, response_token]打包与mask
  → 保存、复现探针和不加载权重的记录审计
```

这是单轮模型回答采样，不是工具结果回传后的多轮Agent轨迹。奖励定义、advantage、old/new/reference策略关系及GRPO参数更新仍未接入。已有评分用于观察，不在本步将其冻结为训练reward。

## 概率与token对齐

`policy_token_log_probs`来自未处理的模型logits；`behavior_token_log_probs`来自生成时处理过的scores。温度和截断可能改变实际采样分布，不能混用字段。HF接口明确区分两类输出；本机集成测试验证了温度0.8时两者不同。[HF生成输出说明](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils)

本次固定temperature=1、top_p=1、top_k=0、repetition_penalty=1、不继承模型隐含生成默认参数。因采样分布不做修改，两种logp的实际最大差为0。后续引擎需要自己验证其logprobs字段含义，不能因字段名相同就假定一致。

每个采样保存prompt_ids与实际发出的response_ids，含真实EOS；不将解码文本重新tokenize后冒充原采样轨迹。把两段ID直接拼接做完整前向，调用已有因果shift函数，取`len(prompt_ids)-1`之后的token logp，与逐步生成分数对照。实际最大误差为4.1962e-5，在预设atol=rtol=1e-4内；这里是float32/KV增量生成与完整前向的数值一致性检查，不要求bitwise相等。

打包的概率已经与response token逐项对应，不需要再次shift。`response_batch.pt`只打包回答区域，不是可以直接喂给HF模型的完整训练batch；完整上下文仍在各条prompt_ids中。

EOS和PAD在本模型中同为0，mask按长度建立。例如`[token, EOS, PAD]`对应`[1,1,0]`，不能写成`token_id != 0`。实际400个响应token中保留16个真实EOS，另外256个padding位置被排除。

## 固定配置与实际结果

配置为`configs/rollout_cpu_smoke.json`，从既有v2偏好train中选符号题、自然语言题、顺序题、问候各1条。三条算术共享数字family，且包含已知易错题；不从开发集采样，不是新性能测试集。每题种子为`42 + group_index*1000 + sample_index`，每次只生成一条，以便先把token/概率边界核对清楚。

| 输入类型 | 回答数 | 不同文本数 | 严格通过 | 语义通过 |
| --- | --- | --- | --- | --- |
| 符号算式 | 4 | 1 | 4 | 4 |
| 自然语言 | 4 | 3 | 0 | 0 |
| 顺序描述 | 4 | 3 | 0 | 1 |
| 问候 | 4 | 4 | 4 | 4 |

总16条，严格8、语义9、非法JSON4，全部正常发出EOS，无长度截断。错误、重复、空回答或长度截断均应原样记录，本次只是没有出现后两类。`response_batch`形状为`[4,4,41]`，400个有效token。采样加前向重算共约57.75秒，另做1次首样本复现探针，共17次生成；这个时间不是纯解码吞吐，也不能与Phase 3贪心成绩或GPU引擎吞吐直接比较。

问候通过仅检查非空final回答的结构与路由，不评判回答内容质量。

模型初末参数SHA均为`9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb`，无梯度、无更新。首条采样在同模型/设备/版本/种子下重现，不保证跨设备、版本或引擎逐token相同。

## 学习信号诊断

4组的严格0/1结果都恒定：两组全对、两组全错。如果直接用严格通过作为二元reward，现有组内标准化会给这4组零优势，不能据此启动有效的相对策略更新。自然语言/问候的文本多样性也不代表reward多样性。

语义评分仅顺序题有1/4通过：实际输出`20*12+47`，目标为`47+20*12`，数值正确但AST次序不同。它说明严格与语义目标会影响学习信号；后续必须先明确奖励目标并审计，不能只为制造非零方差而改分。本步没有新增训练reward、修改评分器或换温度重采样。

## 阅读文件与产物

1. `rollouts/records.py`：`RolloutSample`保存原生ID和概率，检查长度、有限值和EOS；`pack_rollout_groups`按真实长度打包、保留EOS并屏蔽padding。
2. `rollouts/hf_sampling.py`：显式采样配置、独立CPU种子、两类分数提取；`replay_policy_log_probs`完整前向重算，验证因果对齐。
3. `scripts/collect_rollout_smoke.py`：选择固定训练输入、冻结模型、采集16条、一次重现探针、保存与审计；不更新权重。
4. `test_rollout_records.py`、`integration/test_rollout_sampling.py`：新增12项测试，覆盖EOS-as-PAD、概率错位/非有限、组内输入、真实tiny模型采样与重算、概率分布差异和CPU RNG/权重保持。全套137项通过，1条已有tiny PEFT提示，无skip。

产物目录：`artifacts/trajectories/rollout-smoke-phase4-20261017-01`。

- `run_manifest.json`：配置、模型/adapter/数据/源码哈希、版本、采样概率语义。
- `rollouts.jsonl`：16条回答、原始ID/文本、概率、完整目标与既有规则评分。
- `response_batch.pt`：回答ID、mask、policy/behavior logp四个张量。
- `run_checks.json`：模型初末身份、首样本复现探针。
- `summary.json`与`verification.json`：分组观察和重算审计结果。

```powershell
& ".\.venv\Scripts\python.exe" scripts\collect_rollout_smoke.py --output artifacts\trajectories\rollout-smoke-phase4-20261017-01 --audit-only
```

该命令已运行通过，不加载模型权重，重建token解码、规则评分、分组与mask/张量并核对文件身份。逐token policy概率的完整前向验证发生在采集时；无模型审计不会伪称重新运行了这些前向。不要为阅读代码再运行默认采集命令。

## GPU引擎接入前的具体工作

下一子步是资源与引擎方案，不直接租机：先确认是否已有可用Linux/NVIDIA GPU；如需云端，明确总预算、时限、停止/释放规则，再选择实例。当前未选供应商或价格，未启用付费资源。

接入验收应固定：独立Linux/CUDA环境与引擎版本、base revision与LoRA身份、tokenizer/chat template、EOS和最大长度、随机采样参数、raw/processed logp含义。先验证所选模型/adapter受到引擎支持，再用少量相同输入检查token、结束原因与训练侧前向分数对齐，最后才扩大批量并分别记录解码时间、评分时间、显存和成本。本机CPU数据接口通过不等于GPU/vLLM接口已经通过。
