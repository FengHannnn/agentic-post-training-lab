# V2 SFT 执行诊断 v1

日期：2026-11-01（Asia/Shanghai）。当前工作：检查首轮 SFT 的训练任务掌握情况与训练/推理接入一致性。

## 已复现的失败

本地执行 diagnostics/work/reproduce_sft_sql.py，重放 development-000 的前三个动作。规则/schema 读取正常，首条月度 SQL 返回 verified_metrics=[]，断言以 reproduced: first monthly query fails business verification 失败。这一固定样例用于持续检查业务 SQL 问题。

## 本轮诊断

按顺序检验三项假设：复杂动作尚未学牢、推理接入存在差异、模型偏离专家上下文后难以恢复。保持原训练数据、权重与开发集结果，全部阶段只做推理和评分，optimizer_updates=0。

1. 对 252 个固定训练样本，在 HF FP32 中分别关闭/启用 adapter，统计每工具、每变体的 token 平均 NLL、动作平均 NLL、token 准确率，以及 teacher-forced 全 token 预测正确的动作数。
2. 选择 8 个固定训练上下文：六种工具各一个，加 SQL 错误修复和证据错误修复各一个。固定同一份 prompt/正确 response token，在 HF FP32、HF BF16、vLLM BF16 中分别关闭/启用 adapter，记录正确回答的逐 token logprob、argmax 和自由 greedy 输出。
3. vLLM 中分别用原始模型和本轮 SFT adapter 执行全部 12 个训练任务，保存 24 条完整轨迹，统计成功、恢复、有效 SQL、验证进展、格式、错误和 token 数。
4. 重放 24 条轨迹，核对 token 输入、解码、工具观察和奖励，生成后端差异及 adapter 影响对照。

teacher-forced 评分采用正确历史和正确回答前缀；自由执行使用模型自己的动作形成后续上下文。两项分开报告。跨后端输出差异按原始数值与 token 报告，结合 FP32/BF16 对照定位。

## 输入与预算

- 输入：artifacts/sft-pilot-v1-20261030/ 中的原始训练证据及 final adapter。
- 固定 adapter SHA-256：95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e。
- 固定 Qwen2.5-1.5B revision：989aa7980e4cf806f80c7fef2b1adb7bc71aa306。
- 沿用现有模型缓存、4090D 环境；各后端在独立子进程中运行。
- 上下文 8192，每动作生成上限 256，greedy seed=2026102901。
- 总预算 3600 秒，预留 30 秒，超时停止并打包已有证据。
- 输出独占创建；阶段失败停止后续阶段，保留日志和 review ZIP。

## 云端命令

将 v2-sft-diagnostics-v1-20261101.zip 上传至 /root/rivermind-data/，执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
ls -lh /root/rivermind-data/v2-sft-diagnostics-v1-20261101.zip
unzip -o /root/rivermind-data/v2-sft-diagnostics-v1-20261101.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_sft_diagnostics.py
```

终端逐阶段打印日志路径。五个阶段依次为 prepare、hf-fp32、hf-bf16、vllm、audit。查看当前阶段日志，例如：

```bash
tail -f /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/sft-diagnostics-v1-20261101/hf-fp32.log
```

回传：/root/rivermind-data/AgenticPostTrainingLab/v2/sft-diagnostics-v1-20261101-review.zip。

如原始 pilot 目录缺失，入口会显示缺失文件的完整路径；诊断依赖原运行留下的 examples、model_identity、run_manifest 和 final/adapter。

## 本地检查与文件位置

V2 全部测试 58 passed。新增检查覆盖：8 个固定上下文及真实错误反馈、token/动作两种均值、跨后端比较的长度校验、微型真实 Qwen + PEFT 上诊断评分与 SFT loss 一致、adapter 开关影响、专家轨迹汇总、失败阶段日志/ZIP 保存及已有结果保护。源码编译通过。

实现：src/apt_v2/sft_diagnostics.py；diagnostics/diagnose_sft_v1.py；scripts/run_v2_sft_diagnostics.py。配置：configs/sft-diagnostics-v1.json。测试：tests/test_sft_diagnostics_v2.py。发布包与凭据：artifacts/releases/sft-diagnostics-v1-20261101/。

下一轮 SFT 的数据与训练配置将依据训练任务执行结果、逐工具评分和后端对照确定。
