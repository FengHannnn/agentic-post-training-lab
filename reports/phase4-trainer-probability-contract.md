# Phase 4：训练侧概率契约与无更新 ratio 检查

日期：2026-10-20。当前仍属于 Phase 4，真实 GRPO 更新尚未开始。本次新增实现与 CPU 验证已完成，L40S 实测待执行。

## 为什么进入这一小步

上一轮证明 FP16 下 HF 完整前向与缓存路径本身也有差异，不能把跨引擎误差全部归因于 vLLM。原 16 条 rollout 的 0.001 数值门仍失败，不修改原阈值或原产物。现在先建立独立的训练评分契约，验证没有参数更新时，训练侧的 old/new 概率是否一致。

这一步使用全部已有 16 条回答的原始 token，不重新生成。它不增加样本量，也不解决三个算术组奖励全部零方差的问题。

## 三种概率各自代表什么

| 分数 | 来源 | 本步用途 |
| --- | --- | --- |
| sampler behavior logq | 原 vLLM FP16 随机采样时保存 | 保留真实采样路径，计算与训练评分的差异 |
| trainer old logp | HF FP32，关闭梯度评分 | 作为训练侧 old/new 比值的分母 |
| trainer new logp | 同一个 HF FP32 模型，启用梯度评分 | 检查比值及其计算图 |

逐 token 比值为 exp(new_logp - old_logp)。模型没有更新，预先固定最大绝对 logp 差异容许值 1e-5，并记录 ratio 偏离 1 的最大值。该新门只检查训练侧内部一致性，不替代旧跨引擎门。

训练路径固定为原始 FP32 checkpoint + FP32 LoRA、eager attention、eval 模式、dropout=0、use_cache=False、完整序列逐条评分、禁用 TF32。old 与 new 的输入 token、形状、回答范围完全相同；回答评分包含 EOS，padding 按长度屏蔽。只有 LoRA 参数可训练，共 230400 个。

这里重新加载原始 FP32 权重，并验证完整参数摘要为 9b77c99e51b68d12dc0249ccfaf9f642e7c049ecf99c7f7c53d2c0b83d72cadb。上一轮诊断的 model.float() 保留了 FP16 舍入，此次与其权重表示不同。因此新路径的 sampler/trainer 差异要重新记录，不能沿用上次 FP32 compute 的结论。

即使本步通过，样本依然来自采样器 q；训练侧重算 old logp 不会自动把采样分布变为 p。本步只记录 log(p/q)，没有应用重要性校正、裁剪或损失函数。training_admitted 固定为 false。旧模型版本与未来 KL reference 的角色也不能因为初始化权重相同而混为一谈。

## 实现与输出

入口先重审旧产物，绑定原 rollout manifest 与已完成诊断的 SHA256，核验原固定依赖及 L40S 型号，再加载模型。每条记录分别计算无梯度 old 和保留计算图的 new，记录比值、采样器差异、奖励和 advantage。每条评分后释放计算图。

云端脚本不执行 backward，不创建优化器，不更新参数。前后参数摘要必须相同，所有 parameter.grad 必须为空。tiny 单元测试中的一次 backward 仅用来证明梯度确实可达 LoRA；不发生在真实云端模型检查中。

输出目录 artifacts/evaluations/trainer-probability-contract-v1，已有目录则拒绝覆盖：

- summary.json：逐条分数、ratio、采样器差异、参数身份、输入/源码哈希及检查结论。
- trainer_score_batch.pt：独立保存采样器 logq、训练侧 old/new logp、ratio、原始 token 与 mask。保存的张量是脱离计算图的审计产物，不能直接拿来替代未来训练时需要梯度的 new 分数。

eligible_response_mask 额外排除 greeting；response_mask 仍覆盖全部真实回答 token。padding 位置的占位值只可结合 mask 使用。

## 云端执行

上传 phase4-trainer-probability-update-20261020.zip 到 /root/rivermind-data，使用保存原 rollout 和 diagnostic.json 的同一 L40S 实例：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase4-trainer-probability-update-20261020.zip .
cd AgenticPostTrainingLab
python scripts/check_trainer_probability_contract.py
```

无需安装依赖或重新生成 rollout。先复制终端最后的摘要，再下载 summary.json 供本地复核。若门失败，保存失败结果，不放宽阈值或删除旧数据重跑追逐通过。

## 代码阅读顺序

1. configs/trainer_probability_contract.json：固定训练评分路径、参数身份、输入哈希与新门阈值。
2. src/agentic_post_training/rollouts/trainer_scores.py → score_response：相同原始 token 的 teacher-forced 完整前向；重点看回答起点 len(prompt_ids)-1 与梯度开关。
3. 同文件 no_update_metrics：old 不带梯度、new 带梯度，exp(new-old) 的意义；sampler_trainer_difference：为什么 log(p/q) 另行记录。
4. 同文件 pack_trainer_scores：样本索引对齐、EOS/padding mask、greeting 排除及分数分别保存。
5. scripts/check_trainer_probability_contract.py → main：原产物审计、原始 FP32 参数摘要、两次评分、释放图和前后参数不变检查。
6. tests/unit/test_trainer_probability_contract.py：优先阅读 test_same_path_ratio_one_and_gradients_only_reach_lora 与 test_sampler_difference_is_not_a_no_update_policy_ratio，再看打包和非法输入测试。

## 验证状态

新增 14 项 CPU 单元测试通过；按项目入口 python -m pytest tests -q 全套 232 项通过，无 skip，1 条已有 tiny PEFT 配置提示。最初不指定 tests 目录的命令误收集了冻结基线里的历史测试副本，造成模块名冲突；限定实际测试目录后完整通过，未改动历史快照。真实模型在 CPU 上只进行了原始 FP32 参数身份核验，摘要和 230400 个可训练参数均匹配，未执行真实模型前向。云端实际 no-update ratio 尚无结果。

下一步必须先复核云端结果，再决定采样分布校正与有上限的奖励信号诊断；本步通过也不代表可以直接启动 GRPO。
