# Chosen-only SFT 恢复对照说明

日期：2026-11-02。运行ID：chosen-only-sft-v1-20261102。

用户已选择chosen-only SFT对照。本轮从同一个DPO11初始化，对原169对偏好中的messages/chosen进行直接监督，3遍、129次更新，保持学习率和评测设置，比较恢复动作学习效果。云端训练结果待执行回传。

## 控制变量与目标

| 项目 | 三遍恢复DPO | 本轮chosen-only SFT |
|---|---|---|
| 初始化 | DPO11 | 同一DPO11 |
| 数据行 | 169对偏好 | 同169对投影的169条chosen样本 |
| 更新 | 3遍、129步 | 3遍、129步 |
| 学习率 | 5e-6 | 5e-6 |
| 批量/尾批 | 4/1 | 4/1 |
| 优化器 | fresh AdamW，weight_decay=0 | 相同 |
| loss | 当前/参考的chosen-rejected序列差 | chosen响应的token平均交叉熵 |
| rejected/参考参与loss | 是 | 否 |
| 训练与评测seed | 固定原seed | 相同 |

SFT的输入包含原上下文，历史错误动作及环境反馈继续作为prompt的一部分；仅当前chosen响应和EOS计算交叉熵。每条响应内部按token平均，再在批量中按样本平均，直接复用既有action_loss()。prompt标签全部为-100。

169行包含123个不同的messages/chosen组合。本轮保留重复行及其source_pair_id，维持原偏好数据中的上下文权重。每遍监督17,581个token，三遍52,743个，最长完整输入1376token。

样本语义SHA256：f7c4ab0a82cd019f7287e0b5ed8ec5dda8a8dc5e4d2e1ff4d9896f8af7ef8c0d。编码SHA256：7eb17e79c4a0a6bb72385282277df8e23edf0a78838ab332e7fb998f67f7848f。原pairs.jsonl SHA256：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。

## 初始与对照检查点

before：dpo-train-v1-20261102/final，DPO11。适配器SHA256：fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d。

dpo：dpo-recovery-train-v2-20261102/final，DPO140。适配器SHA256：7393906da901516cdfa05b3f987808a604c6f49c438a410a8d5e5dc06a9c0725。

after：本轮chosen-only SFT模型，从before独立初始化。新增129次SFT更新，保留初始DPO11来源。优化器为新建，三遍中持续更新；保存43、86步及最终129步检查点。

## 同次运行的三模型评测

在同一次运行中评测before、dpo、after三分支，以开发集完整任务和恢复上下文的共同指标比较。两轮DPO已有的跨运行随机采样差异作为历史记录保留，本轮直接生成三模型输出。

| 评测 | 每模型 | 三模型合计 |
|---|---:|---:|
| 完整任务贪心：train12/development4 | 16条 | 48条 |
| 开发完整任务随机：4任务×4次 | 16条 | 48条 |
| 开发恢复贪心：4任务×7场景 | 28个动作 | 84个动作 |
| 开发恢复随机：28上下文×3次 | 84个动作 | 252个动作 |

总计96条完整任务轨迹、336个恢复动作。评测seed和temperature沿用DPO三遍配置。报告会同时比较DPO11→SFT及DPO140→SFT的配对结果。优先观察SQL未验证后重新查询、连续证据/SQL错误、检查失败后修复这四类此前0/12的场景，以及格式、重复动作和完整任务成功。

vLLM评测为三个分支配置max_loras=3，符合本项目vLLM0.11.0的LoRAConfig约束。[vLLM 0.11.0 LoRAConfig源码](https://raw.githubusercontent.com/vllm-project/vllm/v0.11.0/vllm/config/lora.py)。

## 进度、时限与审计

沿用10800秒（180分钟）全流程时限，终端实时显示SFT步数/129、epoch/3、loss、样本曝光/507和预计剩余时间。输出同步保存阶段日志，逐步指标写入training_steps.jsonl。

审计核对原偏好身份、169条chosen投影、原生编码、prompt屏蔽和EOS、监督token数、129次新Adam更新及三个epoch。训练前chosen token分数会与已复核DPO三遍实验中的DPO11固定参考分数核对，用于确认相同起点；这些参考分数只参与审计。

三模型96条完整轨迹及336个恢复动作均执行原生token和环境重放，核对各模型指纹及输入未变。最终适配器包含在回传ZIP，中间权重和.pt优化器状态留在云端；失败时仍保存日志和打包。

## 本地验证

107项测试通过，Python编译检查通过。真实小型Qwen+LoRA完成129次SFT更新并降低交叉熵；改变rejected内容不改变训练输入及标签；prompt被完整屏蔽。三模型审计测试覆盖96条完整轨迹、336个恢复动作，并检测seed篡改；实时输出与失败打包沿用已验证入口。

真实169条数据已编码通过，每条prompt/chosen token与原DPO输入一致。预检数据位于artifacts/datasets/chosen-only-sft-v1-20261102。模型训练与真实L20推理待云端执行。

## 执行命令

上传v2-chosen-only-sft-v1-20261102.zip至/root/rivermind-data/：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
unzip -o v2-chosen-only-sft-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python -u scripts/run_v2_chosen_sft.py
```

阶段顺序：train → greedy → sample → recovery → audit。回传文件：/root/rivermind-data/AgenticPostTrainingLab/v2/chosen-only-sft-v1-20261102-review.zip。