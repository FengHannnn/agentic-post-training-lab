# 恢复DPO增加训练量：三遍训练与实时进度

日期：2026-11-02。运行ID：dpo-recovery-train-v2-20261102。

用户已选择增加训练量，并要求提高运行时限和终端训练进度显示。本轮将相同169对数据从同一个DPO11检查点训练3遍，保持学习率、beta、数据和评测seed一致，便于与上一轮1遍结果比较。

## 训练与时限

| 项目 | 上一轮v1 | 本轮v2 |
|---|---:|---:|
| 数据遍历 | 1遍 | 3遍 |
| 独特偏好对 | 169 | 169 |
| 样本曝光次数 | 169 | 507 |
| 梯度累积 | 4 | 4 |
| 优化器更新 | 43 | 129 |
| 含初始DPO11的累计DPO更新 | 54 | 140 |
| 全流程时限 | 5400秒（90分钟） | 10800秒（180分钟） |

本轮初始和固定参考策略均为dpo-train-v1-20261102/final。新的AdamW在三遍之间持续使用，参考分数在首次更新前缓存，三遍中保持固定。学习率5e-6，beta=0.1，梯度裁剪1。每遍按seed+epoch_index重新打乱，第一遍顺序与上一轮相同。每遍最后一批1对，共43次更新。

检查点位于第43步（第1遍完成）、86步（第2遍完成）和final第129步。上一轮v1的结果目录保留，本轮使用新的run ID。最终模型及每遍检查点保存在云端；回传ZIP包含最终适配器及全部元数据，省略中间权重和所有.pt优化器文件。

固定数据SHA256：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。固定初始/参考适配器SHA256：fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d。

## 终端进度与日志

新增train_v2_dpo_recovery_long.py，用tqdm显示参考分数缓存、训练前评分、训练更新和训练后评分进度。训练条显示已完成步数/129、epoch/3、loss、样本曝光/507以及耗时、预计剩余时间。每遍完成会显示单独的完成消息。

示意：

```text
DPO training: 33%|...| 43/129 [..., epoch=1/3, loss=0.6000, pairs=169/507]
Epoch 1/3 completed; step 43/129; pairs 169
```

总入口run_v2_dpo_recovery_train_long.py通过streamed_process.run_logged()将子进程stdout/stderr同时传到终端与阶段日志，保留回车刷新字符及UTF-8。子进程启用无缓冲输出。每步结构化指标另存training_steps.jsonl，供审计核对。

日志位于artifacts/dpo-recovery-train-v2-20261102下的train.log、greedy.log、sample.log、recovery.log、audit.log。超时会终止阶段进程并保留输出，最终仍会打包review ZIP。

## 评测和审计

沿用上一轮配置：before=DPO11、after=三遍恢复训练模型；完整贪心train12/development4两模型共32条轨迹，开发随机两模型共32条轨迹，开发恢复两模型共224个动作。总计64条完整轨迹和224个恢复动作。重点比较四类此前0/12的恢复场景、重复错误调用、格式有效率与完整任务成功率。独立test集留在后续评测。

审计新增三遍更新历史、129次Adam更新、507次样本曝光、43/86步检查点和training_steps.jsonl一致性检查。其余数据身份、固定参考、原生token和环境重放核对保持相同标准。

## 本地验证

102项测试通过，Python编译检查通过。新增测试包括：真实小型Qwen+LoRA完成3遍129次更新；参考缓存和冻结参数保持不变；第一遍参数与单遍控制模型逐张量一致；实时输出在子进程结束前到达终端和日志；回车刷新及中文输出保留；超时终止、部分日志和失败打包。

本地检查配置为epochs=3、max_steps=129、pair_exposures=507、max_wall_seconds=10800。云端L20的训练和推理结果待执行回传。

## 云端命令

上传v2-dpo-recovery-train-v2-20261102.zip到/root/rivermind-data/后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
unzip -o v2-dpo-recovery-train-v2-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python -u scripts/run_v2_dpo_recovery_train_long.py
```

运行阶段顺序：train → greedy → sample → recovery → audit。终端将直接显示各阶段输出及训练进度条。

结束后回传：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-recovery-train-v2-20261102-review.zip。