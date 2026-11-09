# 恢复专项 DPO 训练说明

日期：2026-11-02。运行 ID：dpo-recovery-train-v1-20261102。

从已复核的 11 步 DPO 检查点继续训练，使用审核通过的 169 对偏好，完成一次数据遍历及训练前后配对评测。云端训练结果待运行后回传复核。

## 训练设置

| 项目 | 设置 |
|---|---|
| 初始模型 | dpo-train-v1-20261102/final，已完成 11 次 DPO 更新 |
| 本轮固定参考策略 | 同一初始 DPO11；训练前缓存 169 对 chosen/rejected token 分数 |
| 数据 | 169 对：原 SFT 模型偏好44 + 规则恢复84 + DPO采样恢复41 |
| 编码入口 | encode_recovery_pair()，兼容三种来源 |
| 遍历轮数 | 1 |
| 梯度累积 | 4 对/次，最后一批 1 对 |
| 本轮更新 | 43 次，累计 DPO 参数更新54次 |
| 优化器 | 新 AdamW，weight_decay=0 |
| 学习率 / beta | 5e-6 / 0.1 |
| 梯度裁剪 | 1 |
| 可训练参数 | q_proj/v_proj LoRA，共1,089,536参数 |
| 检查点 | 第16、32步及最终43步 |

本轮从 DPO11 初始化，参考策略也固定为 DPO11。优化器计数从 0 开始，结束时为 43；SFT126 与前一轮 DPO11 的优化器状态不加载。共同 SFT 基线和原 DPO11 检查点保留。

DPO loss 使用当前策略与固定参考策略对 chosen/rejected 的序列 log probability 之差。参考输入按原生 token 指纹绑定，更新前初始 loss 应约等于 log(2)。

最终数据 SHA256：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。初始/参考适配器 SHA256：fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d。

## 前后配对评测

比较分支明确命名 before（DPO11）与 after（恢复续训完成）。两分支使用相同推理设置和逐任务/逐采样 seed。

| 评测 | 每模型数量 | 两模型合计 |
|---|---:|---:|
| 完整任务贪心：train12 + development4 | 16 条轨迹 | 32 条轨迹 |
| 完整任务随机：development4，每任务4次 | 16 条轨迹 | 32 条轨迹 |
| 恢复上下文贪心：development4 × 7类 | 28 个动作 | 56 个动作 |
| 恢复上下文随机：28上下文，每个3次 | 84 个动作 | 168 个动作 |

合计 64 条完整任务轨迹，以及 224 个恢复动作。开发集恢复上下文来自4个 development任务，共28个上下文；其 task/family 与训练数据分离。恢复上下文评测判断下一步动作是否带来有效进展；完整任务评测记录最终成功和故障恢复。

重点指标：未验证 SQL 后重新查询、重复失败动作次数、严格格式动作比例、证据恢复、逐场景有效进展、完整任务成功率及配对改善/退化。完整随机评测 temperature=0.8、seed=2026110252；恢复评测 temperature=0.8、seed=2026110253。训练 seed=2026110251。

本轮保持独立 test 集用于后续独立评测；训练与本轮评测只使用 train/development。

## 实现与审计

新增 scripts/train_v2_dpo_recovery.py、evaluate_v2_dpo_recovery.py、audit_v2_dpo_recovery.py、run_v2_dpo_recovery_train.py 和 src/apt_v2/dpo_recovery_training.py。既有评测器增加可选检查点映射，既有恢复上下文函数增加显式 allowed_split 参数；原采集流程默认保持 train。

审计覆盖169对数据来源与 token 编码、固定参考分数、训练前后损失公式、43次 fresh Adam 更新、冻结参数梯度、初始检查点和输入指纹，以及完整任务轨迹与开发集恢复动作的原生 token/环境结果重放。最终优化器状态留在云端，审计其文件指纹与每参数step计数。

回传 ZIP 包含最终适配器、配置、源码快照、参考分数、训练指标、完整评测和审计。中间检查点权重及所有 .pt 文件保留在云端，回传包省略这些文件。发生错误时仍保存日志并打包。

## 本地验证

98项测试通过，Python编译检查通过。小型真实Qwen+LoRA完成169对数据、43次更新，固定参考分数不变、冻结参数不变、偏好margin提高。完整审计测试重放64条任务轨迹与224个恢复动作，并检测seed和记录篡改。

真实169对数据重新编码通过，最长1376token；28个开发集上下文构建通过，训练与开发family分离。云端L20训练及推理结果待回传复核。

## 执行命令

上传 v2-dpo-recovery-train-v1-20261102.zip 至 /root/rivermind-data/ 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
unzip -o v2-dpo-recovery-train-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_dpo_recovery_train.py
```

运行按 train → greedy → sample → recovery → audit 顺序执行，总时间预算5400秒。回传：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-recovery-train-v1-20261102-review.zip。