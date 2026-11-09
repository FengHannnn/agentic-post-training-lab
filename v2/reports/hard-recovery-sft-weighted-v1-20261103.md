# 困难恢复定向加权 SFT：运行说明

日期：2026-11-03。
运行 ID：hard-recovery-sft-weighted-v1-20261103。

本轮从已审阅的 chosen-only SFT 检查点继续训练，重点加强缺少已验证指标时重查、错误 SQL 改写、检查失败后修正差额三个行为方向。延用 169 条 chosen 示范，四类困难场景相对其他样本按 4:1 加权，续训 6 轮、258 次更新。运行包已完成本地预检与 120 项测试；L20 训练由下面的命令启动。

## 训练设置

| 项目 | 设置 |
|---|---|
| 父检查点 | chosen-only-sft-v1-20261102/final |
| 父模型训练历史 | SFT255 次更新、DPO11 次更新 |
| 新增更新 | SFT258 次；完成后累计 SFT513 次、DPO11 次 |
| 数据 | 同一批 169 条偏好数据的 chosen，保留原始行顺序与重复行对应的采样权重 |
| 训练轮数 | 6，合计 1,014 次样本使用 |
| 有效 batch | 4，按每轮 43 个 batch 更新，尾 batch 为 1 |
| 学习率 | 5e-6，恒定 |
| 优化器 | 新建 AdamW，weight_decay=0；从父检查点载入 LoRA 权重 |
| 梯度裁剪 | 1 |
| 可训练参数 | q_proj／v_proj LoRA，r8、alpha16、dropout0；1,089,536 个参数 |
| 训练精度 | FP32，沿用现有训练实现 |
| 监督范围 | assistant response 与 EOS，prompt 标签为 -100 |
| 损失 | response token 平均 CE × 归一化场景权重，再对 batch 样本平均 |
| token 监督次数 | 每轮 17,581，6 轮合计 105,486 |
| 上下文 | 训练 max_length2048；真实最长完整示范1376 tokens |
| 时限 | 14,400 秒，4 小时 |

本轮只监督 chosen。rejected 和 reference 不进入损失。token 级权重保持原有方式。

父 adapter SHA256：
078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55

数据 pairs.jsonl SHA256：
c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329

## 权重分配

| 场景 | 行数 | 相对权重 |
|---|---:|---:|
| SQL 未验证后重查 | 24 | 4 |
| 重复证据错误后恢复 | 12 | 4 |
| 连续 SQL 错误后重查 | 24 | 4 |
| 检查失败后修复 | 24 | 4 |
| 原始 SQL chosen | 44 | 1 |
| 格式修复 | 16 | 1 |
| 未知工具修复 | 12 | 1 |
| 已有正确证据时修正引用 | 13 | 1 |

困难样本共 84 条，其余 85 条。原始权重均值为 421/169≈2.4911，除以该均值后：

- 困难样本权重约 1.6057。
- 其余样本权重约 0.4014。
- 全部样本平均权重为 1，困难样本合计占总样本权重约 79.8%。

仍使用每轮全量数据，不做额外重采样。原始 SQL 与容易恢复样本继续提供监督，完整任务评测跟踪正常流程表现。

## 检查点与评测

每轮结束保存模型及 Adam 状态。第 1–5 轮检查点分别为 step43、86、129、172、215，第 6 轮为 final258。

训练结束后统一进行以下评测：

1. 对五个中间检查点，在 84 个训练上下文上各做一次 greedy 恢复动作，共 420 次动作。逐检查点记录工具选择、证据 ID、SQL 验证与数值检查结果。
2. 对初始／最终模型，在 84 个训练、28 个开发上下文上各做 greedy1 次及 temperature=0.8 采样3 次，共 896 次动作。
3. 对初始／最终模型做完整任务评测：训练 greedy12、开发 greedy4、开发随机16，每模型32条，共64条完整轨迹。
4. 重放全部轨迹和恢复动作，复算数据权重、更新历史、分项统计及训练／开发差值。

恢复动作总数为 1,316。本轮固定使用第258步最终模型作为训练结果，检查点曲线用于观察恢复行为何时开始改变。

评测沿用 BF16 vLLM、max_new_tokens256、max_model_len8192、top_p1、top_k=-1、关闭 prefix cache。恢复探针沿用上轮种子公式：训练偏移1,000,000，开发偏移0；前后模型使用相同种子。

主要验收指标：

- 缺验证时是否选择 query_sql，以及目标月度指标是否通过验证。
- SQL 报错后是否生成可验证的源表查询。
- check_result 失败后是否根据正确证据修改错误差额。
- 正确证据已有时的引用恢复，以及原有正常任务成功表现。

## 运行命令

将运行包 v2-hard-recovery-sft-weighted-v1-20261103.zip 上传到云端：

/root/rivermind-data/AgenticPostTrainingLab/v2/

然后完整执行：

```bash
unzip -o \
  /root/rivermind-data/AgenticPostTrainingLab/v2/v2-hard-recovery-sft-weighted-v1-20261103.zip \
  -d /root/rivermind-data/AgenticPostTrainingLab &&
cd /root/rivermind-data/AgenticPostTrainingLab/v2 &&
/root/rivermind-data/envs/apt-vllm/bin/python -u \
  scripts/run_v2_hard_recovery_sft.py
```

ZIP 内部带有 v2/，解压目标是项目目录。命令使用 &&，解压成功后才进入运行阶段。

阶段顺序为 train → checkpoints → recovery → greedy → sample → audit。训练终端显示258步进度条；检查点及前后恢复评测显示420／896动作进度条。各阶段输出同步写入 artifacts/hard-recovery-sft-weighted-v1-20261103/ 下的日志。

完成后回传：

/root/rivermind-data/AgenticPostTrainingLab/v2/hard-recovery-sft-weighted-v1-20261103-review.zip

## 本地验证

已完成：

- 父 adapter 与训练数据哈希核对。
- 169 条 chosen 的文本、编码与父训练一致；examples SHA256 为 f7c4ab0a82cd019f7287e0b5ed8ec5dda8a8dc5e4d2e1ff4d9896f8af7ef8c0d；encoded SHA256 为 7eb17e79c4a0a6bb72385282277df8e23edf0a78838ab332e7fb998f67f7848f。
- 84 个训练上下文逐字命中训练 messages，28 个开发上下文位于该数据集之外。
- 112 个上下文的专家动作均能推进验证或通过检查。
- 权重梯度与手算目标一致。
- 小型真实 Qwen+LoRA 完成258次加权更新，冻结参数保持一致，Adam步数、轮次、样本游标与记录一致。
- 逐检查点及两种数据划分的原生轨迹重放、记录篡改检测、完整任务配对评测与采样种子检查。
- 阶段失败时保留实时日志，并生成回传ZIP。
- 正式 tests 目录120项通过。

## 主要代码与回传文件

新增代码：

- src/apt_v2/hard_recovery_sft.py：场景权重、加权更新和训练历史验证。
- scripts/train_v2_hard_recovery_sft.py：父模型载入、加权续训、检查点、训练前后逐token评分。
- scripts/evaluate_v2_hard_recovery_sft.py：逐检查点恢复、前后恢复与完整任务评测。
- scripts/audit_v2_hard_recovery_sft.py：监督目标、权重、状态、轨迹和统计审计。
- scripts/run_v2_hard_recovery_sft.py：总入口、时限、实时日志和回传打包。
- tests/test_hard_recovery_sft_v2.py：7项新增测试。

回传ZIP包含训练更新记录、weights.json、前后chosen分数、检查点元数据、checkpoint_progress.json、逐动作原生记录、完整任务轨迹、分项CSV、训练／开发对照、审计和最终adapter。中间adapter与state.pt保留在云端，由云端审计检查其哈希；最终adapter随包回传。

报告原件：v2/reports/hard-recovery-sft-weighted-v1-20261103.md。
预检数据：v2/artifacts/datasets/hard-recovery-sft-weighted-v1-20261103/。
