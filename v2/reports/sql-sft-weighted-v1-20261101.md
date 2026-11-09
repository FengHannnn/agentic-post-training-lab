# SQL 加权混合 SFT v1

日期：2026-11-01（Asia/Shanghai）。用户选择：保留现有示范，提高 SQL 动作损失权重。代码、本地验证与 L20 运行包已完成。

## 本轮配置

从已复核的 63-step SFT final adapter 出发，继承其 Adam 状态。继续训练两遍，共 504 次动作曝光、63 次新增更新，Adam 总步数预计 63→126。LoRA、学习率 5e-5、每 8 个动作梯度累积、FP32 与 2048 训练长度沿用首轮设置；梯度裁剪 norm=1。

保持原有 12 个训练任务、36 条轨迹、252 个动作样本，数据 SHA-256 为 dc2ef7679af39005e91bd8f231348dd0a74932e2fd7401942f6681e93fd5819b。共 72 个 SQL 动作、180 个其他动作，监督 token 18,480，最长上下文 1,430 token。

SQL 原始权重 4，其他动作权重 1。用全数据平均权重 468/252 归一化后，SQL 实际系数约 2.153846，其他动作约 0.538462，数据平均系数为 1。每个动作先对自身 response token 的交叉熵求均值，再按动作系数和累积批量计算 loss。SQL 项占总样本权重约 61.5%。

保留其他动作监督以维持协议、证据引用与提交行为。对样本内容、采样顺序和训练参数分别记录；新增两遍的 shuffle 使用原 seed 加 epoch 2、3。

## 评测与完成依据

训练前后分别对 252 个固定动作进行未加权 teacher-forced 评分，报告分工具 NLL、token 准确率与全 token 匹配数。完整执行评测使用 vLLM BF16、相同 greedy 参数，分别加载父 adapter 和本轮 weighted adapter，运行 train 12 + development 4，共 32 条轨迹。

主要观察完整任务成功、有效月度 SQL、已验证指标、故障恢复、格式、调用数和输出 token。SQL 学习重点为子查询结构、paid 筛选、退款发生月份和 current/prior 月份绑定。最终报告描述“继续训练并加入 SQL 加权”这套方案相对父模型的变化；等预算的均匀权重续训可作为后续独立对照分支。

训练完成后执行 CPU 逐轮重放审计。训练、评测、审计共用 5400 秒总预算，并预留打包时间。每 8 次更新和 final 保存 adapter、Adam、进度及 Python/Torch/CUDA RNG。现有 pilot 与诊断产物保留，新输出独占创建。

## 云端运行

上传 v2-sql-sft-weighted-v1-20261101.zip 至 /root/rivermind-data/，执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
ls -lh /root/rivermind-data/v2-sql-sft-weighted-v1-20261101.zip
unzip -o /root/rivermind-data/v2-sql-sft-weighted-v1-20261101.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_sql_sft.py
```

另一个终端查看训练日志：

```bash
tail -f /root/rivermind-data/AgenticPostTrainingLab/v2/artifacts/sql-sft-weighted-v1-20261101/train.log
```

回传文件：/root/rivermind-data/AgenticPostTrainingLab/v2/sql-sft-weighted-v1-20261101-review.zip。

入口核对父 pilot 的数据、固定模型、adapter 身份及 Adam 状态。复用 L20 上现有环境和模型缓存。

## 本地验证

V2 全部测试 66 passed，源码编译通过。新测试覆盖损失系数和全局归一化、原数据逐项一致与 train 隔离、微型真实 Qwen + LoRA 的加权梯度和手算目标一致、成功与预算结束的失败轨迹重放、输入和观察篡改检测，以及评测失败后停止审计并保存 ZIP。

原始回传数据重新编码，252 个样本与已归档 pilot 一致，最大长度 1430，父 Adam 参数状态均记录第 63 步。

## 文件位置

- src/apt_v2/sql_sft.py：权重、数据身份、批量 backward 和轨迹审计。
- scripts/train_v2_sql_sft.py：Adam 继承、加权续训、检查点、前后动作评分。
- scripts/evaluate_v2_sql_sft.py：父模型与 weighted 模型的同设置评测。
- scripts/audit_v2_sql_sft.py：32 条轨迹和数据、检查点审计。
- scripts/run_v2_sql_sft.py：总入口、时间预算和回传打包。
- configs/sql-sft-weighted-v1.json：本轮配置。
- tests/test_sql_sft_v2.py：新增自动化测试。
- artifacts/releases/sql-sft-weighted-v1-20261101/：发布 ZIP 与摘要凭据。

报告以后以项目 v2/reports 为主位置，提供项目文件链接，保持关闭。
