# C2 GRPO v2 预检复核与首轮正式运行

复核日期：2026-11-06。v2 预检通过，首轮 outcome GRPO 正式包进入云端运行准备。

## 预检结果

返回包：grpo-outcome-preflight-v2-20261105-review.zip。SHA256：6addd0ca11afb9477c7571ea01470ddd4cc755b63d21730ada65e45a2b03620f。

压缩包 260 项的 CRC、路径与文件类型检查通过；183 个冻结源文件匹配 v2 发布清单，15 项输入绑定通过。使用返回包中的冻结源代码执行 local audit，所有检查通过。云端运行耗时 461.9235 秒，无错误，以 update_cap 结束；共提交 3 个训练组，完成 2 次真实更新，随后采样最新策略验证刷新。

|组|用途/起始策略|起始类型|成功条数|总动作数/条|回报/条|处理|
|---|---|---|---:|---|---|---|
|0|训练/W|normal train000|4/4|7,7,7,7|.93,.93,.93,.93|零方差跳过|
|1|训练/W|recovery train000 · repeat_sql_error|1/4|12,9,12,12|−.12,.91,−.12,−.12|更新至 version1|
|2|训练/version1|normal train001|3/4|8,8,12,8|.92,.92,−.12,.92|更新至 version2|
|3|刷新/version2|recovery train001 · sql_unverified_requery|1/4|12,12,12,9|−.12,−.12,−.12,.91|刷新采样|

共 16 条 episode、118 个生成动作、9410 个响应 token，另有 4 个概率探针 token。所有 episode 与恢复 prefix 均严格重放通过，最大 prompt 2016 token。表中是训练及刷新组结果；正式阶段使用冻结开发集与 W 做配对比较。

## 更新与加载证据

|指标|第 1 次更新|第 2 次更新|
|---|---:|---:|
|loss|0.0001081392|0.0003470785|
|sampled k3|0|1.4480e−8|
|clip fraction|0|0|
|TIS mean|1.0008447|1.0001971|
|TIS cap fraction|0|0|
|裁剪前 gradient norm|0.7920758|0.2866047|
|参数 delta L2|0.0010418712|0.0007325468|
|改变的 LoRA 张量|112|112|
|Adam step|1|2|

两个更新均满足 old/fresh 同路径评分 gate，最大差为 0；reference 实际张量保持冻结。第 2 次更新成功恢复 Adam 与 RNG，Adam 状态覆盖 112 项。云端 state.pt 保留在 checkpoint 中；返回包包含对应的保存、恢复、校验与提交记录，本地复核使用这些记录及返回的 adapter/评分缓存。

另行从每个实际 adapter 重建 vLLM BF16 packed LoRA 张量（q/v 投影，B 按 alpha/r=2 缩放），4 次加载各 112 个张量的 metadata 与 SHA 均与 worker 实际加载结果匹配。两次更新使用 BF16 有效权重的哈希均发生变化。独立从返回评分缓存重算目标、TIS 与 k3，最大差分别为 1.9372e−8、1.6764e−8。

最终 adapter：checkpoints/group-0002/adapter；SHA256：532b8fb5ddac8044d5af7e6652b4dd95bf60d453fea43f49ee2f3f8f45b87f22。冻结 reference 实际张量摘要：3c7177e7987218a24f9ad2f7c44170ad30fa712b480538ae50030b9af39100b1。

v1 的 callable RPC 已替换为注册 worker extension 的具名字符串方法；v2 在默认序列化限制下完成采样、两次更新和刷新验证。

## 资源与内存调整

运行设备为 NVIDIA L40S，物理显存 50,867,666,944 bytes。环境为 Python 3.11.17 / torch 2.8.0+cu128 / vLLM 0.11.0 / transformers 4.57.6 / peft 0.17.1。采样耗时 370.5307 秒，HF 更新计算耗时 42.6818 秒。更新峰值 allocated 20.49 GiB、reserved 46.82 GiB。

本地检查发现 HF gradient checkpointing 的默认路径仅在 training 模式触发，而本项目评分与更新维持 eval 模式。正式包增加 decoder layer 显式 nonreentrant checkpoint，并在动作完成后释放可用 CUDA cache。测试已覆盖实际 checkpoint 调用、评分一致、全部 LoRA 梯度一致；Qwen2 SDPA 用例中 A/B 两侧梯度均非零。正式云端运行将记录新的实际峰值与 checkpoint 层数。

## 首轮正式实验

从 W（SFT513/DPO11/RL0）独立开始，重新创建 AdamW，不继承预检 RL2。W adapter SHA256：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。

- 24 个训练组，normal/recovery 各 12 组；K=4，共 96 条训练轨迹；另采样 4 条最新策略刷新轨迹。
- W、第 8 组后、第 24 组后各做一次开发评测。每次 132 条完整 episode：4 greedy complete、16 sample complete、28 greedy recovery、84 sample recovery，另做 14 次 greedy 质量动作。
- 三个模型总计 396 条开发 episode、42 次质量动作，使用相同任务、场景、seed 和评测参数。返回 task/scenario 细分、相对 W 的配对变化、训练曲线、评测曲线及审计结果。
- 组数和实际更新数分开记录。零方差组跳过，单组至多更新一次；第 8/24 组是评测节点，对应实际更新数由运行结果给出。
- 保持 outcome 回报和既定 GRPO/TIS/k3 目标；LR=1e−6，clip=.2，beta=.02，TIS cap=2，grad clip=1，采样 T=1。本轮为纯 outcome GRPO。
- 本轮为最多 3 个正式作业中的第 1 个，单轮最多 4 小时，总预算 12 GPU 小时。按预检开销外推约 3–4 小时，运行器按实际累计时间控制预算并预留打包时间。

最终目标仍为 GRPO 超过 W；完整成功率饱和则以恢复成功率为主。数值阈值按已确认边界留待后续确定；本轮输出完整开发配对记录与候选排序，C4 再做新任务最终确认。

## 本地验证与云端执行

128 项测试通过，耗时 86.93 秒；其中 20 项覆盖正式评测计划/配对/审计、防篡改、节点执行顺序、评测中断恢复与内存 checkpoint。发布包已冻结上述契约、验证结果、W 输入和 v2 GPU profile 的 SHA256。干净目录复核通过：202 个条目、201 个绑定文件、160 个 Python 文件 AST 检查、4 个命令入口；包内生产校验代码再次验证 v2 的 4×112 个实际 worker 张量。正式运行 ZIP SHA256：792c66f57832dd42ba6005b10577969efe8899a96e37b6d3033c18bca17ee9e3。

将 grpo-outcome-formal-v1-20261105-run.zip 上传到云端 v2 目录，执行：

```bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-formal-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo.py --config configs/grpo-outcome-formal-v1.json
```

终端显示总阶段、采样/评分/更新/评测进度。完成或异常后下载 v2/grpo-outcome-formal-v1-20261105-review.zip 交回复核。中断后保留云端 artifacts，按同一命令追加 --resume 恢复，已用时间累计计入本轮预算。报告按约定只生成链接。
