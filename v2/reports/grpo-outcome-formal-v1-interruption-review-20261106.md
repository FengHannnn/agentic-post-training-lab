# C2 outcome GRPO 首轮中断复核与续跑

日期：2026-11-06。运行标识：grpo-outcome-formal-v1-20261105。

## 结论与下一步

已提交 17 个训练组，完成 13 次真实更新。第 18 组（group-0017）采样在严格重放时触发 EOS 校验错误；根因是校验器只接受 tokenizer EOS=151645，漏掉冻结模型生成配置中的另一个有效 EOS=151643。修复后的真实失败轨迹完整重放通过，141 项回归测试通过，续跑补丁已完成干净目录安装验证。

本轮继续从 checkpoints/group-0016 的 RL13 策略和原 Adam/RNG 状态开始，完成剩余 7 组、第 24 组开发评测及最终刷新。作业序号仍为第 1 个，累计时间继续计入原 4 小时预算。

## 返回包与已完成训练复核

返回包 SHA256：c063aa5dc137a4e3232ffdee9541a90eea9e8c5e3f666fcfd5f8dc7d94aebdfb。490 个条目的 CRC、路径、重复项与符号链接检查通过。188 个冻结源文件匹配原发布清单，22 项输入文件绑定通过。

17 个训练组的 68 条 episode、471 个生成动作、39471 个响应 token 完成严格重放；13 组有效更新、4 组零方差跳过。13 次更新均改变 112 个 LoRA 张量，累计验证 1456 项张量变化；权重 delta、评分缓存与 token/q 对齐、Adam step、RNG roundtrip、reference 冻结记录及 checkpoint 提交链均通过检查。

17 次训练 worker 加载分别从实际 adapter 独立投影为 BF16 packed LoRA，全部 112 个张量的 metadata/SHA 匹配。独立重算 13 次目标、TIS 与 sampled k3，最大数值差为 5.315e−8。所有更新的 clip fraction 为 0，最大 sampled k3=6.6411e−7；same-path old/fresh 评分差均为 0。reference 实际张量摘要始终为 3c7177e7987218a24f9ad2f7c44170ad30fa712b480538ae50030b9af39100b1。

当前策略 adapter SHA256：c8eeb7e835ecde292c9496883ec33adb5838e4869d2eca1e3ba8f0fe97dc276d。其谱系为 SFT513 / DPO11 / RL13。state.pt 留在云端，返回包提供保存/恢复/提交校验记录；补丁安装时还会实际核对云端最新 state.pt 与 complete.json 的 SHA256。

|组|任务|起始类型|成功数|处理|累计真实更新|
|---:|---|---|---:|---|---:|
|0|train-000|normal|4/4|零方差跳过|0|
|1|train-000|repeat_sql_error|3/4|更新|1|
|2|train-001|normal|3/4|更新|2|
|3|train-001|sql_unverified_requery|1/4|更新|3|
|4|train-002|normal|3/4|更新|4|
|5|train-002|repeat_evidence_error|1/4|更新|5|
|6|train-003|normal|4/4|零方差跳过|5|
|7|train-003|failed_check_repair|2/4|更新|6|
|8|train-004|normal|3/4|更新|7|
|9|train-004|stale_evidence_repair|3/4|更新|8|
|10|train-005|normal|2/4|更新|9|
|11|train-005|format_repair|4/4|零方差跳过|9|
|12|train-006|normal|1/4|更新|10|
|13|train-006|unknown_tool_repair|4/4|零方差跳过|10|
|14|train-007|normal|2/4|更新|11|
|15|train-007|repeat_sql_error|3/4|更新|12|
|16|train-008|normal|3/4|更新|13|

## 已完成开发评测

W 与 group_cursor=8 的模型各完成 132 条 episode 和 14 次 greedy 质量动作。第 8 组节点对应实际 RL6。两组评测共 264 条 episode、1288 个生成动作；严格环境重放、calendar evidence、固定任务/场景/seed、实际 worker 张量绑定与重复 greedy token 一致性全部通过检查。

|指标|W|第 8 组后 / RL6|变化|
|---|---:|---:|---:|
|greedy complete|4/4（100.00%）|4/4（100.00%）|0.00 pp|
|greedy recovery|19/28（67.86%）|20/28（71.43%）|3.57 pp|
|sample complete|16/16（100.00%）|16/16（100.00%）|0.00 pp|
|sample recovery|61/84（72.62%）|60/84（71.43%）|-1.19 pp|

完整成功率已饱和，按已确认边界以恢复成功率为主要观察指标。当前随机恢复为 60/84，较 W 少 1 条；greedy 恢复为 20/28，较 W 多 1 条。两处配对变化如下：

|模式|任务|场景|rollout|W→RL6|
|---|---|---|---:|---|
|sample|development-001|repeat_evidence_error|0|成功→失败|
|greedy|development-002|sql_unverified_requery|0|失败→成功|

随机恢复其余 83 条配对结果保持一致，greedy 恢复其余 27 条保持一致。后续先完成第 24 组节点，再依据完整开发结果选择调参方向；最终新任务确认保留在 C4。数值验收阈值继续沿用此前暂不确定的约定。

## 根因、修复与测试

可重复运行的原始复现：使用返回包中的 tokenizer、冻结 grpo.py 和 group-0017 的第二条完整 episode 调用 replay，稳定报错 ValueError: stop requires one native terminal EOS。定位到第 5 个续跑动作，原生 token 末尾为 [151644,198,151643]；文本解码、prompt、processed q 长度与概率有限性均先于 EOS 检查通过。

冻结 generation_config.json 的 eos_token_id=[151645,151643]。vLLM 0.11.0 的 ModelConfig.try_get_generation_config 即使 generation_config='vllm' 也读取特殊 token 配置，Processor 再把其中 EOS 合并进停止集合。这解释了真实引擎为何以 151643 正常返回 stop。来源：[ModelConfig](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/config/model.py#L1214-L1239)、[Processor](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/engine/processor.py#L356-L384)、[SamplingParams](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py#L437-L459)。

修复仅调整 src/apt_v2/grpo.py 的原生终止校验：从已冻结的本地 generation_config.json 读取 EOS，与 tokenizer EOS 合并；stop 仍要求恰好一个末尾 EOS，前部 EOS 和其他末尾 token 继续拒绝。length 分支按 vLLM 的“先检查 token 上限、后检查 EOS”顺序接受恰好落在上限的末尾 EOS。来源：[check_stop](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/sched/utils.py#L39-L63)。

采样器、训练超参、reward、q/logprob、loss、任务/场景/seed 计划均保持原配置。失败组尚未提交 K=4 的完整组；续跑会使用原 checkpoint 与 seed 重新执行该组，原 partial 轨迹保留在 review 中。

验证记录：

- 原始完整轨迹复现先失败；补丁后 group-0017 的两条真实 episode 均在实际 tokenizer 上完整重放通过。
- 新增 8 项原生 EOS 用例：真实 151643 记录、多个 EOS、非法末尾/内部 EOS、length 上限优先级、配置异常与单 EOS fallback。
- 新增 5 项安装测试：训练/评测/预算/优化器保持、源码/配置/优化器篡改拒绝、重复安装、安装中断恢复。
- 141 项完整测试通过，耗时 92.58 秒；最终 installer 修订后 5 项安装测试再次通过。
- 从原始正式发布 ZIP 干净解包，加入真实返回数据后安装补丁，两次安装验证幂等；17 个训练组、264 条开发 episode、两条失败组 partial episode 再次通过，147 个受保护文件保持一致，188 个原始源码快照保持原样，新增有效源码绑定通过。

补丁同时保存原 release/run manifest、补丁本体、测试证据和新的源码快照到 source_revisions/grpo-native-eos-v1，更新有效运行源码哈希，保留这次修订的完整来源。

## 资源与预算

首次尝试耗时 5865.0854 秒（1 小时 37 分 45 秒），剩余 8534.9146 秒（2 小时 22 分 15 秒）。两个开发评测阶段分别耗时约 30.3、30.6 分钟。按已测开销估计，剩余训练、最终开发评测和审计约需 45–60 分钟。

13 次更新均记录显式 activation checkpoint 覆盖 28 层。此次更新峰值 allocated=8,711,944,192 bytes（8.11 GiB），reserved=9,204,400,128 bytes（8.57 GiB）；预检对应观察峰值为 20.49 / 46.82 GiB。训练和评测进度条继续显示在终端。

## 云端续跑

下载 grpo-outcome-formal-v1-20261105-eos-hotfix-v1.zip，上传至原云端 v2 目录。补丁 ZIP 为 14 项 / 40215 bytes，SHA256：707b092f1ab3ad13a6cbb088e4d12fa345bf1355c3e67978a0d233d53c8f206b。

```bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-formal-v1-20261105-eos-hotfix-v1.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u hotfixes/grpo-native-eos-v1/apply.py && \
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo.py --config configs/grpo-outcome-formal-v1.json --resume
```

安装器先校验原发布文件、17 个提交及最新云端 state.pt，完成后才启动续跑。已有 W/第 8 组评测会先做完整性复核，随后继续 group-0017。

完成后返回 grpo-outcome-formal-v1-20261105-review-attempt-002.zip。剩余工作为 7 个训练组、第 24 组配对开发评测、最新策略刷新与整轮审计。
