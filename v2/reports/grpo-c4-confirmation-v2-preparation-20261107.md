# C4 expanded confirmation preparation

日期：2026-11-07。运行：`grpo-c4-confirmation-v2-20261107`。

按新增2–3小时授权，C4从8个任务扩至12个，预算从4小时扩至7小时；预计约4.5–6小时。用户确认v1尚未启动，实际运行采用本v2包。原v1包、报告及其哈希保留。

## 为什么这样扩容

新增独立数据实例与任务族覆盖，使配对增益接受更多数据检验。每个任务仍做4次sample完整、3次sample恢复，模型权重与采样参数保持原样。前8个任务与未运行的v1一致，新增4个任务补齐2027年3/6/9/12月，合计覆盖全年，6个任务含首次查询临时故障。业务仍为既有月度对账，任务id、family和seed与历史训练/开发/测试隔离。

| 项目 | v1 | v2 |
|---|---:|---:|
| 独立任务实例 | 8 | 12 |
| 每模型episode | 264 | 396 |
| 三模型episode | 792 | 1,188 |
| 每模型sample恢复 | 168 | 252 |
| 每模型sample完整 | 32 | 48 |
| 每模型greedy恢复 | 56 | 84 |
| 每模型greedy完整 | 8 | 12 |
| 最大作业时间 | 4小时 | 7小时 |
| 最低恢复净增 | 4/168 | 6/252 |

## 模型与验收

比较W、C2 outcome RL16、C3 fine RL17，三份适配器均随包提供；新增训练为0。底座仍使用云端固定Qwen2.5-1.5B-Instruct缓存。相同任务/场景/mode/rollout使用同一生成seed；每模型独立进程，只加载LoRA编号1。

实用门槛保持+2.38个百分点，对应sample恢复至少比同机W多成功6/252。sample完整、greedy完整和greedy恢复均不下降。沿用原主指标优先级：W sample完整若为48/48，主要看恢复；否则完整成功数还需增加。固定三模型、任务、规模与门槛后运行，结果中保留全部候选的任务级、场景级配对变化和开销。bootstrap按12个任务族聚类，区间用于描述统计。

## 验证与执行

31项相关测试通过，覆盖原v1兼容、扩容任务与门槛、96个专家可完成性起点、两版作业的成功/失败/超时回传，以及原生396条episode的中断续跑。178个Python文件语法检查、3个命令入口通过。已提交的前5条轨迹在恢复后保持字节一致。

每条episode提交前验证真实worker加载权重，并核验原生token、环境重放及SQL退款日历。总预算25,200秒，含准备、三模型推理和审计，预留120秒打包；中断续跑累计时间。终端显示5个总阶段和每模型410个episode/质量动作进度，并显示当前工具轮次与响应token数。

干净解压验证通过：226个绑定文件哈希一致，179个Python文件语法通过，3个命令入口和17项重点测试通过。三份适配器各112个张量、1,089,536个参数，BF16投影和权重SHA256均匹配。检查后全部绑定文件保持原样，v1原始ZIP哈希也保持原样。

上传 `grpo-c4-confirmation-v2-20261107-run.zip` 到云端v2目录后执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-c4-confirmation-v2-20261107-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_c4.py --config configs/grpo-c4-confirmation-v2.json
~~~

运行结束返回 `/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-c4-confirmation-v2-20261107-review.zip`。中断时在原Python命令后加 `--resume`，同一张L40S继续剩余episode和剩余预算。

## 项目文件与阅读顺序

本报告：`reports/grpo-c4-confirmation-v2-preparation-20261107.md`。完整冻结契约：`reports/grpo-c4-confirmation-contract-v2-20261107.md`。本地回执：`reports/grpo-c4-local-validation-v2-20261107.json`。

以下为相对v2根目录的阅读顺序：

1. `configs/grpo-c4-confirmation-v2.json`：12任务、7小时和验收门槛。
2. `src/apt_v2/grpo_confirmation.py`：任务、配对统计与验收。
3. `scripts/run_v2_grpo_c4.py`：调度、预算和返回包。
4. `scripts/grpo_c4_worker.py`：预检、推理与逐episode恢复。
5. `scripts/grpo_c4_common.py`：输入、原生重放和独立审计。
6. `scripts/build_v2_grpo_c4_release.py`：冻结运行包构建。
