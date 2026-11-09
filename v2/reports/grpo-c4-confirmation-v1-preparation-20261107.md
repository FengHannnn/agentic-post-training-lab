# C4 independent confirmation preparation

日期：2026-11-07。运行：`grpo-c4-confirmation-v1-20261107`。

C4运行包已准备，比较最佳加权SFT W、C2 outcome RL16、C3 fine RL17。新增训练更新为0；云端运行后按预先固定门槛判断独立确认结果。

## 规模与验收

8个新的月度对账实例/任务族标识，seed91000–91007，账期和区域/segment组合独立冻结，覆盖跨年与4个临时查询故障实例。业务模板、SQLite规则和12步总预算沿用既有环境。每模型264条episode，三者共792条；正常另有42次重复greedy质量动作。

每模型分母：sample完整32、sample恢复168、greedy完整8、greedy恢复56。候选需sample恢复至少比同机W多成功4条（+2.38个百分点），sample完整、greedy完整和greedy恢复成功数均不下降。沿用主指标顺序：W完整若为32/32，主要判断恢复；否则完整成功数还需增加。任一GRPO候选满足全部条件即通过实用确认。

报告同时保留配对改善/退步、各任务/场景成功数、动作与token开销，并提供按8个任务族聚类的描述性95% bootstrap区间。三模型身份和所有任务、种子、参数、门槛在推理前固定。

## 运行方式

单张L40S，最多4小时，含准备、全部推理和审计，预留120秒打包。预计约3–4小时，具体耗时取决于新任务的实际轨迹长度。

三份候选适配器均已随包提供，base使用云端已有固定缓存。准备阶段校验完整base和adapter哈希，并绑定当次GPU UUID、驱动与软件身份。每模型独立进程、单LoRA编号1；每条episode提交前核对真实worker加载权重，随后校验原生token、环境重放和独立退款日历证据。64个脚本专家完整/恢复起点均在原12步预算内完成。

终端显示总阶段进度和每模型278个episode/质量动作进度；每次工具调用显示turn与响应token。中断后已验证episode保持原样，只生成剩余episode；每个新session重新做greedy质量和加载权重检查，额外动作单独计数。预算在续跑间累计。失败、超时及正常完成均生成review ZIP，作业运行成功与效果验收分别记录。

## 本地验证

115项测试通过：C4核心19项、作业成功/超时/失败回传3项、既有GRPO回归93项；176个Python文件语法检查通过，3个命令入口检查通过。中断验证先保存5条episode，再恢复至全部264条，前5条文件字节保持不变。验收边界、各类回归保护、跨年SQL日历、原生token、真实BF16权重投影和新任务隔离均已验证。

干净解压检查：223个绑定文件哈希一致，177个Python文件语法通过，3个命令入口通过，17项重点测试通过。三份候选均核验为112个张量、1,089,536个参数，BF16投影及实际adapter SHA256均匹配；解压目录运行检查后全部绑定文件保持不变。

本地验证使用真实SQLite环境、实际适配器BF16张量投影和原生token夹具；L40S上的Qwen生成与最终成功率由云端确认作业产生。

## 上传与执行

上传 `grpo-c4-confirmation-v1-20261107-run.zip` 到云端v2目录，执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-c4-confirmation-v1-20261107-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_c4.py --config configs/grpo-c4-confirmation-v1.json
~~~

中断后在同一张L40S、相同目录执行原Python命令并追加 `--resume`；仍沿用本轮剩余预算。

首轮返回：`/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-c4-confirmation-v1-20261107-review.zip`。续跑返回attempt编号ZIP。将返回包交回后，本地重放并重算全部验收结果。

## 文件与代码阅读顺序

本报告位于 `reports/grpo-c4-confirmation-v1-preparation-20261107.md`；冻结契约 `reports/grpo-c4-confirmation-contract-v1-20261107.md`；验证回执 `reports/grpo-c4-local-validation-v1-20261107.json`。

以下路径相对v2根目录：

1. `configs/grpo-c4-confirmation-v1.json`：候选、评测规模和约束。
2. `src/apt_v2/grpo_confirmation.py`：新任务、配对统计和验收判断。
3. `scripts/run_v2_grpo_c4.py`：作业调度、累计预算和失败打包。
4. `scripts/grpo_c4_worker.py`：预检、推理与逐episode续跑。
5. `scripts/grpo_c4_common.py`：原生轨迹、权重证明及独立审计。
6. `scripts/build_v2_grpo_c4_release.py`：冻结源码与候选权重发布。
