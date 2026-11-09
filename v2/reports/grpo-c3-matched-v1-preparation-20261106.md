# C3 细粒度 GRPO 与新 L40S 实例运行准备

日期：2026-11-06。运行标识：grpo-c3-matched-v1-20261106。

## 新 L40S 实例

可以迁移到另一张 L40S。模型权重、任务和算法配置由哈希核验；新实例在prepare阶段记录新的GPU UUID、软件环境与驱动。W、outcome对照和fine分支在这张新卡上按相同协议评测，运行耗时和显存单独记录。本轮作为新的C3运行，从W/新Adam开始，两支各自独立，原C2 RL16候选与结果保留。

即使硬件型号相同，也需要核对驱动、库版本、seed与生成轨迹；本轮保留重复greedy质量检查及原生token重放。[vLLM 0.11.0复现说明](https://docs.vllm.ai/en/v0.11.0/usage/reproducibility.html)

## 首轮设计

|项目|安排|
|---|---|
|预算|用户确认单作业最多4小时，包含双分支训练、全部评测和审计|
|起点|两支均独立从W（SFT513/DPO11/RL0）开始，新建AdamW，冻结W reference|
|训练|每支24组、K4，normal/recovery各12组，共192条训练轨迹|
|共同参数|LR1e-6、T1、clip.2、beta.02、TIS cap2、grad clip1，原12轮动作预算|
|对照变量|outcome为整轨迹优势；fine为可验证potential shaping后的逐动作return-to-go及同相对位置组内优势|
|标准开发评测|W、outcome最终、fine最终各132条episode和14次greedy质量动作|
|新增业务评测|三模型各4条完整结账→下一账期任务，共12条greedy episode，单独记录|
|刷新与审计|两支最终策略各再采样4条，严格重放、权重/Adam/RNG/评分缓存与配对统计核验|

fine沿用Phi=.2×已验证指标数、终局Phi归零；逐动作shaped reward为outcome加potential差，从当前动作累计到终局形成return-to-go。组内按相对动作位置、对仍有该动作的轨迹计算均值与population std；不足2条或零方差位置的优势为0。终局回报相同但进展时序不同的轨迹可产生动作信用，因此同时报告实际更新数和不同零方差口径。本轮为纯GRPO。

完整成功率饱和时以恢复成功率为主要观察指标；数值阈值按此前约定暂存，C4继续负责新任务最终确认。

## 真实 SQLite 依赖扩展

新工具close_period要求已验证的月度数值与证据，实际写入period_closings表。后续SQL读取该结转行与下一账期ledger_movements，得到next_balance_cents，并在两个金额扰动世界验证SQL。

四个脚本专家任务全部通过，标准路径9或10个动作。删除结转行后查询返回空结果；篡改结转值会改变下一账期余额并触发完整性检查。该扩展在本轮用于开发迁移评测，训练保持原任务起点分布。

## 本地验证

160项不同用例已验证：159项整体验证通过；补齐业务中断回传后，C3的19项针对性测试通过，细粒度真实梯度与独立目标复算用例再次通过。169个Python文件语法检查通过。测试包含同回报不同进展的动作信用、prefix排除、singleton深度、真实LoRA梯度、SQL依赖删除/篡改、防硬编码、新UUID接受与作业内身份约束、双分支交替调度和断点恢复。

发布包已通过干净目录验证：215个条目、214个绑定文件哈希一致，170个Python文件语法检查通过，5个命令入口通过；包内19项C3测试及4个依赖任务再次通过。ZIP SHA256：3d84416592efb58eebde51a0cdfae246167a2806110e2c350294b37d30a09e36。云端执行后返回review ZIP用于复核C3实际结果。

## 代码阅读顺序

以下均为v2根目录相对路径。

|相对路径|大致作用|
|---|---|
|configs/grpo-c3-matched-v1.json|整轮双分支安排和4小时时限|
|configs/grpo-c3-outcome-v1.json / configs/grpo-c3-fine-v1.json|共同参数与两种信用方法|
|scripts/run_v2_grpo_c3.py|双分支训练、评测、续跑和回传入口|
|src/apt_v2/grpo_credit.py|逐动作return-to-go与组内优势|
|src/apt_v2/grpo_optimizer.py|将动作优势用于真实梯度更新|
|src/apt_v2/period_closing.py|结转记录写入、下一账期查询和真实依赖证明|
|scripts/evaluate_v2_grpo.py|标准开发与新增业务完整评测|
|scripts/audit_v2_grpo_c3.py|双分支审计、配对比较和信用曲线|
|tests/test_grpo_c3_v2.py|细粒度、依赖业务、调度与恢复验证|

## 云端执行

将grpo-c3-matched-v1-20261106-run.zip上传到克隆后的v2目录，保持基座缓存与Python环境路径可用。包内附W adapter，prepare会核对完整基座哈希并绑定新L40S身份。

```bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-c3-matched-v1-20261106-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_c3.py --config configs/grpo-c3-matched-v1.json
```

终端显示总阶段、rollout、评分、反传与评测进度条。中断时保留artifacts，按同命令追加--resume恢复，累计已耗时间不重置。结束后返回grpo-c3-matched-v1-20261106-review.zip。
