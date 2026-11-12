# Fine GRPO 三方独立复评运行说明

本轮固定 W、旧 local RL43 和新收尾信用 RL47，按同一协议评估 12 个全新任务族。每模型 396 条标准轨迹、48 条专项轨迹，恢复采样的信用对照阈值为 +6/252。[实验契约](grpo-fine-closeout-followup-contract-v1-20261112.md)

## 从公式看信用变化

一组有 K=4 条同起点轨迹。设第 i 条轨迹总动作数为 T_i，计入起点前缀，S_i 为提交是否被验收。

```text
R_i = S_i - 0.01 * T_i
A_outcome,i = (R_i - mean(R)) / (population_std(R) + 1e-8)
A_local,i,t = A_outcome,i + 0.2 * n_i,t
A_closeout,i,t = A_local,i,t + c_i,t
```

同组回报完全相等时，终局优势为 0。n_i,t 是本动作首次认证的指标数，起点已认证和重复认证的指标均不重复计数。终局项对同一条轨迹的每个动作相同，n_i,t 只落在产生认证的动作。

收尾项 c_i,t 单独记账。第一次有效检查且还有至少一步时分配 0.1，成功提交时分配 0.2 减去之前已分配的收尾加成。有效直接提交拿到 0.2，检查通过再提交各拿 0.1。若前缀已经通过检查，续跑中重复检查拿 0，成功提交拿 0.2。重复检查、失败检查、无效提交及耗尽动作预算的检查均加 0。一次有效检查后任务最终失败时，最多保留 0.1 的收尾加成。

举例而言，四条轨迹都成功、总动作数相等，且续跑没有新增指标，则原 local 对检查和提交的优势都是 0。新信用给有效检查和提交各 0.1，重复检查维持 0。若某条失败轨迹的终局优势为 −1，首次有效检查只把该动作的优势提高到 −0.9。终局约束仍体现在总优势中。

优化器将该动作优势代入裁剪策略项。在未被裁剪的区域，正优势使策略项倾向提高已采样响应 token 的概率，负优势相反。KL 参考项、重要性权重、梯度裁剪和按 token、动作、轨迹平均的方式沿用上一轮配方。[动作信用实现](../src/apt_v2/grpo_credit.py) · [优化目标](../src/apt_v2/grpo_optimizer.py)

136 项相关回归及干净解压后的 12 项定向测试通过。原生 SQLite 重放 264 条专家证明。三个 CLI 入口、模拟中断与续跑、累计预算和回传收据均通过核验。[本地验证记录](grpo-fine-closeout-followup-validation-v1-20261112.json)

## 云端运行

将 `grpo-fine-closeout-followup-v1-20261112-run.zip` 上传到仓库根目录，即 `v2/` 的上一级，然后执行。

~~~bash
(
set -e
unzip -o grpo-fine-closeout-followup-v1-20261112-run.zip -d .
cd v2
../../envs/apt-vllm/bin/python -u scripts/run_v2_grpo_fine_closeout_followup.py --config configs/grpo-fine-closeout-followup-v1.json
)
~~~

每个模型的评测和回传打包均显示进度。预计 4 小时 45 分至 5 小时 20 分，墙钟上限 6 小时。中断后，在 `v2/` 给 Python 命令加 `--resume`。已提交完整评测的模型直接重放核验，尚未提交完整评测的模型重新采集。作业结束后回传终端 `Download` 指向的 review ZIP 和 `Receipt` 指向的 JSON。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo-fine-closeout-followup-v1.json | 主配置、阈值和预算 |
| grpo-fine-closeout-followup-branch-v1.json | 共用的评测模板 |
| grpo_fine_closeout_followup.py | 新任务族与冻结模型血缘 |
| grpo_repair_study.py | 独立任务检查和原生前缀 |
| grpo_common.py | 评测分支校验 |
| grpo_fine_closeout.py | 按实际分母判定主指标 |
| run_v2_grpo_fine_closeout_followup.py | 运行、配对、分阶段统计和审计 |
| build_v2_grpo_fine_closeout_release.py | 原生验证与发布核验 |
| test_grpo_fine_closeout_followup_v2.py | 权重、隔离、配对和阈值回归 |
