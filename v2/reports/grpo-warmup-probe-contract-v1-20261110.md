# Warm-up probe v1 冻结契约

本轮使用既有W、旧B24和新B15，在同一批train前缀上比较实际首动作、别名条件分数及维护动作。run_id为grpo-sql-warmup-probe-v1-20261110；本轮优化器更新为0。

上一轮结果见 reports/grpo-sql-retention-results-v1-20261110.md。旧B24的development目标成功8/12，新B15为0/12；当前先直接检查训练起点上的目标动作和关键token，再固定后续目标重复次数的对照。

## 模型、任务和预算

| 模型 | 血缘 | adapter SHA256 |
| --- | --- | --- |
| W | SFT513 / DPO11 / RL0 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| 旧B24 | SFT537 / DPO11 / RL0 | d81a31b392ca5e4e043a43aa2c8fa75ee2968e97b09cbe857f732d6e22dd5226 |
| 新B15 | SFT528 / DPO11 / RL0 | 38a9a6d68ed1a0ea0620aaa726ca6443fe701691a88c533634761663c6a72b31 |

模型输入、原生tokenizer和任务计划冻结在 artifacts/inputs/grpo-warmup-probe-20261110/。三模型使用相同基座revision与tokenizer，均记录worker实际加载的112个BF16 LoRA张量证明，并由CPU独立投影重算。

两轮累计已用9500.091520188143秒。本诊断所有attempt累计最多1800秒，包含准备、推理、审计和实际打包；两轮加本次的最大合计11300.091520188143秒，处于既有21600秒总预算内。外部receipt记录本轮与合计耗时，resume复用已提交轨迹并累计已耗预算。

## 固定诊断计划

每模型24个train起点，三模型使用完全相同的消息和seed：

- 12个原复合缺别名起点，覆盖全部12个train任务；各做2次同seed greedy首动作，核验原生响应token重复一致性。
- 12个维护起点，每个train任务1个；旧证据修复、失败检查修复、SQL错误恢复、检查后提交各3个，且每种状态同时覆盖旧/新train池。

合计108次首动作生成。温度0、原生上下文8192、响应上限256；恢复前缀计入12次总动作预算，每条生成在真实环境中执行一次并逐条重放。实际greedy生成与teacher-forced正确动作分别记录。

目标首动作有效定义为新增本月或上月认证；旧证据维护要求新增差额认证；SQL恢复要求新增月度认证；失败检查要求正确检查通过或有效提交；已检查状态要求有效提交。所有统计以实际执行结果为准。

## 关键token分叉诊断

每个目标起点构造两条执行可核验的响应：正确query_sql，及仅删去AS current_net_cents的query_sql。正确响应产生认证，删去别名的响应产生未认证查询。

原生token序列在共同SQL前缀后分叉：输出AS的首token与直接收尾SQL的首token分别记录，分叉索引129或135。关键别名片段为5个token。两响应均来自同一公开train前缀。

每条候选序列使用vLLM原生prompt_logprobs，在T=1、无截断条件下做teacher-forced prefill打分。保存响应token及对应logp、别名片段token均值NLL、完整响应token均值NLL，以及正确分叉token减省略分叉token的logp差。每次打分额外生成1个token并计数，共72次打分请求、72个附带生成token。

这些分数作为prefill条件分数诊断，与实际decode认证结果并列展示，独立于训练行为概率q和策略更新。后续配方以实际动作执行认证为主要依据。本轮输出三模型逐任务对照，正式基线仍为W。

## 提交、恢复和审计

每个起点在生成、反事实打分、原生token及真实执行验证完成后原子提交；session绑定模型、配置、prepared输入及GPU身份。中断保留原始请求和部分动作，恢复验证并复用已提交起点。

完成的模型封存轨迹、session证明及原始请求哈希。根审计独立重放72个起点的108次动作，复算72条打分记录及所有汇总。已提交统计与所有完整请求日志计数分别保存，重试开销单列。

云端review保留三个adapter、tokenizer、输入、原生token、分数、执行记录、worker证明、冻结源码、日志和审计。完整诊断结束后回传review ZIP及外部receipt，再据结果冻结目标重复次数的单变量实验。

## 本轮代码阅读顺序

| 文件名 | 大致作用 |
| --- | --- |
| grpo_warmup_probe.py | 固定train计划、真实执行、原生分叉及汇总规则 |
| grpo_warmup_probe_worker.py | 三模型推理、条件打分、轨迹提交与独立审计 |
| run_v2_grpo_warmup_probe.py | 30分钟调度、断点恢复和累计预算打包 |
| validate_v2_grpo_warmup_probe.py | 专家、反事实、token和输入身份验证 |
| build_v2_grpo_warmup_probe_release.py | ZIP唯一条目、CRC和逐文件哈希核验 |
| grpo-sql-warmup-probe-v1.json | 三模型、固定计划和总预算配置 |
| test_grpo_warmup_probe_v2.py | 原生分叉、执行、模型证明及中断恢复测试 |
