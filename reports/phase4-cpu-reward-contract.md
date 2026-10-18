# Phase 4：CPU reward 契约与离线审计（2026-10-18）

## 路线与范围

当前仍处于 Phase 4（GPU 环境与 rollout 引擎）。用户选择在租用云 GPU 前，先完成 CPU reward 契约与审计，作为 rollout 到训练之间的接口准备。原路线将 verifier/reward 与 GRPO 列在 Phase 5，本步提前准备其中的奖励接口，不代表进入或完成 Phase 5。Phase 4 的 GPU 引擎接入仍待进行。本步不训练、不加载模型、不重新采样，不修改冻结基线、旧评分器或历史 rollout。

## 契约 arithmetic-strict-v1

目标是当前算术请求的严格结构正确性，不是一般数学语义等价。每条 completion 的奖励只有 0 或 1，无格式部分分、长度奖励或人工补分。

- 仅 expected_type=tool_call 的算术题 eligible=true。
- 正常 EOS 结束、严格 JSON/schema/calculator 路由合法、表达式安全执行、参考 AST 匹配且结果正确，奖励 1；其他情况奖励 0。
- finish_reason=length 即使文本已构成正确 JSON 也为 0。未知结束原因报错；不自动补全或修复文本。
- 空白和冗余括号由现有 AST 规则忽略，操作数顺序仍有意义。
- 问候保留诊断但 eligible=false、reward=0。后续 trainer 必须依据 eligible 排除整组；0 分不表示问候回答错误。现有“非空 final”不足以证明内容质量。
- strict_pass 与 semantic_pass 是原有评测诊断，未包含本契约的截断和范围门，不能直接当训练奖励。

采用严格目标是保守的第一版选择：它拒绝把预先算出的答案常数作为表达式的捷径，也漏奖交换操作数等数值正确的表达式。数值评分只验证一个输入，可能奖励常数或巧合结果，不能自动作为符号等价证明。若以后希望奖励等价表达式，应单独定义和版本化验证器，不能只为制造方差放宽规则。本契约不能证明模型的内部推理过程正确，也不是通用防奖励投机方案。

## 审计实现与边界

`rewards/contract.py`独立封装规则；`rewards/audit.py`重算评分，验证目标、组内输入、编号、EOS/logp记录结构，调用现有长度打包和组内优势函数。错误与重复回答原样保留。组内使用总体标准差（unbiased=False），std<=1e-8 时优势为零；其他组 advantage=(reward-mean)/std。输出按 sample_index 对齐。

`scripts/audit_rewards.py`读取原始 JSONL，输出一个新 JSON，记录输入和相关源码 SHA256；文件已存在则拒绝覆盖。本审计重算奖励但不加载 tokenizer 或模型，不重新证明 text 与 token 的解码一致性、不验证概率值、不重新证明数据来自训练 split。未来训练接入必须继续运行原 rollout 身份/解码审计及训练数据身份检查，不能用本报告替代。源码哈希是本次身份记录，不是对历史采样身份的重新验证。

没有创建 trainer；eligible 门、old/new/reference 的冻结与 detach、采样分布校正、KL 和 ratio 契约仍须后续实现。零 advantage 只说明奖励策略项没有相对信号，不表示未来带 KL 的总目标必然无梯度。

## 代码阅读顺序

以下路径相对项目根目录。先理解单条奖励，再看分组审计和命令入口，最后用测试检查边界。

1. `src/agentic_post_training/rewards/contract.py` → `score_reward`：本次首先阅读。看清 eligible、finish_reason、严格通过如何共同决定 reward，以及 reason、semantic_only 的含义。
2. `src/agentic_post_training/evaluation/tool_calls.py` → `score_output`：已有依赖，按需回看。确认 JSON/schema、路由、表达式执行、AST 和结果检查的先后关系。
3. `src/agentic_post_training/rewards/audit.py` → `audit_records`：理解记录重建、同组目标与输入检查、旧评分重算、sample_index 排序，以及奖励与 advantage 的逐条对齐。
4. `src/agentic_post_training/algorithms/losses.py` → `group_relative_advantages`：已有依赖，重点回看总体标准差、零方差分支，并对应 `[1,0]` 与 `[1,1]` 两组奖励的优势。
5. `scripts/audit_rewards.py` → `main`：串起 JSONL 读取、调用审计、输入/源码哈希和新报告写出，理解哪些身份检查由本入口完成。
6. `tests/unit/test_reward_contract.py`：先读 `test_reward_counterexamples`、`test_truncation_and_greeting_scope`，再读 `test_signal_and_order`、`test_reject_corrupt_records`，验证误奖、漏奖、组内信号与坏记录边界。

阅读后对照本报告下方的四组结果，并查看 `artifacts/evaluations/reward-contract-v1-20261018/audit.json` 中的 groups 和 records。无需重新采样即可理解本步流程。

## 现有样本实际结果

输入：`artifacts/trajectories/rollout-smoke-phase4-20261017-01/rollouts.jsonl`。

输入 SHA256：`d502be18043e2433a887daa85f2e39673bf1ea2f808c7c6cacd55ac79f6e9df3`。

输出：`artifacts/evaluations/reward-contract-v1-20261018/audit.json`。

| 组 | eligible | 奖励 | advantage |
| --- | --- | --- | --- |
| symbolic | 是 | 1,1,1,1 | 0,0,0,0 |
| natural | 是 | 0,0,0,0 | 0,0,0,0 |
| sequence | 是 | 0,0,0,0 | 0,0,0,0 |
| greeting | 否 | 0,0,0,0（占位） | 0,0,0,0（排除） |

16 条记录审计通过，12 条算术 eligible，奖励合计 4，3/3 eligible 组为零方差（100%）。1 条回答仅语义通过。不存在 reward 相对学习信号；审计通过不等于可以用本批数据开展有效 GRPO 更新。

## 反例与验收

新增 23 项测试覆盖正确表达式、空白/冗余括号、交换操作数、答案常数、算术错误、除零、非法调用、空文本、代码围栏、错误路由、重复 JSON 键、NaN、错误工具、多余字段、坏 JSON、截断、问候范围、未知停止原因、混合奖励的正负优势、重复回答零优势，以及目标/评分/编号/输入/EOS 篡改。

2026-10-18 实际验证：新增测试 23/23 通过；全套 160/160 通过，无 skip，保留 1 条已有 tiny PEFT 配置提示。离线审计退出码 0。没有重新运行历史训练或采样。

复核命令（使用新的输出文件名，避免覆盖）：

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_reward_contract.py -q
& .\.venv\Scripts\python.exe scripts/audit_rewards.py --input artifacts/trajectories/rollout-smoke-phase4-20261017-01/rollouts.jsonl --output artifacts/evaluations/reward-contract-v1-review/audit.json
```

本步完成标准是规则可执行、反例通过、奖励与优势可审计，并如实发现无学习信号；不是要求模型成绩提升。没有必要为了本步收尾重复采样。下一步准备云 GPU 具体资源与成本方案，随后小规模采样观察奖励分布；若严格奖励持续稀疏，单独审阅训练题选择或新验证器，不静默改奖励。
