# Phase 4：采样分布校正契约与策略损失接入

日期：2026-10-21。当前仍属 Phase 4。本轮实现独立的 token 级截断重要性权重与策略损失，并使用已有真实分数在 CPU 上完成代数检查。没有加载真实模型、重新采样或更新模型参数。

## 本次决策与公式

使用逐 token 的两层比值：

```text
w_t = stop_gradient(min(2, exp(old_logp_t - sampler_logq_t)))
r_t = exp(new_logp_t - old_logp_t)
L = -mean_eligible_tokens(w_t * min(r_t*A, clip(r_t,0.8,1.2)*A))
```

old 是更新前固定的 HF FP32 训练策略分数，new 是需要梯度的当前训练分数，sampler 是实际生成时的 vLLM behavior 分数。w 校正采样器与 old 的条件 token 概率差异；r 描述训练侧策略变化，两个比值不能混用。权重上限 2、PPO clip epsilon 0.2、无权重归一化，按有效 token 数取均值。EOS 属于真实回答；padding 和 greeting 不进入损失与分母。

选择这种解耦 token 权重，是为了在当前已有逐 token 分数接口上接入明确、有限的近似策略目标。该形式参考 [verl 官方 rollout correction 文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) 对 decoupled token IS 与截断权重的定义；本项目独立实现，没有引入 verl 依赖。上限 2 是本轮固定的工程配置，不是依据旧误差门反向调出的通过阈值，也没有证据证明它对本任务最优。

必须保留的限制：token 权重只处理给定已观测历史下的条件概率比，未校正前缀状态分布；截断会引入偏差，组内归一化 advantage 又依赖整组采样。不能据此声称实现完整轨迹或整个组联合分布的无偏重要性校正，也不能由少量已观测权重接近 1 推出完整策略分布相近。

本轮 beta=0，仅接通策略项，不实现 KL reference 或真实 trainer。原始 grpo_clipped_loss 保持不变，新模块独立实现，避免改动旧云端源码身份。新模块未来训练时需要实时计算的 new logp；保存的脱图分数只供本次接口审计。

## 实现与边界

- algorithms/rollout_correction.py：校正权重固定、拒绝带梯度的 old/q/advantage；在 log 空间先截断再 exp，避免大正差溢出。padding 和排除样本的 NaN/Inf 在计算前屏蔽；有效位置要求有限且非正 logp。无有效 token 则报错，不返回看似正常的零损失。
- corrected_grpo_policy_loss：权重乘在 PPO minimum 之后；正、负 advantage 的 clip 方向分别测试。有限性检查拒绝溢出的 ratio/损失。
- scripts/audit_rollout_correction.py：绑定已复核 summary SHA256，核对原始输入与评分源码，重建 token/mask/advantage，计算校正损失。对分数张量叶节点做 autograd 检查，未执行真实模型前向或模型反向，也没有优化器。

本步只允许已有完整 EOS rollout 的既定数据进入审计。未扩大输入或定义 length 截断的奖励/估计新规则；后续若采到 length，必须依据原 reward 契约处理并显式记录，不能将其伪装为 EOS。

## CPU 实际结果

输入是已取回的 trainer-probability-contract-v1/summary.json，SHA256=f3c5df02a15531ffc680f2d1363164724c7dd3aef4489a1f122d1935c8073d93。保留全部 16 条用于身份对齐，损失只用其中 12 条算术回答、325 个 token。

| 指标 | 结果 |
| --- | ---: |
| 最小 token 权重 | 0.9942519650 |
| 最大 token 权重 | 1.0209987130 |
| 平均 token 权重 | 1.0002122304 |
| 上限触发比例 | 0 |
| 校正策略损失 | 0 |
| 分数梯度最大绝对值 | 0 |

权重统计来自打包为 FP32 的保存分数，在双精度中计算比值。这里不是重新加载模型的概率重算。advantage 仍全为 0，因此损失与梯度为 0 是应有结果；重要性权重不能制造奖励信号。原跨引擎 0.001 门仍失败、原数据不覆盖；training_admitted=false。权重未触发截断仅描述这批数据，不构成后续任意 rollout 的保证。

结果归档于 artifacts/evaluations/rollout-correction-contract-v1/audit.json，含输入与源码哈希。CPU 可复现命令：

```powershell
.\.venv\Scripts\python.exe scripts/audit_rollout_correction.py --output-dir artifacts/evaluations/rollout-correction-contract-recheck
```

输出目录必须不存在。本轮已在本地完成，无需启动云 GPU、上传 ZIP 或安装依赖。

## 测试结果

新增 18 项测试，全套 python -m pytest tests -q 为 250 passed，无 skip，1 条已有 tiny PEFT 提示。覆盖：q=old 退化回原策略目标；正负 advantage 在上下 clip 边界的行为；固定单个历史、枚举两个动作验证未触发截断时的条件重要性恒等式；非单位权重的有限差分梯度核验；零 advantage 的零梯度；padding/greeting 排除；极端权重先截断；非法梯度、shape、mask 和分数拒绝。

条件两动作测试只证明该固定历史下的代数恒等式，不代表序列/组级无偏性测试。

## 代码阅读顺序

1. configs/rollout_correction_contract.json：先理解 cap、clip、token 均值、beta=0 和不准入训练的范围。
2. src/agentic_post_training/algorithms/rollout_correction.py → token_tis_weights：old/q 的意义、detach、先截断再指数、有效位置 mask。
3. 同文件 corrected_grpo_policy_loss：与旧 grpo_clipped_loss 对照，重点看 w 与 r 分工、minimum 对正负 advantage 的作用及分母。
4. tests/unit/test_rollout_correction.py：优先阅读 test_equal_sampler_reduces_to_original_grpo_policy_objective、test_weighted_positive_negative_advantages_and_policy_clipping、test_autograd_matches_finite_difference_with_fixed_nonunit_weights，再看单历史恒等式和异常测试。
5. scripts/audit_rollout_correction.py → main：绑定真实输入、重建 mask、分数叶节点 autograd、实际零信号的解释；注意没有模型或优化器。

## 下一步

概率校正近似与代数接口已明确。下一小步可在固定 reward、采样温度与分组大小的条件下，冻结一个有上限的算术训练输入清单，扩展随机 rollout 并统计非零方差组；同时保存原 sampler logq 和训练侧 old 分数。不得按 reward 挑选回答、去重、修改 reward 造信号或重新标记旧门通过。新产物使用新契约身份，实际训练准入仍需单独检查，Phase 5 尚未启动。
