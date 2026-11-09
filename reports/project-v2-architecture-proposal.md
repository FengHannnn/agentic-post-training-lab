# Project V2 Architecture Proposal

Date: 2026-10-29
Status: Proposed architecture; no new model downloaded, no cloud resources started, no historical checkpoint modified.

## Direction

建议将135M阶段封存为v1工程基线，v2升级为面向可验证经营数据分析任务的多轮Agent训练。模型容量、长程环境与训练信号分别做实验，不把135M阶段的显著提升作为新架构前置条件。本文件是设计建议，具体模型、业务场景和新预算仍待用户决定。

当前证据不能证明瓶颈只来自参数量。此前还观察到示范分布偏移、奖励稀疏、自然语言运算关系错误以及重复查看开发集的局限。升级模型合理，但必须先测新模型起点。

## Model migration

推荐主候选Qwen/Qwen2.5-1.5B-Instruct，0.5B同系列作为快速测试或容量对照候选；此选择考虑工程迁移和同系列对照，不声称它是当前最强模型。

单纯换模型属于中等工作量，已有SFT/GRPO数学核心、序列化轨迹、评分审计与保存恢复经验可复用。需要重新建立：
- 模型与revision、tokenizer、chat template、EOS/PAD/停止token、LoRA目标与张量数量；
- 新的SFT或原始Instruct基线、reference及不可变来源清单；
- 新rollout、模型/行为/old/reference概率与同路径无更新检查；
- 新Adam状态与输入绑定；原135M adapter、Adam、缓存logp不能迁入新模型；
- GPU显存与吞吐、FP32/eager或新精度的数值检查、预算。

原代码有120个LoRA张量、230400参数、模型摘要、64输出token与512上下文等固定约束。对新模型应参数化配置并重新验收。135M与Qwen跨系列比较不能把差异完全归因于参数量；0.5B与1.5B同系列比较也应明确预训练差异的限制。

仅1.5B policy/reference两个FP32权重约12GB（按1.5e9参数估算），不是总训练显存；激活、词表logits、KV cache和教师还需要额外空间。先在现有L20上实测，避免同时常驻多教师。新精度方案要重新做数值检查。原200元预算不能直接视为新多教师长程实验预算。

## Task environment

建议首个落地场景：有证据引用和错误恢复的经营数据核对与分析Agent。

示例请求：读取订单、退货、库存与口径说明，核对某月净销量或净销售额变化，找出异常来源，提交附有可复核数字与来源的分析结果。

工具初始集合：
- inspect_schema / read_reference
- query_sql / run_calculation
- validate_result / inspect_error
- save_artifact

模型负责动态选择查询、处理观察结果、识别错误、修改方案与决定何时提交；评分面向最终产物及可验证状态变化。先做短多轮任务，再延长决策链。优先可重置、可复现、可程序验证的本地环境，不从开放互联网和自由文档创作起步。

定义任务族隔离的新train/dev/hidden-test。最终指标包括任务成功率、数值与来源正确率、故障后恢复率、无效操作、工具/模型token成本和预算内成功率。独立隐藏测试集不参与逐轮调参。

## Runtime and trainer boundaries

Task specification -> Agent runtime -> tool/environment transitions -> trajectory store -> verifiers and credit assignment -> training backend -> evaluation

轨迹需包含episode、turn、工具调用、观察、环境状态/快照、终止原因、策略版本、token与行为概率，以及可回溯的奖励来源。训练mask只覆盖模型生成的动作/回答，工具观察不作为模型采样token参与策略梯度。

保留最终任务验收作为锚点。中间信号按实际检查和环境进展提供，支持恢复到检查点、有限重试与重新采样，不奖励重复检查、无效调用或仅形式完整的错误。部分进展与最终成功的权重和门控需做消融，避免复现v1分级奖励问题。

Agentic RL属于较大架构改动：多轮状态管理、工具执行、错误恢复、动作mask、长程rollout调度和状态级评分是主体工作量。参考Agent Lightning的runtime/trainer解耦思路，训练后端先评估成熟框架，保留v1作为小型可审计对照。

## Three different training signals

1. Outcome reward: 最终任务是否正确完成。继续保留。
2. Process / turn credit: 哪次检索、查询、验证或修复改善了可完成性。先使用可执行里程碑，再研究状态价值差或有限分支反事实估计。
3. Token-level MOPD: 在学生自身访问的前缀上，教师提供token分布指导，路由器选择或组合教师信号。它与环境奖励和推理时工具路由分别实现。

TRACE是2026年turn-level信用分配研究之一，利用冻结reference对gold-answer的logprob构造状态价值变化。自由知识工作产物通常没有唯一gold answer，不能直接套用其标量答案设定；本项目先从有可验证结构化目标的任务开始。

MOPD-Router研究在token粒度分配多教师指导，不要求部署时每个token都切换教师。初始先单教师OPD；有稳定增益后比较按任务路由、多教师均值与token路由。教师必须通过对应任务能力验证。标准token分布混合要求兼容token空间；不同词表/分词不能直接逐token平均。只有文本输出的接口不能替代所需前缀上教师概率。

## Proposed experiments

1. 新模型零训练/少样例任务基线，检查任务是否处于可学习难度。
2. 固定模型和环境，对比Instruct/SFT、最终奖励RL。
3. 固定模型与数据，对比最终奖励RL、加入可验证过程信号、加入单教师OPD。
4. 固定其余设置，对比多教师按任务路由、均值、token路由。
5. 以新任务族及隐藏测试检查泛化、错误恢复、成本与reward hacking。

单次实验只引入必要变量；不同时升级模型、任务、奖励与多教师后把所有收益归于某一种算法。

## GDPV clarification

用户已确认GDPV指GDPval。GDPval是面向真实职业工作产物的评测基准，可用于任务与验收设计；它不是逐token奖励或纠错算法。不能把自建的小型经营分析环境称为完整GDPval复现。

## Primary references

- Qwen 0.5B model card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Qwen 1.5B model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- GDPval paper: https://arxiv.org/abs/2510.04374
- MOPD-Router: https://arxiv.org/html/2609.30837v2
- TRACE: https://arxiv.org/abs/2607.13988
- Agent Lightning v1.0: https://arxiv.org/abs/2608.17528
- verl OPD implementation documentation: https://verl.readthedocs.io/en/latest/algo/opd.html
