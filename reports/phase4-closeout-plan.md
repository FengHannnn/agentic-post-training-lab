# Phase 4 一次性收尾安排与执行入口

日期：2026-10-21。用户选择暂停逐步读代码，优先一次完成 Phase 4 剩余工程工作。本地实现已完成，云端最终批次与回传验收尚待执行；不能提前标记 Phase 4 已完成。

## 还剩多少

按原始项目安排，Phase 4 主线是 GPU 环境与 rollout 引擎：资源、模型/adapter 接入、原生 token/模板/终止条件对齐、概率接口、吞吐和成本追踪。SGLang 是可选对照，TRL 需要另外决定；真实 GRPO 参数更新、KL reference、训练监控和效果对照属于 Phase 5，不作为本阶段收尾的新要求。

此前已完成 L40S 环境、HF/vLLM 原生 LoRA 接入、单序列重算、16 条随机采样、reward/产物审计、有限数值诊断、训练侧 FP32 无更新 ratio 和 token TIS 策略项。本阶段确实推进了部分 Phase 5 前置工作，但不再继续无限细分。

剩余合并成一轮云端批次和一次本地回传验收：

1. 固定 48 个算术训练输入×4 回答，检查批量 rollout 接口及奖励零方差情况。
2. 同一引擎热身后，在相同 16 个输入/seed 上各执行一次批量与串行请求，保存吞吐、时间和复现差异。
3. 对全部 192 条回答执行固定 HF FP32 old/new 评分、参数摘要不变检查、token TIS 与 mask/reward 审计。
4. 自动记录本次作业耗时与按量费用估算、源码/输入身份，生成一份 ZIP；取回后审计并写最终阶段结论。

无需重新租另一种 GPU、安装第二套推理引擎、换模型或继续调 DPO。

## 冻结实验与停止规则

configs/phase4_closeout.json 固定全部 case ID。输入只来自 dpo_v2 train，12 个数学 family，每个选 4 种 category 各 1 条；按 family 顺序轮换排序后的问法，最终 natural/sequence/symbolic 各 16 条，4 个 category 各 12 条。选择不查看 reward。保留重复回答与失败，不筛选成功样本、不使用 development、不修改 reward。

主批次 192 次生成，热身后的配对 benchmark 另 16+16 次，共 224 次请求；benchmark 回答不加入 reward 训练组。采样沿用 T=1、top_p=1、无 top_k 过滤、无惩罚、最大 64 token、并发上限 4、固定 seed。EOS 保留，length 原样记录并依据冻结奖励契约给分。

本轮最多约 30 分钟，费用估算额度取 min(10元,200元累计预算剩额)，为终止清理预留 10 秒。入口要求输入真实时价和截至启动时已花费用；超时终止整个阶段进程组，停止后不自动重采样或重试。估算只覆盖作业运行时间，平台开机等待、空闲、存储和计费取整另算，不能把该估算当完整账单或账户级强制限额。结束后须手动关机，平台实例不由脚本自动操作。

输入检查失败不会进入后续阶段。失败/超时后保留已生成文件，并尽可能打包错误与阶段退出状态。不删除失败数据，不因没有学习信号而自动追加样本。输出目录或回传 ZIP 已存在则拒绝复用。

## 数值结论与验收边界

旧 16 条数据的跨引擎 0.001 门继续失败，绝不回写通过。此次明确使用新身份 phase4-closeout-48x4-v1：保留 vLLM 实际 sampler logq，训练侧 old/new 采用原始 FP32 权重、eager 完整前向、无 cache、无 TF32；检查 new/old，无参数更新时阈值仍为 1e-5。

token TIS 为有偏近似，不能宣称无偏校正了序列或组联合分布。新契约接口通过也不证明旧跨引擎门通过。benchmark 分别保存同 seed 的 token 复现率和 logp 在旧 0.001 限内的比例；如有不一致如实报告，不提高阈值，也不宣称不同执行方式保证完全可复现。

若 48 组仍全部零方差：这是 Phase 5 训练信号不足的真实结果，不把它改写成 Phase 4 引擎接入失败，也不为追求非零信号无限采样。若出现非零组，只能说明本批有组内区分，不能证明策略已提升。training_admitted 始终 false，没有模型 backward/optimizer step。

收尾时分别判定“实验已完成”“新契约接口是否通过”“存在的限制”和“Phase 5 是否可训练”。如果出现数据/身份损坏或评分错误，先修复具体缺陷，不能仅为赶进度签收。

## 云端一次执行

将 phase4-closeout-update-20261021.zip 上传到 /root/rivermind-data，使用保存有原始数据与 trainer summary 的 L40S 实例：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase4-closeout-update-20261021.zip .
cd AgenticPostTrainingLab
python scripts/finish_phase4.py
```

入口会在终端询问两个数字：当前 L40S 每小时人民币价格、截至启动时项目租机累计已花金额。填写纯数字，不带货币符号。也可用 --hourly-price 与 --spent-yuan 显式传入真实值，不要随意填 0 绕过预算。无须为此反复来回确认实验配置。

脚本依次启动 generate、score、audit 独立进程，保证 vLLM 退出释放后再加载 HF。它不需要用户逐步复制命令。每个阶段有摘要，最终输出：

- 项目内 artifacts/trajectories/phase4-closeout-20261021-01/：原生记录、生成与 benchmark 明细、逐 token 训练分数、奖励审计、打包张量、源码快照、summary、cost_and_run。
- /root/rivermind-data/phase4-closeout-review-20261021.zip：可直接从 JupyterLab 下载的单个回传包。

下载 ZIP 后关机，并上传该包供本地最终验收。若提前失败，上传已生成的 ZIP 或终端错误，不直接重跑整批。

## 本地验证

真实冻结基线的 74 份快照和 8 份 base cache 校验通过，真实 tokenizer 已编码 48 个输入，最长 prompt 为 64 token，带 64-token 回答预算共 128，低于上下文 512。本地只校验冻结基线，没有伪称核验 Linux 原始 cloud bundle manifest，也未加载 GPU/执行真实新采样。

新增测试覆盖固定训练输入的 family/category/问法平衡与请求上限、原生 token/seed/目标/解码检查、重复回答保留、奖励重审、复现率分开统计、预算耗尽/异常拒绝，以及合成 192 条完整产物的 CPU 审计和篡改 mask 拒绝。云端 vLLM/HF 联合运行仍待实际执行。

新增 19 项测试通过，全套 python -m pytest tests -q 为 269 passed，无 skip，1 条已有 tiny PEFT 提示。入口语法编译通过。本地预检记录归档于 artifacts/evaluations/phase4-closeout-preflight/local_preflight.json；它只记录 CPU 验证，不包含真实新 rollout 或吞吐结果。

## 代码阅读顺序（本次可暂缓）

1. configs/phase4_closeout.json：48 个冻结输入与请求、时间、费用上限。
2. scripts/finish_phase4.py → main/run：一次执行、预算输入、阶段隔离、超时停止及自动 ZIP。
3. 同文件 generate：原生 192 条与热身后 16 条串行/批量配对；score：同序列 old/new、无更新与参数摘要。
4. src/agentic_post_training/rollouts/closeout.py → select_cases/validate_records/benchmark_summary/budget_limits：输入身份、mask、复现率与费用边界。
5. scripts/finish_phase4.py → audit：奖励与张量重算、新契约检查、阶段结果与训练准入分开。
6. tests/unit/test_phase4_closeout.py 与 test_phase4_closeout_audit.py：实际数据选择约束、异常拒绝、合成完整产物与 mask 篡改检查。
