# V2 SFT / DPO Results Handoff

更新：2026-11-09，Asia/Shanghai。文件1：阶段结果交接。下一对话以同目录文件2 `PROJECT_HANDOFF_V2_GRPO_CONTINUATION_2026-11-09.md` 为执行入口。

## 1. 阶段状态与使用方式

SFT/DPO 阶段已经收尾，后续主线是 GRPO。正式 GRPO 共同起点仍为 W；短程 DPO12 保留为 DPO 阶段成果，不替换 W。V1 已封存，不进入本轮算法迭代。

本文件提取接续必需的结果、身份与证据入口。逐实验完整表格、轨迹和原始审计继续引用既有产物。历史文档中的“下一步”是当时安排；当前待办以文件2和最新用户指示为准。

原交接来源：项目 `handoff/PROJECT_HANDOFF_V2_GRPO_2026-11-05.md`。本次确认桌面 `C:/Users/app/Desktop/后训练项目/PROJECT_HANDOFF_V2_GRPO_2026-11-05.md` 与项目副本字节一致。用户最初给出的桌面根目录位置现已迁移。

## 2. 基本配置与项目定位

| 项目 | 固定信息 |
|---|---|
| 项目目标 | 面向大模型后训练 / RL 算法实习，其次兼顾 Agent 开发；形成可复现的训练、恢复、信用分配与审计案例 |
| 本地 V2 根目录 | `C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2` |
| 本地 Python | `C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/.venv/Scripts/python.exe` |
| 云端 V2 | `/root/rivermind-data/AgenticPostTrainingLab/v2` |
| 云端 Python | `/root/rivermind-data/envs/apt-vllm/bin/python` |
| 模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 固定 revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 云端模型缓存 | `.cache/qwen2.5-1.5b-instruct`，相对云端 V2 |
| LoRA | q_proj / v_proj；r=8，alpha=16，dropout=0；112 个张量、1,089,536 个可训练参数 |
| 推理 / 训练 | vLLM BF16 生成；HF/PyTorch/PEFT FP32 评分与训练；项目自实现训练逻辑 |
| 常用环境 | Python3.11.17、Torch2.8.0+cu128、CUDA12.8、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1；每轮读取自身 execution_environment |

任务是多轮 SQLite 月度业务对账：读规则和 schema，查询已支付订单净收入、扣除按发生月份记账的退款，取得当前/上月两个指标，计算差额，检查后提交带证据的结果。金额为整数 cents。

原短任务为 train12、development4、test4，总动作预算12；约一半实例注入一次临时查询故障。工具为 read_policy、inspect_schema、query_sql、calculate_change、check_result、submit。模型每次生成严格 JSON 工具动作，环境实际执行并登记证据。SQL 执行成功与指标认证成功分别记录。专家示范与数据构造详情引用原交接第4–5节和相应代码。

## 3. 主线模型履历与身份

```text
固定 Qwen 基座
  → SFT pilot63
  → SQL weighted SFT126：S0（SFT126 / DPO0）
  → 首轮 DPO11：D11（SFT126 / DPO11）
      ├─ 恢复 DPO43 / 129：独立对照分支
      └─ chosen-only SFT129：P（SFT255 / DPO11）
          ├─ matched DPO258，以及 E1–E4 机制排查分支
          └─ 困难加权 SFT258：W（SFT513 / DPO11 / RL0）
              ├─ uniform 短程 DPO12：SFT513 / DPO23
              └─ 后续各 GRPO 实验分别从 W 独立初始化
```

“最佳加权 SFT W”是阶段名称；W 继承11次历史 DPO。P 也继承11次 DPO。历史纯 SFT126 基线独立保留。DPO12 表示该短程阶段新增12次更新，累计 DPO23。

| 模型 | adapter_model.safetensors SHA256 |
|---|---|
| S0，纯 SFT126 | `e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0` |
| D11 | `fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d` |
| P | `078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55` |
| W | `f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f` |
| 原 matched DPO86 | `43100acde91f178de59ac7a65f29867862a02a7f1a6cbcd134b7d81a7318f46c` |
| 最终短程 DPO12 | `77c737965c25dfd3110177bb0856938a651cb387bc7153b588b3ce56adf6fc7e` |

实际权重入口（均相对 V2）：

- W 本地原始来源：`artifacts/reviews/hard-recovery-sft-weighted-v1-20261103/extracted/hard-recovery-sft-weighted-v1-20261103/final/adapter`；GRPO 冻结输入为 `artifacts/inputs/grpo-w-20261105/adapter`。
- W 云端来源：`artifacts/hard-recovery-sft-weighted-v1-20261103/final/adapter`。
- DPO12 本地：`artifacts/reviews/dpo-sft-start-short-v1-20261105/extracted/dpo-sft-start-short-v1-20261105/training/checkpoint-0012/adapter`。
- DPO12 云端：`artifacts/dpo-sft-start-short-v1-20261105/training/checkpoint-0012/adapter`。
- 候选登记：`artifacts/candidates/<run_id>/review.json`；该目录通常是登记，不等于实际权重位置。
- 纯 SFT 基线索引：`artifacts/baselines/current_sft_baseline.json`。

## 4. 数据与训练目标的交接要点

SFT 原始36条专家/SQL修复/证据修复成功轨迹，252个准入动作。错误动作留在上下文，正确准入 response 作为监督目标；prompt labels=-100，监督响应 token CE。SQL 加权阶段 query_sql 原始权重4，其余1，均值归一。

恢复偏好池固定169行：原44对 + 脚本失败84对 + 原生模型失败41对；chosen 是同状态下经执行验证的纠正动作。169行中有123组不同 context/chosen，重复行用于匹配数据分布。临时服务故障单列，不当作模型偏好错误。

- 169对数据 SHA256：`c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329`。
- 原44对 SHA256：`508464a0314982c7e6ade95c57afa3be081afc80f237e4fb20c8abe2079ffb78`。
- 数据实际来源：`artifacts/reviews/dpo-recovery-collect-v1-20261102/extracted/dpo-recovery-collect-v1-20261102/pairs.jsonl`。
- 困难加权：sql_unverified_requery、repeat_evidence_error、repeat_sql_error、failed_check_repair，共84行权重4，其余85行权重1；归一后困难行约占总权重79.8%。
- DPO：chosen/rejected 响应 token logp 分别求序列和，比较相对冻结 reference 的 margin，再求 `-logsigmoid(beta * margin)`。
- chosen-only / weighted SFT：只对 chosen 做 token CE；rejected/reference 不进入该 CE。

详细构造、模板、EOS、权重分布和数据隔离见原交接第5节及 `reports/dpo-recovery-collect-v1-review-20261102.md`。

## 5. SFT / DPO 结果梗概

下表每一行使用各自同轮对照。早期“恢复首动作有效”与后期“完整恢复终局成功”分开解释。

| 实验 / 改动 | 主要观察 | 结果入口（相对 V2/reports） |
|---|---|---|
| SFT pilot → SQL weighted SFT | 基座 dev0/4；pilot学会格式；SQL加权后 train12/12、dev4/4，历史test greedy4/4 | `sft-pilot-v1-review-20261030.md`；`sql-sft-weighted-v1-review-20261101.md` |
| 首轮 SQL DPO，44对、11步 | 同轮train完整26/48→28/48；独立test12/32→13/32 | `dpo-train-v1-review-20261102.md`；`dpo-independent-test-v1-review-20261102.md` |
| 恢复 DPO，169对 | 1遍完整10/16→12/16；3遍完整11/16→11/16，首动作修复略有改善 | `dpo-recovery-train-v1-review-20261102.md`；`dpo-recovery-train-v2-review-20261102.md` |
| chosen-only SFT | 同轮完整均11/16，恢复首步D11/DPO/SFT为29/84、35/84、37/84；形成P | `chosen-only-sft-v1-review-20261102.md` |
| 困难加权SFT，从P继续258步 | 完整11/16→15/16；恢复首步37/84→74/84；形成W | `hard-recovery-sft-weighted-v1-review-20261103.md` |
| W独立任务评测 | 相对P完整采样24/32→28/32；首步36/84→66/84；复合完整恢复均0/32 | `hard-recovery-independent-test-v1-review-20261103.md` |
| matched DPO，从P，hard4、258步 | W/DPO完整16/16对0/16，完整恢复59/84对13/84；长期更新发生退化 | `hard-recovery-dpo-matched-v1-review-20261103.md` |
| E1：检查更新阶段 | DPO43/86/129/172/215/258完整10/13/11/10/1/0（各/16） | `dpo-regression-checkpoints-v1-review-20261104.md` |
| E2：hard4→uniform | 同轮恢复15/84→33/84；长程uniform仍落后W | `dpo-uniform-control-v1-review-20261104.md` |
| E3：DPO加chosen CE | 同机纯加权→加CE，完整0/16→10/16，恢复10/84→37/84 | `dpo-chosen-ce-control-v1-review-20261105.md` |
| E4：chosen CE→完整续跑CE | 完整恢复36/84→60/84，greedy14/28→25/28；完整任务8/16→7/16 | `dpo-continuation-ce-paired-v1-review-20261105.md` |
| 旧DPO86直接比较 | W/DPO86完整16/16对12/16，恢复60/84对51/84 | `dpo-early86-vs-weighted-sft-v1-review-20261105.md` |
| 最终W起点短程DPO12 | 达到运行前固定的阶段收尾容差，见下一节 | `dpo-sft-start-short-v1-review-20261105.md` |

机制证据与各控制变量解释集中在 `reports/dpo-control-experiments-overview-20261105.md` 和原交接附录A。主要结论是：偏好 loss / margin 与任务成功需分别监控；正确动作保真和完整恢复续跑覆盖各有作用；本项目最后采用强起点、小学习率、uniform和预定短程停止的组合收尾。

## 6. 最终 DPO 收尾结果与边界

作业 `dpo-sft-start-short-v1-20261105`：W 同时作策略起点和新 reference，169对、uniform、LR5e-7、beta0.1、batch4、grad clip1，无辅助CE；预定4/12/24步检查，连续保留Adam/RNG，达到首个合格节点即停止。

| 同轮指标 | W | DPO4 | DPO12 |
|---|---:|---:|---:|
| 完整 sample | 16/16 | 16/16 | 16/16 |
| 完整恢复 sample | 62/84 | 60/84 | 59/84 |
| 完整 greedy | 4/4 | 4/4 | 4/4 |
| 完整恢复 greedy | 20/28 | 19/28 | 21/28 |

预定收尾门槛：完整sample至少15/16且最多落后W1次；sample恢复最多落后4/84；两项greedy成功数不下降。第12步合格，24步未运行。该阶段目标是“比W好或接近”，结论为保持完整任务表现、恢复处于既定容差内。GRPO后续目标另为最终超过W。

最终新增12更新、48对曝光；adapter真实变化，整体相对W L2约0.0273%。本地复核396条轨迹/1919个动作及输入、原生token、概率、配对与选择；云端保存Adam/RNG状态，本地回包不含state.pt。完整审计路径见该报告及 `artifacts/reviews/dpo-sft-start-short-v1-20261105/`。

## 7. 阶段安排与留存边界

- SFT、DPO、E1–E4已完成，作为结果资料保留。下一对话不自动重启DPO搜索。
- 四分支末期推理确认 `dpo-inference-confirmation-v1-20261105` 仅准备过，未运行，保留为可选扩展；E4的多LoRA BF16分叉机制仍可专项研究。
- 历史test-000～003已多轮使用，归为历史回归；新的最终确认应在模型/协议确定后冻结新任务。
- 多轮评测使用独立模型进程、单LoRA编号1、规范化配置；同一seed跨机器/数值路径的输出变化按新运行记录处理。
- 报告英文文件名、中文正文、结果先行，避免“不能说明”等防御性表达；不自动打开。后续代码阅读清单只列新增/改动文件及简短作用。
- 本文件是结果交接，不授权新的云端预算。GRPO完整进展、代码来源及未完成项见文件2。

## 8. Suggested skills

- `handoff`：继续更新交接，路径 `C:/Users/app/.codex/skills/handoff/SKILL.md`。
- `grilling`：用户要求 grill-me 或讨论重大新实验边界时使用，路径 `C:/Users/app/.codex/skills/grilling/SKILL.md`。
- `diagnosing-bugs`：复核报错、性能或数值异常时使用，路径 `C:/Users/app/.codex/skills/diagnosing-bugs/SKILL.md`。

下一代理按任务需要调用对应技能；不重复询问已经确认的预算和既有事实。
