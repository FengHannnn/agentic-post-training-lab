# V2 Initial Implementation

日期：2026-10-29

V1已封存，V2独立目录已建立。主模型确定为Qwen2.5-1.5B-Instruct，主体为多轮Agentic RL；首项实现选择可验证的逐轮奖励分配与纠错。多教师OPD作为后续扩展。

## 已完成

- V1完整快照：2015个文件，约327MB，含代码、配置、数据、报告、实验产物、adapter、优化器和原模型缓存。
- V1归档每个成员均重读核验SHA256，ZIP CRC通过；原工作目录完整保留。
- V2任务环境：经营数据对账，读取业务口径、查订单/退款、计算当前与上期净收入、验证证据、提交结构化产物。
- SQL工具只读执行，具备查询预算、来源追踪和可恢复错误。
- 基于已验证进展的逐轮奖励、终局验收、逐轮回报与leave-one-out基线。
- 串行动作token的LoRA PPO更新内核，带old/new同路径检查、固定reference、TIS、裁剪和梯度检查。
- 1.5B模型下载、真实多轮基线、轨迹重放审计与限时回传入口。

## V1封存记录

归档：`C:/Users/app/Documents/Codex/archives/AgenticPostTrainingLab-v1-20261029.zip`

SHA256：`6348f933a916f976eb16670caff132a54cbf09710e99a3a7d86e2a73dfbec500`

原始内容约514MB，压缩后327327114字节。覆盖最新B/C结果及历史实验，包括保留的中间/部分产物；虚拟环境、Python缓存和Git对象数据库不打包。Git HEAD、工作区状态及diff另存元数据，源码工作区内容已完整包含。

最新实验候选C保留为历史候选；原冻结SFT基线及31步候选身份保持。V2从新1.5B模型开始，建立自己的参数、优化器和轨迹身份。

## 首个业务环境

任务要求按指定地区、客户分组和月份计算三项结果：
- current_net_cents
- prior_net_cents
- change_cents

口径按整数分计量：paid订单记收入，退款按实际发生月份扣除，原订单可以来自其他月份；取消订单与其退款排除。每个数字都要带工具证据，最终产物包含三项结果、证据ID和业务口径ID。

工具集合：read_policy、inspect_schema、query_sql、calculate_change、check_result、submit。

当前任务生成器提供train 12、development 4、test 4个不同数据题族。首次模型作业只跑development 4题，包含2题工具故障场景；test保留到方案冻结后使用。模型看到任务描述和工具观察，评分器另行维护答案与扰动数据。

程序参考策略在20题上完成对账，其中10题注入临时查询故障并完成恢复。这是环境与评分流程的验收结果；1.5B模型表现由下一次云端基线采集。

## 逐轮奖励如何工作

每个正确且有来源的指标记作一项已验证进展，状态势函数为：

`Phi(state) = 0.2 × 已验证指标数`

逐轮过程项采用相邻状态势函数之差。终止时势函数置零，使整条轨迹的过程项相互抵消；总回报仍由最终成功与工具成本决定。成功奖励1，每次工具操作成本0.01。训练内核使用逐轮return-to-go与同任务独立轨迹的leave-one-out基线。

这让有效查询与纠错拥有明确的中间信号，同时保留最终任务目标。重复查询或检查已有指标只产生工具成本；纠错不另发可循环领取的固定奖金。

指标认证还会在两组独立扰动数据上重放SQL：正确的查询逻辑随数据变化仍应得到正确结果。硬编码原答案即使读过表，也拿不到进展分。

## 模型与工程设置

- 模型：Qwen/Qwen2.5-1.5B-Instruct。
- 固定revision前缀：989aa79；下载时解析并记录完整commit及每个模型文件SHA256。
- 初始推理：BF16、vLLM、8192上下文、每动作最多256个新token、每episode最多12步、greedy基线。
- 保留原生prompt/action token与raw model logp；工具观察作为下一轮上下文。
- 基线greedy轨迹用于评测，训练入口接收随机策略轨迹及对应行为概率。
- 新LoRA配置起点为q_proj/v_proj、r=8、alpha=16；首轮云端作业采集模型基线，之后接LoRA训练作业。

训练内核已通过真实微型LoRA反向更新测试。完整1.5B训练将以本次基线中的格式成功率、工具成功率与故障恢复表现确定起点和数据需求。

## 云端执行

上传 `agentic-rl-v2-starter-20261029.zip`，然后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm &&
cd /root/rivermind-data &&
V2_STARTER_ZIP="$(find /root -maxdepth 5 -type f -name 'agentic-rl-v2-starter-20261029.zip' -print -quit)" &&
test -n "$V2_STARTER_ZIP" &&
python -m zipfile -e "$V2_STARTER_ZIP" /root/rivermind-data &&
cd /root/rivermind-data/AgenticPostTrainingLab/v2 &&
python scripts/run_v2_baseline.py --download-model --hourly-price 2.18
```

首次会下载约3GB模型文件，后续复用V2独立缓存。入口已自行初始化src路径，无需额外设置PYTHONPATH。

整次模型准备、评测与审计共享最多约30分钟和10元估算上限；按2.18元/小时，时间上限对应约1.09元。实例闲置、存储和打包另计；运行结束后手动停止闲置实例。

回传文件位于v2目录：

`v2-baseline-review-20261029.zip`

结果含模型revision、来源快照、完整episode、动作token、工具反馈、逐轮奖励、最终产物与重放审计。中断时保留已完成episode及当前partial episode，便于定位下一项调整。

## 读代码的顺序

1. `PROJECT_REQUIREMENTS.md`：用户已确认的模型、方向与报告风格。
2. `src/apt_v2/tasks.py`：业务口径、数据题族、参考答案与扰动世界。
3. `environment.py`：工具执行、证据认证、纠错与逐轮奖励。
4. `runtime.py`：真正的“动作→观察→下一轮”循环与中断证据。
5. `rewards.py / trainer.py`：逐轮信用、随机动作准入及LoRA更新。
6. `scripts/prepare_v2.py`：环境预检、模型下载和revision固定。
7. `evaluate_v2.py / audit_v2.py / run_v2_baseline.py`：模型基线、逐轮重放、限时打包。
8. `tests/test_agentic_v2.py`：业务正确性、只读约束、重复奖励、扰动校验、恢复与真实梯度。

## 验证结果

- V2完整测试：46 passed。
- 程序参考策略：20/20任务完成，10/10故障场景恢复，终局过程项抵消检查通过。
- 覆盖只读SQL、来源与扰动数据检查、重复奖励、失败回报、格式纠错、中断证据及审计篡改检出。
- 真实微型LoRA更新通过：仅adapter变化，冻结参数保持；缓存动作身份与随机策略概率准入通过。
- V1归档中的2015个原始文件与当前目录逐一匹配，V2开发未改变历史文件。
- 1.5B云端模型基线是下一次执行内容。

## 扩展边界

本阶段使用可复现合成业务数据和单一净收入口径，先建立短多轮闭环。后续扩展到真实脱敏数据、异常归因、多业务口径及更长轨迹，并比较终局奖励与逐轮奖励的成功率、恢复率和成本。教师OPD可沿用相同轨迹接口，作为独立消融项加入。
