# Phase 5: Balanced Demonstration Warm Start Preparation

状态：按用户授权调整示范组成，新云端运行版本已实现；本地验证见文末。本轮135M云端训练尚未执行，没有新的效果结论。保留原31步候选，原失败热启动与全部历史文件保持原样。

## 本轮调整

上轮24条示范虽正确，但没有任何以左括号开头的表达式。热启动后，实际24题验证正确13→14，原48题诊断38→35，开发56题44→39；8条正确转错均删除了必要括号。此次先检验示范覆盖假说。

| 项目 | 上轮 | 本轮 |
| --- | --- | --- |
| 示范组成 | 原24条 | 原24条 + 12条left_grouped |
| 新增题来源 | 无 | 原4个训练family，各symbolic/natural/sequence三种问法 |
| 每条呈现轮数 | 3 | 2 |
| 总呈现例次 | 72 | 72 |
| 每批例数 / 更新步数 | 4 / 18 | 4 / 18 |
| SFT学习率 / seed | 5e-5 / 202610300 | 相同 |
| 监督回答token总计，含EOS | 2016 | 2040 |

新增示范例如 `(47+20)*12`，全部经执行与结构验证。原24条回答各28token，新12条各29token。训练例次与更新数匹配，监督token增加24（约1.19%），不是严格等token对照；新数据组成与每题重复次数的变化共同构成本轮干预，不能据此声称独立分离了这两个因素。

只使用原训练数据。开发集没有加入训练，也不按开发集具体数字编写新示范。36条各呈现两次，顺序由相同seed生成。

## 起点、评测与后续流程

- 起点为原31步GRPO候选A，完整参数SHA256：
  `70804ab1c9344f7578396e4ccaa90dfe1ea374c69f96cb69ee3970d036eedf7d`。
- 不延续上轮失败热启动B的权重或SFT优化器。SFT仍用独立的新AdamW，训练目标仍为assistant回答与真实EOS的响应损失，只有LoRA可训练。
- A、B（本轮热启动后）、C（本轮GRPO后）都评测原24题、原48题诊断集、原56题开发集；另外列出新增12条括号示范题。每模型140条套题记录，套题有重叠，不合并当作140道独立题。
- 原24题用于与上轮直接对比。新增12题属于训练拟合与保留能力诊断，不计入原24题，不能当作泛化收益。
- 行为验收规则保持：开发严格或验证正确数下降即停止；各评测套题出现新增结构/路由失败也停止。新增12题的评测采用相同结构检查，不新增数学正确数门槛。
- 热启动验收通过后，自动按已授权流程由B重新采样原24题×32=768条，仅正确性二元混合组进行一遍有限GRPO，最多6步。
- 不把新12条示范题加入GRPO rollout范围。没有复用上轮旧回答，也不会把chosen答案伪装为策略采样。
- GRPO延续原31步Adam动量，SFT动量不导入；reference仍为原冻结SFT。原KL/TIS/clip、串行计算、保存重载和断点恢复检查保留。
- 无混合组则明确跳过GRPO。任意阶段失败均停止后续阶段并打包已有证据；不自动晋升基线、不自动进入Phase 6。
- 四阶段总时限仍约30分钟（调度预留打包时间），单作业费用估算上限10元；2.18元/小时下四阶段运行时长上限约1.09元。实例启动、预检、闲置、打包、存储等不含在内，不自动关闭实例。

## 运行命令

上传 `grpo-balanced-warmstart-20261029.zip` 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm &&
cd /root/rivermind-data &&
BALANCED_WARMSTART_ZIP="$(find /root -maxdepth 5 -type f -name 'grpo-balanced-warmstart-20261029.zip' -print -quit)" &&
test -n "$BALANCED_WARMSTART_ZIP" &&
python -m zipfile -e "$BALANCED_WARMSTART_ZIP" /root/rivermind-data &&
cd /root/rivermind-data/AgenticPostTrainingLab &&
python scripts/run_balanced_warmstart.py --hourly-price 2.18
```

依赖原模型缓存、基线和31步候选及原采样选题依据，不需要上传权重；运行前检查哈希。使用新的脚本、配置、结果目录和回传ZIP，原失败结果不需要删除。

完成或阶段失败后，回传项目根目录文件：

`grpo-balanced-warmstart-review-20261029.zip`

先下载结果，再停止闲置GPU。若仍触发热启动验收，保留结果，不绕过检查或删除目录自动重跑。

## 读代码的顺序

1. `configs/experiments/grpo_balanced_warmstart.json`：原24个rollout题目ID、完整36个示范ID、新增12题ID、两轮与固定学习率。
2. `scripts/balanced_warmstart_common.py`：
   - `select_demonstrations`：只从训练集选同4个family的12条left_grouped；检查chosen正确及开头括号。
   - `demonstration_schedule`：36条两轮、18个四例批次、各条恰好出现两次。
   - `context / identity`：原候选与数据身份、开发隔离、36条示范哈希及四套评测ID。
   - `load_rollouts`：仍只准入B重新采样的原24题、真实token和行为logp。
3. `scripts/train_balanced_warmstart.py`：
   - `warmstart`使用36条示范进行SFT，评测保留原24题并单列新增12题。
   - `generate / grpo`保留原768条新采样、Adam延续与有限更新。
   - `audit`按36×2重新计算训练顺序，验证四套评测与A/B/C收益归因。
4. `scripts/run_balanced_warmstart.py`：一条命令执行四阶段、限时、失败停止与回传打包。
5. `tests/unit/test_balanced_warmstart.py`：选题、格式、等例次调度、不可变超参数与失败停止测试。

原 `demonstration_warmstart` 三个脚本、原实验配置、src及历史来源快照均未改动。

## 本地验证

- 完整tests目录：435 passed，1条既有微型PEFT提示；本轮新增14项全部通过。
- 36条示范全部正确、编码无截断；新12条与预定ID一致，训练/开发题族与输入隔离。
- 调度确认18步、72例次，每条恰好两次；含EOS监督token总数2040。
- 真实135M原31步候选CPU重建参数身份一致；原Adam31步动量导入误差0，父RNG未恢复。
- 79个本轮来源文件校验通过；上轮79个执行来源仍完全一致，旧源码未修改。
- git diff --check通过。新云端vLLM、L20与135M本轮训练尚未实际运行；不提前声明效果。
- 保留原跨引擎严格概率门的历史限制；TIS与sampled k3仍按既有近似监控处理。

## 结果回来后先看什么

先看原48题与开发集上此前8条必要括号退化是否消失，再看原24题是否出现新增正确、8道natural难题是否改善。新增12题拟合变好只是覆盖验证，不能单独证明泛化提升。

若热启动通过，再分别看B−A和C−B；只有C相对B改善才能归于本轮GRPO。仍无显著收益时继续逐项排查，不自动无限训练。
