# Phase 5: Correct Demonstration Warm Start and Fresh GRPO

状态：新实验代码已实现；本地预检与测试见文末。135M 模型的本轮云端训练尚未执行，不能据此宣称效果提升。仍停留在 Phase 5，不自动进入下一阶段。

## 本轮要回答的问题

此前分级奖励可以使全错组出现非零 advantage，但可能奖励“结构较完整的错误”。本轮改为先用正确示范提高正确回答出现的概率，再检查重新采样后的 GRPO 是否带来额外收益。

| 模型 | 起点与操作 | 对照用途 |
| --- | --- | --- |
| A | 原31步GRPO候选，保持权重 | 固定起点 |
| B | A + 24条正确示范的小规模SFT | B−A 只归因于监督热启动 |
| C | B重新采样后有限GRPO | C−B 衡量本轮GRPO增量，C−A衡量组合方案 |

三者都使用同一GPU、原生贪心生成、64个输出token上限；评测包含实际24道训练题、原48道训练诊断题和56道开发题。合计每模型128条评测记录，跨套题有重叠，不能当作128道独立题合并统计。24题上的提升是训练拟合证据；开发集才用于检查泛化。历史未完成一致性审核的本地CPU补评不用于本轮准入或效果结论。

## 冻结的实验设置

- 从原31步候选开始，完整参数SHA256为 `70804ab1c9344f7578396e4ccaa90dfe1ea374c69f96cb69ee3970d036eedf7d`。不使用此前两个37步奖励对照分支。
- 使用已有偏好数据中相同24个训练输入及其 chosen 正确答案，不新增开发题或人工编造答案；所有 chosen 在预检中重新验证为正确。
- SFT：3轮，每轮24例，每批4例，共18步；学习率5e-5，FP32、串行累积、LoRA only。损失仅覆盖assistant回答与真实EOS，prompt和padding不计入。顺序由固定seed随机打乱。
- SFT为不同目标，使用独立的新AdamW。原GRPO优化器文件保留，不把SFT动量混入GRPO。
- B保存重载并完成评测后，vLLM从B实际重新采样24×32=768个回答；温度1.0，无top-p/top-k截断，保留原生token、EOS和真实行为logp。示范答案不插入采样组，旧31步候选的采样也不复用。
- GRPO延续用户先前选择的原Adam动量链：导入原候选累计31步的动量，参数从B开始；新输入绑定包含B、采样和评分身份，不恢复父轮RNG。SFT的18步与GRPO累计Adam计数分别报告；GRPO最终计数31至37。
- GRPO学习率1e-5；reference继续使用原冻结SFT，beta=.02、clip=.2、TIS cap=2、梯度裁剪1，保留既有sampled k3与clip fraction停止条件。没有同时更换reference来扩大实验变量。
- 主奖励是“验证正确=1，否则=0”。复用已实现的等价表达式验证：要求完整EOS、合法工具调用、运算结果与结构验证，不奖励仅数值巧合、可执行错误或部分结构。离线评分的0.1/0.3不进入本轮训练。
- 只取同时有0和1的混合组，按冻结题目顺序每批最多4个完整组，一遍最多6步。不对全错/全对组做只有KL的更新。无混合组时明确记为跳过GRPO，C等于B，不伪称强化学习成功。
- 首个GRPO步做一次丢弃后重放的断点恢复检查；实际optimizer调用次数与逻辑步数分开计数。
- 一次四阶段作业最多1800秒，并受单作业10元估算上限限制；2.18元/小时对应四阶段最多约1.09元。预检、打包、实例启动、闲置、存储和计费取整不在该估算内，不自动关机。

SFT之后若开发严格正确数或验证正确数下降、任一评测套题出现新的结构失败，保留A/B和失败证据并停止后续采样与GRPO。GRPO停止条件触发时保留实际已完成步数和C评测，不自动追加训练。C相对B的开发回退与结构回退单独报告，不自动晋升基线。

## 代码阅读顺序

1. `configs/experiments/grpo_demonstration_warmstart.json`：看预算、训练规模、旧候选身份和固定case IDs。
2. `scripts/demonstration_warmstart_common.py`：
   - `encode_demonstration`：正确性验证、prompt mask与EOS。
   - `context / identity`：验证原候选、基线、数据与来源快照。
   - `load_rollouts`：把B的真实采样与B权重绑定，检查token/行为logp/解码。
   - `choose_batches`：二元奖励、混合组和单遍有限调度。
3. `scripts/train_demonstration_warmstart.py`：
   - `sft_step / warmstart`：仅正确示范的响应损失，A/B评测与保存重载。
   - `generate`：独立进程里的vLLM重新采样。
   - `grpo`：FP32 old/new同路径检查、冻结reference、原Adam导入、有限更新及恢复重放。
   - `audit`：重算奖励与三组指标，独立列出B−A、C−B、C−A。
4. `scripts/run_demonstration_warmstart.py`：整次限时运行；任何阶段失败就停止后续阶段并打包实际证据。
5. `tests/unit/test_demonstration_warmstart.py`：真实微型LoRA的SFT→新采样→GRPO→Adam延续与恢复，以及失败停止测试。

旧训练器、src文件、历史source快照和顶层configs未修改。新配置放在experiments子目录，避免破坏原候选冻结的来源身份。

## 云端执行

上传 `grpo-demonstration-warmstart-20261028.zip`。沿用原实例、原模型缓存及历史产物，无需上传权重。压缩包内有项目目录前缀，在数据目录解压即可；会核验所需历史文件，缺失就停止。

```bash
conda activate /root/rivermind-data/envs/apt-vllm &&
cd /root/rivermind-data &&
WARMSTART_ZIP="$(find /root -maxdepth 5 -type f -name 'grpo-demonstration-warmstart-20261028.zip' -print -quit)" &&
test -n "$WARMSTART_ZIP" &&
python -m zipfile -e "$WARMSTART_ZIP" /root/rivermind-data &&
cd /root/rivermind-data/AgenticPostTrainingLab &&
python scripts/run_demonstration_warmstart.py --hourly-price 2.18
```

完成或阶段失败后，从项目根目录下载：

`grpo-demonstration-warmstart-review-20261028.zip`

若上传文件尚未找到或预检失败，不会开始训练；保留终端报错即可定位问题。运行目录固定且拒绝覆盖，不要删除失败结果后盲目重跑。下载完成后手动停止闲置GPU。

## 如何判断下一步

先检查B是否改善正确答案的生成及格式，再看B的768条新采样是否增加混合组、是否覆盖此前全错的题。之后必须看C−B的逐题变化与开发表现；仅B−A改善不能称为GRPO改善。一次固定seed和有限题目规模不足以证明显著提升，本轮不预设必然成功，也不自动无上限地训练。

## 本地验证

- 当前 tests 目录完整回归：421 passed，1条既有微型PEFT配置提示；新增12项全部通过。命令为 python -m pytest -q tests。
- 真实135M父模型CPU重建完整参数SHA256与31步候选一致；120个LoRA张量、230400个可训练参数。
- 原Adam动量导入误差0，累计step=31、父checkpoint局部step=15，父RNG未恢复。
- 全部24条chosen重新验证正确，每条监督回答28个token（含EOS），未截断。79个来源文件身份校验通过。
- 真实微型模型验证SFT后从新参数分布采样，再做GRPO更新；导入父Adam及恢复重放通过。
- 云端vLLM、L20数值路径和135M新训练尚待实际运行。原跨引擎严格数值门的历史限制仍保留；TIS与sampled k3继续是近似监控。
- git diff --check通过。未自动打开报告。
