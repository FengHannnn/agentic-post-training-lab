# DPO 退化机制排查：扩展实验记录

记录日期：2026-11-05，Asia/Shanghai。

## 用户最新定位

用户明确：DPO阶段的主目标是得到比加权SFT更好或足够接近的DPO模型，即可完成阶段。已经完成的退化机制排查作为扩展成果保存；后续用户要求编写交接文档时，再将本记录整理并纳入。

当前主线是dpo-early86-vs-weighted-sft-v1-20261105：固定已有DPO86检查点与加权SFT作直接对照，两模型、264条主轨迹、训练更新0。该作业已准备运行包，等待云端执行和返回结果。当前只把已做过的排查归档为扩展，不追加机制实验，也不将额外机制验证设为主线收尾前提。

## 已完成的扩展实验

| 编号 | 问题与控制设计 | 已审主要结果 | 可用于报告的解释 |
|---|---|---|---|
| E1 更新阶段定位 | 对同一原加权DPO训练的43/86/129/172/215/258步作固定输入评分与生成对照，含父模型和加权SFT；训练更新0 | 86步随机完整任务13/16、greedy恢复21/28；258步降至0/16、6/28。stale chosen CE从父模型0.004793升至258步1.065367 | 训练后期偏移积累，偏好loss改善与完整任务表现下降同时出现；早期检查点保留较好表现 |
| E2 困难权重对照 | 相同chosen-only父模型、169对、fresh Adam、6遍258更新，仅把困难原始权重4改为全1 | 均匀DPO随机完整任务2/16、完整恢复33/84；同轮原加权控制1/16、15/84。E3同硬件重训纯加权后恢复10/84，冻结均匀模型33/84 | 困难权重放大恢复退化，均匀权重仍保留正确动作保真损伤 |
| E3 chosen保真 | 同机两支fresh训练，纯加权DPO与加权DPO+chosen CE，系数0对1，精确reference字节相同，每支258更新/1014曝光 | 随机完整任务0/16→10/16，完整恢复10/84→37/84；恢复配对改善28、退化1。stale chosen CE 1.065365→0.055512 | 额外维持正确动作概率明显缓解退化；相对偏好训练的正确动作保真值得单独观察 |
| E4 完整续跑覆盖 | 同机fresh chosen CE控制与续跑CE；DPO部分、CE系数1和1014次CE动作曝光保持，替换为train专家完整续跑动作 | 随机恢复36/84→60/84，greedy恢复14/28→25/28；随机完整任务8/16→7/16。同轮加权SFT完整任务16/16、恢复57/84 | 完整续跑监督改善后续校验和提交；完整任务的前段SQL/格式生成仍是主要短板 |

数字保留各自作业与同轮对照，交接时按运行ID逐项说明。固定development为4个任务及重复rollout，表内各实验的评测分母和采样口径保留；本轮原因排查未使用test。

## 关键实现与公平性记录

原偏好数据169对，SHA256 c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。父模型chosen-only-sft-v1-20261102，权重078182…、SFT255/DPO11。

hard4：84个困难样本、85个其他样本，原始权重4/1，全数据平均权重归一为1；困难样本占归一化权重约79.81%。batch4为每次更新4个偏好对，逐对累积梯度、一次裁剪与optimizer.step，尾批1对；每遍43次更新，6遍258次、1014次偏好曝光。

DPO对chosen/rejected动作的response token logp求和，与冻结父模型reference比较相对margin，经beta0.1的负log-sigmoid得到偏好loss；场景权重乘在每对loss外，再按实际batch大小平均。CE为单个正确动作response token均值（含EOS），动作间均匀平均。E3/E4中的CE不再乘hard4权重。

E3/E4精确父模型reference缓存SHA256 448b6c8af6fb18ae35fcad02f904167fea1f75d293fff0c53a70968c280b2a65；reference_run虽然名为uniform-control，内容仍为原父模型分数。E3/E4两支训练前初始评分之间最大差0，评分阶段与该缓存最大差0。

E4 CE池来自train的12任务×7错误上下文，共84条成功专家完整续跑、288个动作：calculate_change72、check_result84、submit84、query_sql48。固定seed2026110504，打乱全池循环后截断为1014次，每个动作3–4次曝光；实际监督查询167、计算256、校验296、提交295次。

E4两支均258次fresh Adam更新、累计SFT255/DPO269，辅助CE联合更新单列。控制CE响应105486token、额外forward0；续跑CE响应85203token、额外forward输入1202952token、额外forward1014；每支DPO输入1941912token。动作预算与token计算量分别保留。

## E4的主要证据与案例

有效首动作后失败：chosen控制33次，续跑CE5次；各自有效首动作后的成功为30/63与50/55。随机恢复submit调用37→60，工具错误83→45，总生成动作482→393。上述条件转换率按各模型自己的有效首动作子集统计；固定84轨迹的主配对改善28、退化4。

固定development专家恢复探针：check_result CE 0.039817→0.004419，submit CE 0.007715→0.000867；两者约下降89%。calculate_change CE 0.068445→0.060064，query_sql CE 0.003664→0.004326。原169对stale chosen CE 0.055512→0.292447，记录替换CE数据后的保真取舍。

续跑CE完整任务9个随机失败中，8个始终未验证任何月度指标，另1个到末轮才验证change_cents，预算在校验/提交之前耗尽。典型错误包括非严格JSON、错误SQL列/函数、未验证月度证据引用与无进展重复查询。

## 执行问题与修复记录

1. E2原审计在跨作业reference末位差异检查处中止。最大单token差约0.000131845，beta×reference margin差对应的DPO loss变化上界约0.0000209966。随后单独修复数值审计，208个原文件保持一致，没有重新训练；补审计及本地重放通过。
2. 原E4在prepare因GPU UUID变化停止，更新0。E3 UUID为086e2842…，当前为92e14e85…，用户完整runtime比较其余字段一致。修复后在同一新GPU上fresh训练chosen控制与续跑CE，各258次，并保留作业内严格UUID/软件/训练精度检查、失败差异记录和新run_id。旧失败记录保留。
3. E4新chosen控制重训得到与历史E3 CE完全相同的adapter权重ad6a8f…；FP32原始169对/238探针分数逐项相同。在同轮多LoRA BF16原生推理中，greedy恢复5/28条生成轨迹分叉，其中2条终局不同；随机恢复28/84条生成轨迹不同、4条终局不同。已保存相同prompt/seed/token前缀后的分叉。

第3项的具体引擎原因仍为扩展待查项。可能来源包括LoRA加载/编号与进程状态、target_modules序列化顺序、固定条件下的引擎重复性。独立dpo-inference-confirmation-v1-20261105运行包已准备，尚未收到云端执行结果；用户收缩主线后，该包不作为当前必做作业。DPO86直接对照内保留单LoRA独立进程和小规模必要检查即可。

## 审计与原始证据位置

E1：dpo-regression-checkpoints-v1-20261103，已复核384条主轨迹、2692个生成动作。

E2：dpo-uniform-control-v1-20261104及audit-repaired补审计，已复核528条主轨迹、3202个生成动作。

E3：dpo-chosen-ce-control-v1-20261104，已复核528条主轨迹、3333个生成动作；返回包文件名带review-Copy1，内部run_id保持原名。

E4：dpo-continuation-ce-paired-v1-20261105，已复核528条主轨迹、3056个生成动作，另重建84条train专家续跑和固定CE采样；校验130个冻结源码文件、27份provenance、5份真实adapter。返回包SHA256 14e24e215d6ad108758f937a6c24d9bc92a77c213e3931da44f8c198fcc9cabb。

原始review规范封存在项目v2/artifacts/reviews/<run>/extracted/<run>；E2修复包外层为<run>-audit-repaired。各报告、原始分数、配置、模型SHA、训练历史及严格重放结果均已保留。state.pt与中间权重按原约定保留云端，云端auditor校验实际Adam状态；返回包提供历史、绑定与完成元数据供本地复核。

E4相同权重推理分叉明细位于canonical review外层identical_adapter_generation_comparison.json；流程/提交分析位于trace_analysis.json。

## 后续写入交接文档时的组织方式

用户后续提出交接要求时，将本记录作为“扩展：DPO退化机制排查”整理，包含：研究问题、每项控制变量、同轮结果、训练/评测公平性、主要证据与典型案例、运行修复与复现入口、已完成和扩展待查项。主线章节只保留早期DPO86直接对照及最终阶段结论。

本次仅新增扩展记录，不改动当前已发布的DPO86运行包、冻结配置或交接正文。此前报告中以推理确认作为“下一项”的文字按其历史时间保留；当前接续以用户最新目标及DPO86直接对照说明为准。

## 报告索引

- reports/dpo-regression-checkpoints-v1-review-20261104.md
- reports/dpo-uniform-control-v1-review-20261104.md
- reports/dpo-chosen-ce-control-v1-review-20261105.md
- reports/dpo-continuation-ce-paired-v1-review-20261105.md
- reports/dpo-regression-investigation-summary-20261105.md
- reports/dpo-early86-vs-weighted-sft-v1-preparation-20261105.md（当前主线）
