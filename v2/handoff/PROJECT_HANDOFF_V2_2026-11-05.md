> 历史交接归档：当前接续请读 PROJECT_HANDOFF_V2_GRPO_2026-11-05.md。截至2026-11-05，DPO短程第12步已达收尾标准，用户确认下一对话进入GRPO；下文保留当时状态。

# Project V2 handoff — 2026-11-05

当前阶段：按用户最新目标，固定早期DPO86与加权SFT做直接对照。DPO更好或足够接近即可结束本阶段；原因排查材料保留，当前不执行末期CE模型的额外推理确认。GRPO等待本轮性能结论。

用户偏好：中文报告，直接写结果和证据；对话简要列主要结果并提供报告链接；报告不自动打开。云端作业由用户上传 ZIP 并使用给定绝对路径 Python 执行；无额外价格参数。

## 当前结果

E1：DPO86 随机完整任务 13/16、greedy 恢复 21/28；DPO258 降到 0/16、6/28，正确计算/校验/提交 CE 随训练后期恶化。

E2：均匀权重 DPO 随机完整任务 2/16、恢复 33/84；同轮原加权模型 1/16、15/84。原审计在旧 reference 的跨硬件末位差异处停止，已用独立数值策略补审计；208 个原文件保持一致，528 条轨迹/3,202 个生成动作复核通过。修复包 SHA256 1392f4bce45f14b74d407ace78f4bdf854d59906116647aa13f9c8c44d9794b4。

E3：新纯加权控制与 chosen CE 分支各 fresh Adam 训练 258 次，从相同父模型出发，精确 reference 字节相同，系数 0 对 1。两支更新前 fresh reference 和评分阶段 reference 与缓存逐 token 差均为 0。

| 模型 | greedy 完整任务 | 随机完整任务 | greedy 完整恢复 | 随机完整恢复 |
|---|---:|---:|---:|---:|
| 加权 SFT | 4/4 | 16/16 | 20/28 | 59/84 |
| 均匀 DPO | 3/4 | 2/16 | 15/28 | 33/84 |
| E3 纯加权 DPO | 0/4 | 0/16 | 6/28 | 10/84 |
| E3 DPO+chosen CE | 3/4 | 10/16 | 17/28 | 37/84 |

CE 使随机恢复配对 28 次改善、1 次退化；stale chosen CE 1.065365→0.055512。随机有效首动作后成功 10/51→31/63，CE 分支仍有 32 次有效首动作后的失败。C1 表现验收未通过。报告：reports/dpo-chosen-ce-control-v1-review-20261105.md。

E3 返回文件名 dpo-chosen-ce-control-v1-20261104-review-Copy1.zip，SHA256 cefcce0d29880985e01a975d57c6e501f2b066118590363db677bd9f5de286f0。冻结源码114、provenance24、真实adapter5、5×169偏好评分与5×238探针均已核对；本地严格重放528条轨迹、3333个生成动作，SQLite3.53.1；test未使用。训练state.pt云端审计验证，保留云端；本地核对完整返回历史/绑定/完成元数据。

## 模型与 reference

- 父模型 chosen-only-sft-v1-20261102/final：078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55，SFT255/DPO11。
- 加权 SFT hard-recovery-sft-weighted-v1-20261103/final：f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f，SFT513/DPO11。
- 均匀 DPO dpo-uniform-control-v1-20261104/final：6f6de631322216e24a824f03d3ae4a90929accd2ecbc08624908275384dfe11a，SFT255/DPO269。
- E3 纯加权 training/weighted_control/final：479785a1bfa1c97682e64f8b6dda5c6391c25219afc13a15b6a83ffcf56bdd65，SFT255/DPO269。
- E3 chosen CE training/dpo_ce/final：ad6a8f60ef71c9b56e62f3d6373bf1e258d56b1bf87487a999f83e25351b743b，SFT255/DPO269，辅助联合CE更新258。
- 原169对：c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。
- E2/E3/E4 精确父模型reference：448b6c8af6fb18ae35fcad02f904167fea1f75d293fff0c53a70968c280b2a65，reference_run=dpo-uniform-control-v1-20261104；其内容仍是父模型分数。

## E4历史结果与评测记录

E4 paired回传SHA25614e24e215d6ad108758f937a6c24d9bc92a77c213e3931da44f8c198fcc9cabb。云端7阶段成功，516次更新（两支各258）。本地核验130份冻结源码、27份provenance、5份adapter、5×169偏好与5×238探针原始分数；重建84条train专家续跑/288CE动作/1014抽样。重放528条development轨迹、3056个生成动作；test保留。runtime为L40S UUID92e14e85…；训练前reference与缓存差0、两支初始差0。

| 模型 | greedy完整任务 | 随机完整任务 | greedy恢复 | 随机恢复 |
|---|---:|---:|---:|---:|
| 加权SFT | 4/4 | 16/16 | 20/28 | 57/84 |
| 历史E3 CE | 4/4 | 8/16 | 16/28 | 34/84 |
| 新chosen控制 | 4/4 | 8/16 | 14/28 | 36/84 |
| 完整续跑CE | 4/4 | 7/16 | 25/28 | 60/84 |

续跑CE随机恢复配对改善28/退化4，有效首动作后失败33→5，check/submit CE约下降89%。完整任务9个失败中8个未验证任一月度指标。主指标仍落后SFT，C1表现验收未通过，GRPO暂缓，E1–E4训练原因排查完成。

新chosen控制权重SHA与历史E3完全相同ad6a8f…，FP32原始pair/probe分数一致。原生多LoRA BF16记录有greedy恢复5/28轨迹分叉，2条终局不同；已保存完全相同prompt/seed/token前缀后的分叉证据。adapter配置仅target_modules序列顺序不同。该问题作为独立推理重复性待确认项，训练与记录复核结果保留。

此前准备的dpo-inference-confirmation-v1四分支末期模型确认不在本轮执行。用户要求回到阶段目标：DPO比加权SFT好或差不多就收尾；当前只对已有86步候选作直接性能比较。

## 当前 DPO86 直接对照

运行ID：dpo-early86-vs-weighted-sft-v1-20261105，training_enabled=false，optimizer_updates=0。固定两模型：weighted_sft与dpo_0086，不临时选择其他checkpoint。

DPO：hard-recovery-dpo-matched-v1-20261103/checkpoint-0086，实际2遍/86更新/338偏好曝光，累计SFT255/DPO97。adapter SHA256 43100acde91f178de59ac7a65f29867862a02a7f1a6cbcd134b7d81a7318f46c；complete SHA256 d2e5716ee3955e457eaeb7727781e65f05a4d56fa9442d762ee2f06f60573c1c。SFT：hard-recovery-sft-weighted-v1-20261103/final，新增SFT258更新、累计SFT513/DPO11，权重f8fb95…。

选择依据E1同轮随机完整任务13/16对SFT16/16，greedy恢复均21/28。本次固定同4个development任务、原seed/温度，补齐随机恢复：每模型greedy完整任务4、随机完整任务16、greedy恢复28、随机恢复84，共132条；总264条。主看完整任务，恢复辅助，报告DPO减SFT成功数/百分点与配对改善退化，返回后直接讨论收尾。

每模型独立进程、单LoRA编号1，规范化target_modules配置顺序；每模型7个固定错误上下文首动作各重复2次，共28次附加生成，作为同一评测内的小质量检查。14400秒上限、实时阶段日志、同名保护、失败回传，原模型前后hash验证。test本轮保留。

prepare固定E1 audit/config/status与原matched config/input_binding hash，严格核对86步完成元数据和云端真实adapter SHA。86步权重本地未回传，已在E1云端审计绑定；实际文件保留云端，缺失/不符在推理前停止并回传。本轮review会带回86步真实adapter，便于封存。

## 当前执行

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-early86-vs-weighted-sft-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_early86.py
~~~

回传dpo-early86-vs-weighted-sft-v1-20261105-review.zip。下一轮核验两个模型身份、264条主轨迹和小质量检查，直接汇报完整任务与恢复对照。达到更好或用户认可的接近程度即可收尾；如有明显差距再讨论已提过的单次保守短程DPO，当前不预先发布新训练分支。

执行说明reports/dpo-early86-vs-weighted-sft-v1-preparation-20261105.md；验证handoff/dpo-early86-local-validation-20261105.json。历史E1–E4总结reports/dpo-regression-investigation-summary-20261105.md保留作解释材料。

项目C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2；工作脚本C:/Users/app/Documents/Codex/2026-11-03/dpo/work；用户交付outputs。报告不自动打开，对话简要写结果。
