# E4 专家续跑 CE 覆盖对照执行说明

2026-11-05，Asia/Shanghai。运行 ID：dpo-continuation-ce-control-v1-20261105。

E3 已通过云端审计与本地复核：加入 chosen CE 后，随机完整任务 10/16、随机完整恢复 37/84；同轮加权 SFT 为 16/16、59/84。CE 分支 63 次有效首动作中有 32 次随后失败。按既定顺序执行 E4，检验完整专家续跑监督的增益。C1 表现验收保持未通过，GRPO 暂缓。

## 冻结的单变量设计

DPO 部分保持 E3 配置：chosen-only SFT255/DPO11 父模型、原 169 对、hard4 归一化权重、精确父模型 reference、beta 0.1、LR 5e-6、fresh AdamW（weight_decay=0）、batch 4、seed 2026110301、6 遍、258 次更新、1,014 次偏好曝光、grad clip 1。固定取最终检查点。

联合目标仍是每批平均 w_i×DPO_i + 1×CE_i。DPO 对 chosen/rejected 动作的 response token logp 求和，CE 在每个正确动作的 response token 上求平均（含 EOS），再对批内动作求平均。CE 的动作权重均为 1。每个 batch 只做一次 optimizer.step，DPO 与 CE 的梯度共同裁剪与更新。

唯一数据替换：E3 的 CE 来自原 169 个单步 chosen；E4 的 CE 来自 train 的固定错误上下文后的专家完整续跑动作。E4 仍从父模型重新初始化；冻结的 E3 CE 模型是主对照。

## 数据池、曝光与计算量

12 个 train 任务×7 个固定错误场景=84 条成功专家续跑，保留真实错误前缀和环境观察作为上下文，监督注入前缀之后的专家动作。真实环境逐步执行至成功提交，池中 288 个动作：calculate_change 72、check_result 84、submit 84、query_sql 48。

固定 CE seed 2026110504。对整个池做无放回打乱循环，最后一轮截断至总计 1,014 次；288 个动作各出现 3–4 次。固定抽样工具曝光：calculate_change 256、check_result 296、submit 295、query_sql 167。每次偏好曝光配一个排定的 CE 动作，每批仍平均 4 个，最后一批 1 个。

只使用 train 构造训练池；development 仅用于固定评测和探针，test 保留。预检会重建专家池、完整续跑证据和采样序列，核对消息身份、文件哈希、EOS 和最大长度 2048。

E3 复用 DPO chosen forward 计算 CE；E4 的 CE 上下文独立，新增 1,014 次 CE forward。CE 动作曝光预算相同，token 计算量逐项记录在 ce_summary.json、training_summary.json 和每步历史：DPO 输入/响应 token、CE 响应 token、CE 额外 forward 输入 token。E3 CE 响应 token 为 105,486，额外 CE forward 为 0。本地真实 tokenizer 预检：E4 CE 响应 token 为 85,203，额外 forward 输入 token 为 1,202,952；DPO 输入 token 为 1,941,912。最长 CE 输入 1,527 tokens。云端预检与最终审计会重算这些预算。

## 参考模型与硬件

父模型 adapter SHA256：078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55。

reference 缓存 SHA256：448b6c8af6fb18ae35fcad02f904167fea1f75d293fff0c53a70968c280b2a65。该文件保存父模型的逐 token 分数，来源为 E2 已修复审计的精确缓存；E3、E4 都使用其原始字节。

E3 CE 对照 adapter SHA256：ad6a8f60ef71c9b56e62f3d6373bf1e258d56b1bf87487a999f83e25351b743b；纯加权对照 479785a1bfa1c97682e64f8b6dda5c6391c25219afc13a15b6a83ffcf56bdd65。

执行前核对与 E3 相同的 NVIDIA L40S、GPU UUID 086e2842-344c-9528-cbae-b8fcba665e1c、显存/计算能力、Python/torch/CUDA/cuDNN/transformers/peft/vLLM 版本和训练精度设置。训练前重新打分父模型，与精确缓存比较；全部检查在首个优化器更新之前完成。若机器身份发生变化，预检停止并打包日志，随后据实际环境准备匹配控制。

## 固定评测与输出

4 个模型：加权 SFT、E3 纯加权 DPO、E3 DPO+chosen CE、新 continuation CE。每模型完整任务 greedy 4、随机 16；恢复 greedy 28、随机 84，共 80 条完整任务+448 条完整恢复=528 条轨迹，保留原生 prompt/response tokens 和采样种子。FP32 额外评分父模型，共 5×169 对与 5×238 个固定专家动作探针。

主比较为 E3 chosen CE→E4 continuation CE 的配对完整任务/恢复成功。另报告有效首动作后成功转换、工具选择/提交、分场景表现和正确动作 CE。该比较中的两支都为 SFT255/DPO269，并另记辅助 CE 联合更新 258 次。

本作业仅新增训练 258 次更新。已审 E3 对照只读加载。中间 checkpoint 的 adapter 与 state.pt 留在云端；返回包包含新模型最终 adapter、完整训练历史、真实 CE 池/续跑证据/采样序列/计算量、原始分数、所有轨迹、冻结源码和审计。总时限 14,400 秒，逐阶段实时日志，失败或超时也生成返回包；保护已有同名输出。

## 本地验证

E4 新增 8 项检查通过（首组 7 项与追加配置检查 1 项）；源码编译、5 个 CLI 入口与真实输入预检通过。验证回执：handoff/dpo-continuation-ce-local-validation-20261105.json。覆盖真实 LoRA 下相同 CE 输入的等价性、独立续跑上下文的梯度更新、固定曝光与实际 Adam 计数、train 隔离、原生完整重放、分数重算、篡改检测、硬件与配置固定、失败归档。实际 GPU 训练与生成等待执行本作业。

## 云端执行

将 dpo-continuation-ce-control-v1-20261105-run.zip 上传到云端 v2 目录后执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-continuation-ce-control-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_continuation_ce.py
~~~

命令使用虚拟环境解释器的绝对路径，终端显示 base 也可运行。完成后回传 dpo-continuation-ce-control-v1-20261105-review.zip。保留云端原有 E2 已修复审计目录及 E3 完整 artifacts，用于冻结对照与输入来源验证。
