# DPO 独立任务评测说明

日期：2026-11-02。运行 ID：dpo-independent-test-v1-20261102。

固定已复核的 11 步 DPO 检查点，与共同的 126 步 SFT 基线开展配对评测。全过程只进行推理与审计，参数和优化器更新数为 0。

## 任务与设置

使用 make_tasks() 中 test-000 至 test-003，共 4 个任务、4 个 family，其中 2 个有首次查询临时故障。任务 ID、family、segment 及数据库 seed 与训练集分离；业务规则、表结构及工具保持同一套定义。此固定测试集在 SFT 阶段已做过贪心评测，本轮对两个冻结模型做配对比较，并加入新 seed 的随机采样。

| 方式 | 每模型轨迹数 | 两模型合计 |
|---|---:|---:|
| 贪心，每任务 1 次 | 4 | 8 |
| 随机，每任务 8 次 | 32 | 64 |
| 合计 | 36 | 72 |

随机 temperature=0.8，top_p=1，top_k=-1，基础 seed=2026110221。逐任务、逐轨迹、逐轮 seed 设置在两分支一致。32 次随机采样覆盖 4 个独立任务，每个任务重复 8 次；报告会同时展示逐任务结果与汇总结果。

vLLM BF16 推理，上下文 8192，单动作最多生成 256 token，每任务最多 12 个动作。全流程时间预算 5400 秒，保留打包时间。

## 冻结输入

SFT：artifacts/sql-sft-weighted-v1-20261101/final。适配器 SHA256：e41bef8a51d8d968bcfee4a606da364888a191c4e1ee85399215bc12b85813f0。

DPO：artifacts/dpo-train-v1-20261102/final。适配器 SHA256：fca8a4e07224169569b6cb4f81ab0daa0abe03f6d92caed0efcdf584313e6d6d。

准备阶段核对：固定模型 revision、两检查点文件、SFT 训练示范身份、DPO 的 44 对偏好身份及 train 来源、独立 test 清单。运行前后核对输入指纹。

## 输出与审计

scripts/run_v2_dpo_test.py 依次运行 prepare、greedy、sample、audit。scripts/test_v2_dpo.py 管理冻结输入和审计；既有 evaluate_v2_dpo.py 执行配对推理，随机采样函数新增可选 split 参数，原训练任务评测默认保持 train。src/apt_v2/dpo_test.py 提供原生 token 重放与配对统计。

输出包括两个模型的完整轨迹、原生 token 与概率、分支统计、逐任务成功数、成功/失败转移，以及 checkpoints_unchanged 和 0 更新审计。重放逐轮核对提示词、解码、动作、观察、奖励、终止、错误与证据进展，并核对轨迹汇总及最终 artifact。

回传包保留推理结果、配置、源码快照及检查点绑定；排除已有模型权重和优化器文件。出现运行失败时仍会保存日志并打包。

## 本地验证

87 项测试通过，Python 编译检查通过。新增测试覆盖原生 token/环境记录篡改检测、test 完整集合及配对 seed、72 条模拟轨迹的完整审计、冻结输入准备和检查点篡改、入口失败后的日志打包。真实 L20 上的推理结果待执行后回传复核。

## 云端命令

上传 v2-dpo-independent-test-v1-20261102.zip 至 /root/rivermind-data/ 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
unzip -o v2-dpo-independent-test-v1-20261102.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_dpo_test.py
```

回传文件：/root/rivermind-data/AgenticPostTrainingLab/v2/dpo-independent-test-v1-20261102-review.zip。若遇到错误，保留现有目录和 ZIP，直接回传检查。