# L20 环境迁移复核

日期：2026-11-01（Asia/Shanghai）。结果：四阶段全部通过，可以在 L20 继续执行 SFT 诊断。

| 项目 | 实测结果 |
|---|---|
| GPU | NVIDIA L20，44.39 GiB 显存，计算能力 8.9，BF16 可用 |
| Python / torch | 3.11.17 / 2.8.0+cu128 |
| vLLM / Transformers / PEFT | 0.11.0 / 4.57.6 / 0.17.1，与原环境一致 |
| 模型缓存 | 固定 Qwen2.5-1.5B revision，7 个文件校验通过 |
| 脚本专家 | 20/20 任务通过 |
| 训练探针 | 512 token，FP32 策略与冻结 reference，一次真实 LoRA 更新通过 |
| LoRA | 1,089,536 参数、112 张量，首步 56 张量变化，冻结底座保持一致 |
| 概率与重载 | 同路径及 adapter 重载最大 logprob 差均为 0 |
| 训练探针显存 | allocated 14.4072 GiB，reserved 14.6582 GiB |
| vLLM | BF16、8192 上限，128 和 8176 token 输入均生成 OK 并正常 EOS |
| 作业时间 | 73.18 秒 |

验证范围为上述训练与推理探针；完整 SFT 诊断将在实际训练上下文上检查 adapter、三后端输出和 12 个训练任务执行表现。SFT pilot 文件与最终 adapter 的迁移校验由诊断入口进一步完成。

本地复核：ZIP CRC 通过，18 个成员文件摘要匹配，四阶段独立结果与汇总一致，检查脚本与本地版本一致。ZIP SHA-256：50be9c69419b91876a6a180ad5db7f696f4f1332c365a84bf0e6280e801d5c25。

原始回传包和本地核验记录保存在 artifacts/reviews/environment-l20-20261101-e44637e9/。后续云端硬件按 L20 记录。

## 下一步

上传并解压 v2-sft-diagnostics-v1-20261101.zip 后执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data/AgenticPostTrainingLab/v2
python scripts/run_v2_sft_diagnostics.py
```

回传 /root/rivermind-data/AgenticPostTrainingLab/v2/sft-diagnostics-v1-20261101-review.zip。
