# V2 4090D Environment Check

日期：2026-10-30

目标：确认克隆到4090D的实例能运行V2的1.5B模型基线，并测量短序列LoRA训练路径。脚本适配当前V2路径，保持原实验、正式模型缓存和基线配置原样。

## 检查内容

1. Python与依赖版本：torch、CUDA、vLLM、Transformers、PEFT、Accelerate、Safetensors、Hugging Face Hub。
2. GPU实际名称、计算能力、驱动、显存总量/空闲量、BF16矩阵运算与反向传播。
3. Qwen2.5-1.5B-Instruct缓存完整性、固定revision及V2程序参考策略。
4. FP32策略＋FP32冻结reference，512-token序列的一步LoRA更新、参数冻结、同路径logp和保存重载。
5. 按原V2配置启动BF16 vLLM，测试短请求和8176-token输入，保存原生token与raw logp。

各阶段独立进程，HF训练结束后再启动vLLM。训练显存不足时继续检查推理路径，便于分别判断两种用途。不同依赖版本会记录差异，最终按实际探针结果判断运行兼容性。

## 执行

上传 `v2-4090d-environment-check-20261030.zip`，在新实例执行：

```bash
conda activate /root/rivermind-data/envs/apt-vllm &&
cd /root/rivermind-data &&
ENV_CHECK_ZIP="$(find /root -maxdepth 5 -type f -name 'v2-4090d-environment-check-20261030.zip' -print -quit)" &&
test -n "$ENV_CHECK_ZIP" &&
python -m zipfile -e "$ENV_CHECK_ZIP" /root/rivermind-data &&
cd /root/rivermind-data/AgenticPostTrainingLab/v2 &&
python diagnostics/check_environment.py --download-model
```

若克隆实例已有完整Qwen缓存，会直接复用；缺少缓存时下载约3GB。入口会自行设置src路径，沿用apt-vllm环境，不安装或升级依赖。

默认整次检查最多900秒。若要同时估算费用，请使用新4090D实例的实际时价，例如：

```bash
python diagnostics/check_environment.py --download-model --hourly-price 你的实际每小时价格
```

设置时价后，另按10元上限缩短检查时间。费用记录覆盖本次诊断运行，实例闲置与存储另计。

## 回传

脚本结束会打印完整下载路径：

`v2-4090d-environment-review-20261030-<run-id>.zip`

下载并回传该ZIP。各阶段的结果、完整日志、显存峰值和数值检查会统一保存。即使某项失败或超时，也保留已有诊断证据。

结果字段：
- `cuda_ready`：CUDA、依赖和BF16基础检查。
- `model_cache_ready`：正确revision及缓存文件校验。
- `v2_baseline_ready`：V2的BF16 vLLM配置通过。
- `short_context_lora_ready`：双FP32模型、512-token LoRA探针通过。
- `all_checks_passed`：以上完整检查通过。

探针adapter与优化器只写到新的诊断结果目录，完成一次测试更新，不进入正式训练链。

## 代码阅读

- `diagnostics/check_environment.py::inventory`：依赖、GPU、BF16基础计算。
- `model_ready`：缓存校验与可选下载。
- `train_probe`：真实1.5B LoRA反向、冻结reference、logp与重载。
- `vllm_probe`：按V2实际配置测试短/长上下文。
- `verdict / run`：区分推理与训练结论、分进程、限时与结果打包。
- `diagnostics/tests/test_environment_check.py`：控制流程、错误留档和超时测试。

## 本地验证

- 12项本地测试通过：结论分类、缓存缺失、训练OOM后的独立推理检查、超时停止和日志打包。
- 在本地CPU环境实跑了依赖/CUDA未就绪路径，按预期跳过模型下载并生成完整诊断ZIP。
- 原V2启动包的文件身份保持；4090D真实计算结果由新实例运行本脚本生成。

## 适用范围

本次验证覆盖8192上下文的推理配置与512-token串行FP32 LoRA训练探针。正式多轮训练的batch、轨迹长度和精度，将依据本次显存峰值进一步配置；模型效果继续由经营数据任务基线评测。
