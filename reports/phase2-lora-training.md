# Phase 2：LoRA 最小训练闭环

## 阶段目标

把已验证的数据送进真实模型，理解前向计算、loss、反向传播、优化器更新，保存 LoRA adapter 并重载验证。当前仍是小规模实验：只有 4 条英语训练样本和 2 条初始评测样本。最初验证运行 3 步，当前配置已改为 20 步；不能仅凭这两条评测题声称广泛能力提升。

使用 SmolLM2-135M-Instruct、CPU、float32、PEFT 和 PyTorch。暂时手写短训练循环，方便对照 Phase 1；后续较完整实验再引入 TRL。英语数据对应模型语言特点，原中文示例继续用于预处理教学。

## 运行方式

在项目根目录执行，始终使用本项目独立虚拟环境：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install --no-cache-dir -e ".[train]"
& ".\.venv\Scripts\python.exe" -m pytest tests -v
& ".\.venv\Scripts\python.exe" scripts\train_sft_lora.py
```

首次运行训练脚本会下载模型权重，与上一步仅下载 tokenizer 不同。模型缓存仍在项目 `.cache/huggingface`。安装命令禁用 pip 下载缓存，以避免受限环境下写入默认缓存时卡住；这不影响模型缓存。CPU 耗时与内存取决于本机配置，当前使用短序列、小 batch 和 20 步；首次验证也可临时改为 3 步。没有量化，没有 bitsandbytes，也不需要 CUDA。

权重已经缓存后可添加 `--local-files-only` 离线运行。可以通过 `--config` 指定另一个 JSON 配置。配置中的数据路径相对项目根目录解析。

## 阅读顺序

1. `configs/sft_lora_cpu.json`：认识步数、学习率、batch、LoRA 参数。
2. `src/agentic_post_training/trainers/sft_lora.py`：依次阅读 `check_split`、`attach_lora`、`model_inputs`、`evaluate_loss`、`train_steps`。
3. `scripts/train_sft_lora.py`：串起加载、训练前检查、训练、保存、重载。
4. `tests/integration/test_lora_training.py`：理解如何证明只更新 adapter，并验证保存恢复。

## LoRA 基础

线性层原来计算 `Wx`；LoRA 计算 `Wx + (alpha/r)BAx`。基础权重 W 冻结，只有低秩矩阵 A、B 更新。r 控制低秩容量，alpha 控制缩放。此实验将 LoRA 放在 attention 的 q_proj 和 v_proj，rank=4、alpha=8、dropout=0。

少量可训练参数能减少梯度和优化器状态的内存，但仍需运行完整基础模型的前向与相关反向计算，并不是只计算小 adapter。LoRA 的默认初始化通常让 B 为零，所以初始 adapter 不改变基础模型输出，第一步某些 A 的梯度可能为零。

## 每次训练更新

1. 用 `collate_sft` 组成 batch。
2. `optimizer.zero_grad` 清除上一轮梯度。
3. 模型读取 input_ids、attention_mask、labels，内部执行 causal shift 并计算回答 token 的交叉熵。
4. `loss.backward` 计算 adapter 梯度。
5. `clip_grad_norm_` 限制过大梯度；日志记录的是裁剪前范数。
6. `optimizer.step` 更新 LoRA 参数。

不能将 `loss_mask` 作为模型参数传入，因此 `model_inputs` 只保留模型接受的三个字段。也不能提前 shift labels，否则会错位两次。

此循环按固定顺序循环采样，使用恒定学习率，没有 scheduler、梯度累积、断点续训和混合精度。重复运行会创建新的产物目录，不覆盖已有实验。

## 评测与日志

训练前后在评测集上计算回答交叉熵，按有效预测 token 数加权，而不是把不同长度样本的平均 loss 直接平均。评测会关闭梯度和 dropout。还会保存同一道评测题训练前后的贪心生成结果；生成只输入问题，不把标准答案送给模型。默认最多生成 32 个新 token，长 JSON 可能被截断，这不代表训练标签被截断。

脚本检查训练集和评测集的 ID、相同 system/prompt 是否重叠；这只是精确去重，不检测语义近似题。短训练的 loss 不必单调下降，也不保证评测改善。

## 保存与重载

每次运行在 `artifacts/checkpoints/sft-lora-时间-随机编号/` 创建：

- `adapter/`：LoRA 配置与 adapter 权重，基础模型需要另外加载。
- `tokenizer/`：此次使用的 tokenizer 配置。
- `train_log.jsonl`：每步 loss、梯度范数、学习率和有效回答 token 数。
- `summary.json`：配置、依赖版本、基础模型提交版本、训练前后结果和重载检查。

保存后释放训练模型，再加载相同基础权重并挂载 adapter。相同输入在 eval 模式下的 logits 应在容差内一致。这个检查验证保存加载，不证明模型泛化。没有保存优化器状态，因此这份 adapter 用于推理或另开训练，不是完整续训 checkpoint。

## 测试范围

集成测试在本机创建随机初始化的极小 Llama，实际运行 PEFT、训练两步、检查基础参数不变、adapter 改变，再保存并重新加载。不联网、不下载预训练权重。未安装训练依赖时，该集成测试会跳过；安装后才会实际执行。正式 SmolLM2 权重训练需要另行运行脚本。

## 下一步

先读训练函数并运行一次，将训练参数数量、每步日志和 summary 结果对照代码。流程跑通后再扩充数据、增加结构化输出指标、设计训练步数和 LoRA 超参数对比。

## 参考资料

- [PEFT 官方入门：配置、保存与加载](https://huggingface.co/docs/peft/en/quicktour)
- [SmolLM2 模型说明](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
