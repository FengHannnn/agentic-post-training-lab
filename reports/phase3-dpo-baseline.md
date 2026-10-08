# Phase 3 第 1 步：冻结 v2 SFT 基线

日期：2026-10-08。

## 本步目标

为后续 SFT-only 与 SFT＋DPO 比较固定起点。此处的“冻结”是独立复制 adapter、tokenizer、训练日志、数据、配置、实现和已有逐题评测，并记录 SHA256；不是重新训练，也不是将文件设为操作系统只读。快照仍可被修改，使用前必须校验。

原训练目录为 `artifacts/checkpoints/sft-lora-20261008-202445-a3d10354`。基线快照为 `artifacts/baselines/sft-v2-phase3`，其中 `checkpoint/` 可作为后续模型加载目录，`manifest.json` 记录来源、文件大小、哈希、运行条件和指标。已冻结的原逐题输出在 `evaluations/`。

基础模型为 `HuggingFaceTB/SmolLM2-135M-Instruct`，固定提交 `12fd25f77366fa6b3b4b768ec3050bf629380bac`。基础权重仍使用项目本地缓存，只记录哈希，没有复制进快照；迁移时需另行携带或获取这个提交的权重。此快照与原件位于同一磁盘，不替代异地备份。

Git 跟踪冻结脚本、本说明与 `artifacts/baselines/sft-v2-phase3/manifest.json`；完整快照和模型权重被忽略。manifest 记录冻结时的文件内容，之后更新工作目录的文档或脚本不会改变这份快照。克隆仓库只能获得清单，必须另行恢复清单对应的快照和固定 base 缓存，才能运行 `--verify` 或加载基线模型。

## 固定的训练事实

- SFT train/dev：144/32 条；batch=1，288 步。
- LoRA：q_proj/v_proj，rank=4、alpha=8、dropout=0。
- 学习率 0.001；seed=42；CPU float32、eager attention、2 个线程。
- 开发 loss：2.193550 → 0.029906。
- 历史保存重载验证：logits_match=true，最大绝对差 0.0。本步未重新执行模型重载。

## 冻结的行为记录

| 开发诊断 | v2 基线 |
| --- | --- |
| 原 20 题：JSON/schema/route | 20/20 |
| 原 16 道工具题：严格参数/数值正确 | 15/16 |
| 原 20 题：请求执行闭环 | 19/20 |
| 括号主诊断：严格 AST | 33/48 |
| 括号主诊断：数值正确 | 33/48 |
| 括号主诊断：成对两题数值正确 | 11/24 |
| Calculate 紧凑算式控制：数值正确 | 12/12 |

评测固定为 CPU float32、eager attention、贪心生成、num_beams=1、最多 64 个新 token，保留 raw_prediction，评分只去 tokenizer 特殊 token。继续使用严格 JSON/schema、AST、数值与成对指标，不修复模型输出。

这些数据全部属于已查看的开发诊断集。后续可以用于比较、分析退步，但不能作为未见最终测试成绩。第 2 步构造偏好数据时应保护它们的 ID、prompt 与精确表达式，不直接搬用这些题的正确答案作训练标签。

## DPO 必须从哪里开始

policy 与 reference 初始都必须是同一固定 base＋v2 SFT adapter。policy 在这份 adapter 上继续更新；reference 保持 v2 权重不变、关闭梯度并处于 eval 模式。不能只通过关闭 adapter 把裸 base 当作 v2 reference。

SFT-only 分支一直使用快照中的 v2 adapter；DPO 结果保存到新目录，禁止覆盖基线。第 3 步应验证 reference 无梯度、权重未变，policy adapter 有更新，以及 DPO adapter 保存重载一致。

本步记录的是 reference 初始化契约，尚未实现 DPO trainer 或在运行时冻结 reference。

## 阅读和运行

阅读顺序：本文 → `scripts/freeze_sft_baseline.py` → 快照 `manifest.json` 中的 training_summary、generation、dpo_initialization、files。

在项目根目录运行：

```powershell
# 仅在尚不存在快照时创建；已有目录时会报错，避免覆盖。
& ".\.venv\Scripts\python.exe" scripts\freeze_sft_baseline.py

# 后续使用前校验快照及固定 base 缓存。
& ".\.venv\Scripts\python.exe" scripts\freeze_sft_baseline.py --verify artifacts\baselines\sft-v2-phase3
```

哈希校验只证明内容与记录一致，不证明标签语义正确、指标泛化或模型行为必然可跨硬件复现。已有评测不是本步重新生成的结果。

## 下一步

第 2 步先定义小规模偏好样本的 system/prompt/chosen/rejected、参考表达式和来源记录；在新的输入上收集 v2 输出并构造可解释偏好。审计两条回答的 token 边界、长度、chosen 的工具执行正确性、rejected 的失败原因，以及训练/开发/受保护诊断集隔离。人工复查自然语言与参考表达式是否一致，再进入 DPO 训练。
