# Phase 4：随机 rollout 数值差异定位与有限路径诊断（2026-10-20）

## 原始证据与重审结果

用户上传 rollout-vllm-review-20261020.zip，35 个目录/文件条目，展开文件合计 194748 字节。已检查解压路径并完整保留到本地 artifacts/trajectories/rollout-vllm-phase4-20261020-01。原始采样、重算、reward、mask 和源文件快照不修改。

本地逐项检查原始源码快照与当前源码 SHA256 一致、基线 74 个快照及 8 个基础模型文件一致、规则评分/解码、长度 mask 和保存张量一致、reward/advantage 一致、概率误差和 summary 一致。因 Linux 保存的源码键使用 /，本地 Windows 库存使用反斜杠，本地审计包装层仅统一路径键分隔符，未修改任何原始清单或源码。原 cloud_bundle_manifest.json 未包含在这次回传 ZIP，本地使用已知冻结清单核验，不声称重新检查了最初上传包的完整 envelope；原云端该检查已通过。

记录审计通过，数值门未通过，这两个结论不矛盾。原始 rollouts.jsonl SHA256=d81a6ef5a6de2b04e606de12df80fc2ae18c26aa04a17610fdeccaa05abf64be；replay_checks.json SHA256=63533683ea805b55d9eef71f415afc05f64c0fd125f3d8d12ab3be7459da0592。

## 逐 token 实际定位

从原始 JSON 得到最大误差 0.023653745651245117。此前将截图读为 0.03265，是读数错误，已在原报告纠正。378 个回答 token 中，24 个超过 0.001；它们分布在 10 条回答中。

| 组 | 超限回答/4 | 组内最大逐 token 误差 |
| --- | ---: | ---: |
| symbolic | 0 | 0.0005218470469117165 |
| natural | 4 | 0.02228379249572754 |
| sequence | 3 | 0.013725042343139648 |
| greeting | 3 | 0.023653745651245117 |

最大误差位于 greeting 的 sample_index=1，第 8 个回答 token（零起始 index=7）`Hope`：vLLM raw logp=-2.0386781692504883，HF FP16 完整前向=-2.015024423599243。另一明显位置是 natural 的 sample_index=0，第 18 个 token `2`：-0.354079931974411 对 -0.33179613947868347。两者不是仅 JSON 前缀的数值变化。

| 按 vLLM 的 -logp 分区 | token 数 | 超限 token 数 |
| --- | ---: | ---: |
| [0,0.01) | 323 | 0 |
| [0.01,0.1) | 25 | 4 |
| [0.1,1) | 17 | 13 |
| [1,3) | 8 | 4 |
| [3,100) | 5 | 3 |

误差集中于模型较不确定的回答 token；高置信度 token 全部低于上限，解释了单题贪心探针为什么可能通过而随机采样失败。这里没有证明“低概率导致误差”这个独立因果关系。

natural 组四条的 prompt/response ID 完全相同，但 vLLM 在 `2` 位置的 raw logp 分别约为 -0.35408、-0.34877、-0.35306、-0.34877；HF 完整重算相同，均约 -0.33180。相同 token 历史的分数在请求间变化，支持检查批处理/内核/缓存的数值路径，但不能仅据此排除 LoRA 路径差异或引擎实现问题。不能只凭一次训练样本上的误差就提高上限，也不能认定发生了统一的一位 shift。

## 下一次有限诊断

configs/rollout_numeric_diagnostic.json 固定原始五份文件哈希、原阈值 0.001，以及四条代表记录的 flat index=[0,4,9,13]：symbolic sample0、natural sample0、sequence sample1、greeting sample1。选择依据是已经观察到的失败位置，是事后诊断，不是新开发评测。

scripts/diagnose_rollout_numerics.py 对这四条已有 token 做三条 HF 评分路径：

1. **FP16 完整前向**：重新执行原评分路径，并与旧 replay 分数比对，检查重复评分的一致性。
2. **FP16 KV cache 逐步强制评分**：先输入 prompt，逐步喂入已经记录的回答 token，提取每个实际 token 的 logp；绝不调用 generate、argmax 或采样。与完整前向对比，隔离 HF 内部缓存/分步执行的影响；再与 vLLM 保存分数比较。
3. **FP32 计算、保留 FP16 舍入权重**：把同一个已加载 FP16 模型 model.float()，不重新加载原始 FP32 adapter，以保持已舍入权重数值不变。禁用 TF32，完整前向评分，观察计算精度变化的影响。它不是原始 CPU float32 模型的严格复现，也不是重新生成一批 FP32 回答。

每条路径均包含相同 EOS，原生 token 不变，模型冻结，无优化器和反向；总共只处理四条记录。比较六组误差，不宣称任一路径天然是绝对真值。若 FP16 full/cached 差异显著而 cached 更接近 vLLM，说明执行路径有贡献；若改 FP32 计算改变分数，说明计算精度有影响；都不能自动证明它们是唯一因素。

原 16 条实验的 rollout_interface_checks_passed=false 保留，不修改阈值、不覆盖原始重算。诊断退出 0 表示诊断数据已保存，不表示原数值门被重新判为通过；training_admitted=false。此次只做这一轮有限路径对照，若仍不能解释，再基于结果制定下一小步，避免无边界精度/引擎调参。

## 云端执行

上传 phase4-numeric-diagnostic-update-20261020.zip 到 /root/rivermind-data，保持同一 L40S 环境：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
python -m zipfile -e phase4-numeric-diagnostic-update-20261020.zip .
cd AgenticPostTrainingLab
python scripts/diagnose_rollout_numerics.py
```

输出 artifacts/evaluations/rollout-numeric-diagnostic/diagnostic.json，已存在则拒绝覆盖。复制最后终端摘要即可先查看各路径误差，随后取回 JSON 复核。无需重新采样、安装依赖或重新上传权重。

## 代码阅读顺序

1. `configs/rollout_numeric_diagnostic.json`：五份输入哈希、四条代表记录及原始误差限。
2. `scripts/diagnose_rollout_numerics.py` → `full_logps`、`cached_logps`：相同序列如何在完整/缓存路径评分，特别看喂入的是原记录 token，以及 mask 长度随缓存递增。
3. 同文件 `main` 前半：原实验重审、身份与 GPU 版本、四条记录的选取。
4. 同文件 `model.float()` 与比较输出：区别计算精度、权重舍入和原始 checkpoint 精度，理解诊断通过不等于训练准入。
5. `tests/unit/test_rollout_numeric_paths.py`：随机 tiny Llama 上完整/缓存评分一致，首条回答和 EOS 均覆盖；钩子检查每次只喂入正确的前一个 token，禁止调用 generate。

## 验证状态

本地新增 7 项 tiny 模型/边界测试通过；全套 218 项通过，无 skip，1 条已有 tiny PEFT 配置提示。未进行本地 GPU 或正式模型评分。L40S 四条路径诊断尚待用户执行。
