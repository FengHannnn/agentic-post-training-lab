# Phase 1：后训练目标函数基础

## 阶段目标

在引入训练框架和语言模型之前，先直接使用 PyTorch 张量实现并测试
SFT、DPO 和 GRPO 的底层计算机制。

## 已实现内容

1. 因果语言模型的下一 token 对齐：`logits[t]` 预测 `labels[t + 1]`。
2. 仅对回答 token 计算的 SFT 掩码交叉熵。
3. 用于偏好优化的掩码序列对数概率。
4. 使用冻结参考策略作为基线的 Sigmoid DPO。
5. GRPO 的组内相对奖励标准化。
6. 带可选 KL 惩罚的 PPO 风格 GRPO 截断策略损失。

## 测试覆盖的不变量

- 提示词 token 和填充 token 不参与回答损失计算。
- 回答掩码为空时主动报错，而不是静默产生 0 或 NaN。
- 增大 chosen 相对于 rejected 的策略概率差会降低 DPO loss。
- DPO 梯度会提高 chosen 的概率并降低 rejected 的概率。
- 非退化 GRPO 组的 advantage 具有零均值和单位方差。
- 组内 reward 完全相同时不产生学习信号。
- 策略概率比超出信赖区域时会被截断。
- KL 估计值保持非负。

## 本地运行

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -e .
& ".\.venv\Scripts\python.exe" -m pytest tests -v
```
