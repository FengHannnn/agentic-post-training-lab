# 恢复DPO已复核

169对一遍43更新（DPO11→DPO54），全部169相对margin为正，平均loss0.693147→0.597563。开发随机成功10/16→12/16；证据错误34→14。两条改善来自首条SQL正确，核心4恢复场景仍0/12；随机恢复29/84→31/84，贪心12/28保持。

本地64轨迹528动作+224恢复动作重放通过。候选适配器933ff3f97674d8b7a9bbafe6bc81292b8c0608ba7205cca3842f35559989820f已存artifacts/candidates/dpo-recovery-train-v1-20261102。共同SFT保持。报告：reports/dpo-recovery-train-v1-review-20261102.md。

建议下一步从同DPO11用同169对训练3遍，保持其余参数；可选chosen-only SFT对照。分支待用户选择。
