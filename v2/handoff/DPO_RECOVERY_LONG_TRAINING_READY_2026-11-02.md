# 三遍恢复DPO已准备

用户选择增加DPO训练量，同时提高时限并显示终端进度条。新run dpo-recovery-train-v2-20261102，从同DPO11初始化和固定参考，169对3遍=507曝光，129更新，新的AdamW，lr5e-6 beta0.1。累计DPO140。时限10800秒（180分钟）。

终端tqdm显示参考缓存、评分和训练进度；streamed_process同时写终端和日志，training_steps.jsonl保存每步数据。checkpoint43、86和final129。评测沿用64完整轨迹+224开发恢复动作及原seed，便于和v1比较。102项测试通过。

云端运行python -u scripts/run_v2_dpo_recovery_train_long.py。等待dpo-recovery-train-v2-20261102-review.zip。报告：reports/dpo-recovery-train-v2-20261102.md。
