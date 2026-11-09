# 恢复偏好数据已复核

最终169对（原44 + 规则84 + 原生模型41），train12，pairs SHA256 c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329。84上下文、252模型采样本地重建通过，最长1376token。

SQL未验证、重复证据错误、重复SQL错误和检查失败场景各36次均未产生有效进展。重复证据场景36次都重复上一错误动作，与12对规则偏好完全重复并被去重。

数据：artifacts/datasets/dpo-recovery-v1-20261102；统一编码：apt_v2.dpo_recovery.encode_recovery_pair。下一步固定恢复DPO训练起点、参考策略和步数。报告：reports/dpo-recovery-collect-v1-review-20261102.md。
