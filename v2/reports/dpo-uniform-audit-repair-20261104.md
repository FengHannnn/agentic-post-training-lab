# E2 审计补跑说明

原运行 dpo-uniform-control-v1-20261104 已完成 258 次训练更新和 528 条评测轨迹；原审计停在跨次 reference 单 token 绝对差 0.000131845 > 1e-4。模型、tokenizer 与 token 对齐匹配，本次重新评分与本次训练缓存差为 0。补丁使用绝对+相对 token 容差，并增加平均、序列与 DPO loss 影响上界检查。

补丁仅新增独立审计入口和数值校验模块，复用云端现有训练、评分和轨迹。实际 Adam state、checkpoint 与全部评分和轨迹由审计器重新验证，新增训练/生成更新为 0。

上传 dpo-uniform-control-v1-20261104-audit-repair-run.zip 至云端 v2 后执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-uniform-control-v1-20261104-audit-repair-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_uniform_audit_repair.py
~~~

补跑入口只调用 audit_v2_dpo_uniform_repair.py；上限 1,800 秒，日志 audit-repair.log。原 artifacts/dpo-uniform-control-v1-20261104 必须保留；使用现有环境和源码，先运行本补丁，再上传下一阶段的源码包。

预期回传：dpo-uniform-control-v1-20261104-audit-repaired-review.zip。

完成状态查看 audit_repair_status.json 的 error=null 和 audit.json 的 passed=true。原 run_status.json 保留首次失败，便于追溯。

补跑前核对原返回包中所有文件哈希、原冻结源码、补丁 manifest；补跑后再次检查全部原产物哈希。原 review ZIP 存在时也核对其 SHA256。审计结果写入新增文件；若审计失败，仍打包修复日志和状态。已有补跑状态或同名补跑 ZIP 会拒绝重复运行。

新 reference_numerics_audit.json 记录旧绝对门槛是否通过、修复策略、全部数值误差与两个 reference 文件哈希。当前实际最大 DPO loss 影响上界约 2.10e-5；训练缓存按原哈希保留。

本地已严格复核 528 条轨迹、3,202 步，以及四模型偏好与专家探针评分。结果报告 dpo-uniform-control-v1-review-20261104.md：均匀 DPO 随机完整任务 2/16、完整恢复 33/84；原加权 DPO 同轮分别 1/16、15/84。C1 继续未通过。
