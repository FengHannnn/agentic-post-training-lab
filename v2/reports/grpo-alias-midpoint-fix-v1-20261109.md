# GRPO alias midpoint fix v1

## 本轮状态

返回包通过 ZIP CRC 检查，共 2,231 项。两支均已提交 48 组：control 实际更新 25 次，curriculum 实际更新 37 次。第 24 组两次中点诊断已成功，分别耗时 297.09 秒与 304.32 秒；尚待两支最终评测、策略刷新和总审计。

原作业累计运行 14,489.50 秒（4 小时 1 分 29.50 秒），原六小时预算剩余 7,110.50 秒（1 小时 58 分 30.50 秒）。续跑预计约 1.5 小时，仍沿用原六小时累计上限。

## 原因与修复

驱动在固定第 24 组边界执行中点后，在完成训练的收尾位置又调用一次 midpoint()。生产日志使用排他创建模式 x，同一 attempt 的第二次调用因日志已存在而报 FileExistsError。

本次删除收尾处的重复调用，保留第 24 组边界诊断及日志排他创建保护。测试日志由覆盖写改为与生产一致的排他创建，并加入“两支训练均完成后的失败续跑”用例。旧代码已复现相同报错，修复后复测记录见随包 validation JSON。

本轮训练配置、信用、学习率、任务、评测计划、权重和优化器状态保持原值。安装器先校验原运行包及根目录/两支冻结源码，再备份旧文件，仅更新驱动、测试及对应源码哈希绑定，记录 source_migrations。所有原有训练与评测产物、配置、计时在安装前后逐文件核对。

修复记录与旧文件备份：artifacts/grpo-alias-curriculum-v1-20261109/hotfixes/alias-midpoint-fix-v1/。安装器可重复执行，已安装时校验后返回 already_applied。

## 验证结果

生产日志排他创建规则下，旧驱动已复现同名日志 FileExistsError。修复后 6 项驱动回归通过，另 1 项已完成训练后的续跑回归通过；安装器 7 项保护检查通过。真实返回包副本安装前后 1,466 个原有产物的 SHA256 保持一致，48/48 组、25/37 次更新和 14,489.50 秒累计计时保持原值。云端安装时同样核对现存 state.pt 等文件。

## 使用方式

将 grpo-alias-midpoint-fix-v1-20261109.zip 上传到 /root/rivermind-data/AgenticPostTrainingLab，执行：

    cd /root/rivermind-data/AgenticPostTrainingLab &&
    unzip -o grpo-alias-midpoint-fix-v1-20261109.zip -d . &&
    cd v2 &&
    /root/rivermind-data/envs/apt-vllm/bin/python -u hotfixes/alias-midpoint-fix-v1/apply.py &&
    /root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo_alias_curriculum.py --config configs/grpo-alias-curriculum-v1.json --resume

安装通过后，--resume 沿用已完成的 48 组策略和原计时，进入两支最终评测、刷新与审计。进度条继续在终端显示。使用同一实例上的原 artifacts 目录，其中保留完整 Adam/RNG 状态。

下一返回包：/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-alias-curriculum-v1-20261109-review-attempt-002.zip。

## 代码阅读（仅本次新增或改动）

| 文件相对路径 | 大致作用 |
| --- | --- |
| scripts/run_v2_grpo_alias_curriculum.py | 删除训练收尾的重复中点调用，续跑直接进入最终评测。 |
| tests/test_grpo_alias_driver_v2.py | 使用生产日志排他创建规则，验证中点仅调用一次及已完成训练后的续跑。 |
| hotfixes/alias-midpoint-fix-v1/apply.py | 校验、备份和安装修复，更新冻结源码绑定并证明实验产物与计时保持原值。 |

报告项目路径：v2/reports/grpo-alias-midpoint-fix-v1-20261109.md。
