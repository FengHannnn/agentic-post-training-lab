# Fine GRPO 实验结项

2026-11-13。多轮 SQLite 对账实验选定 **原 local Fine GRPO（RL43）** 为 Fine 候选，W 作为正式基线。12 个独立任务族的配对复评中，RL43 的完整采样和恢复采样分别比 W 多成功 2 条和 6 条，并通过预先冻结的验收。

## 最终复评

| 标准指标 | W | 原 local RL43 | 新收尾信用 RL47 |
| --- | ---: | ---: | ---: |
| 完整采样（主指标） | 24/48 | 28/48 | 33/48 |
| 恢复采样 | 177/252 | 183/252 | 196/252 |
| 完整 greedy | 11/12 | 12/12 | 12/12 |
| 恢复 greedy | 66/84 | 67/84 | 70/84 |

新收尾公式在两批各 48 组的固定轨迹上分别多产生 7 个有更新信号的组，每批都有 32 次检查、40 次提交从零优势变为正优势。在线训练后，新信用策略比原 local 多完成 5 条完整采样和 13 条恢复采样。两支都拿齐三项认证的 205 条配对恢复轨迹中，提交成功净变化为零。恢复成功数的净差集中在证据未齐的轨迹，按场景分布在重复证据错误、格式错误和重复 SQL 错误。[固定轨迹重算记录](grpo-fine-closeout-credit-results-validation-v1-20261112.json) · [三方复评报告](grpo-fine-closeout-followup-results-v1-20261113.md)

## 权重与回包

| 用途 | 策略 | Adapter SHA-256 |
| --- | --- | --- |
| 正式基线 | [W](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/w/initial/adapter/adapter_model.safetensors) | `f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f` |
| Fine 候选 | [原 local RL43](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/local/checkpoints/group-0046/adapter/adapter_model.safetensors) | `eb78febd0fd2cce0cdac6a930894d59eb07cada9b320ecaf1249f1932f9f708e` |
| 信用对照 | [新收尾信用 RL47](../artifacts/grpo-fine-closeout-followup-v1-20261112/branches/closeout/checkpoints/group-0047/adapter/adapter_model.safetensors) | `1c209b708c7cdf334f6ec103e0f5974193078e9d6f3acc8773654b37faec2b5a` |

三支 Adapter 已逐一核对 SHA-256。两个原始 review ZIP 已复制到项目内归档目录，归档副本的字节数和 SHA-256 均与收据一致。

| 作业 | 实际墙钟 | 回包与收据 |
| --- | ---: | --- |
| 信用对照训练、评测及首轮独立复验 | 6 小时 5 分 13 秒 | [ZIP](../artifacts/review-archives/grpo-fine-closeout-credit-v1-20261112-review-attempt-001.zip) · [收据](../artifacts/grpo-fine-closeout-credit-v1-20261112/returned_packaging_receipt.json) |
| 12 任务族三方独立复评 | 4 小时 42 分 59 秒 | [ZIP](../artifacts/review-archives/grpo-fine-closeout-followup-v1-20261112-review-attempt-001.zip) · [收据](../artifacts/grpo-fine-closeout-followup-v1-20261112/returned_packaging_receipt.json) |

两轮耗时均含审计和打包。三方复评的 6 小时预算余量为 1 小时 17 分 01 秒。本次结项新增 GPU 用时为 0。

## 本地重放

从 `v2/` 运行下方 PowerShell 命令。使用当前仓库的 Python 虚拟环境、项目内 SQLite 3.54.1 和已解压的三方回包。冻结输入中的来源证据也参与校验。命令会重放已保存的评测、重新计算配对与验收，再更新回包目录中的 `local_audit.json`。[本地环境说明](../README.md#本地验证) · [冻结输入来源证据](../artifacts/inputs/grpo-fine-closeout-followup-20261112/source-evidence.json)

~~~powershell
@'
import ctypes
import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sqlite_runtime = ctypes.WinDLL(r".local/sqlite3541/sqlite3.dll")
import sqlite3

assert sqlite3.sqlite_version == "3.54.1"
sys.path.insert(0, "scripts")
from tqdm import tqdm
from run_v2_grpo_fine_closeout_followup import audit

output = Path("artifacts/grpo-fine-closeout-followup-v1-20261112")
with tqdm(total=1, desc="Independent evaluation replay", unit="audit") as progress:
    local = audit(output, local=True)
    cloud = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert local["scope"] == "local_returned_files"
    assert cloud["scope"] == "cloud_evaluation"
    assert {key: value for key, value in local.items() if key != "scope"} == {
        key: value for key, value in cloud.items() if key != "scope"
    }
    assert local["passed"] and local["new_training_updates"] == 0
    progress.update()
print("Selected candidate:", local["selection"]["selected_candidate"])
'@ | & '..\.venv\Scripts\python.exe' -u -
~~~

本次本地重放耗时 2 分 44 秒，选择结果和配对统计与云端一致。信用分配与三方复评的定向回归测试 12 项通过，pytest 耗时 6.48 秒。 V2 全量回归在运行中停止，本次结项以定向测试和三方重放为验证范围。[本地审计](../artifacts/grpo-fine-closeout-followup-v1-20261112/local_audit.json) · [云端审计](../artifacts/grpo-fine-closeout-followup-v1-20261112/audit.json) · [复评验证记录](grpo-fine-closeout-followup-results-validation-v1-20261113.json)
