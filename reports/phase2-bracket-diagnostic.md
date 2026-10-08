# Phase 2：括号与文字问法诊断

## 设计

使用现有 v1（240 步）和 v2（288 步）adapter，不再训练，不修改数据标签。48 条输入由三组数字、四种问法、四种结构组成。

数字组 original=(2,6,5)，复查原失败题；unseen_small=(3,7,4)，检查另一组小数字；seen_large=(3,41,12)，包含 v2 已见训练表达式，作为已见样本探针。后者不是未训练评测题，逐题输出会标记是否见过训练输入和表达式。

问法 spaced 为带空格算式，compact 为紧凑算式，sequence 为明确运算步骤，natural 为 sum/product 等自然语言。

主实验的 spaced 使用 Calculate 前缀，compact 使用 Compute 前缀，两者同时改变前缀和空格，不能单独归因于空格。因此另加 12 条 compact_sameverb 控制题，使用 Calculate 加紧凑算式：与主实验 spaced 对比仅改变空格，与主实验 compact 对比仅改变前缀。

四种结构：a*(b+c)、a*b+c、(b+c)*a、b+c*a。同一数字组、问法、乘数位置构成一对括号/无括号题。24 对题中每对的期望结果不同，以便观察模型是否区分运算顺序。

## 指标

保留原严格 AST 匹配，另外报告数值语义正确：schema、路由、执行结果正确即可，允许交换加法或乘法顺序。这个指标只验证本组数值结果，不是通用符号代数等价证明。

成对通过要求括号与无括号两题都数值正确。否则模型始终输出同一个表达式，也可能偶然通过单题。

结果按问法、数字组、表达式结构分组，并保存逐题原始输出。不会修复 JSON 或帮模型补括号。

## 运行

```powershell
& ".\.venv\Scripts\python.exe" scripts\diagnose_brackets.py --run-v1 artifacts\checkpoints\sft-lora-20261008-180828-05ced0f0 --run-v2 artifacts\checkpoints\sft-lora-20261008-202445-a3d10354 --local-files-only
& ".\.venv\Scripts\python.exe" scripts\diagnose_brackets.py --run-v1 artifacts\checkpoints\sft-lora-20261008-180828-05ced0f0 --run-v2 artifacts\checkpoints\sft-lora-20261008-202445-a3d10354 --data data\eval\bracket_spacing_control_v1.jsonl --local-files-only
```

只加载一份基础模型，挂载两个 adapter 后依次切换，采用相同 tokenizer、模板和固定贪心生成。结果保存到 artifacts/evaluations/bracket-diagnostic-时间-编号/。

## 解读边界

若步骤描述比显式算式好，可以怀疑措辞影响；若已见大数题好、小数值未见题差，可以怀疑数据覆盖差异。此诊断同时改变了数字范围与训练是否见过，只能提出假设，不能区分数字大小本身的因果影响。v1/v2 也同时改变了数据和步数。

查看结果后，这些题也成为诊断开发数据，不能再视为最终未查看测试集。

## 实测结果（2026-10-08）

已在本机 CPU 上运行两份已有 adapter，每份测试 48 条主诊断与 12 条控制题，共 120 次生成。没有重新训练或修改 adapter。

主诊断中，严格 AST 通过数从 v1 的 28/48 提升到 v2 的 33/48；数值正确从 30/48 提升到 33/48。括号与无括号两题同时正确的对数从 9/24 提升到 11/24。

| 问法 | v1 数值正确 | v2 数值正确 |
| --- | --- | --- |
| 带空格算式，Calculate | 10/12 | 11/12 |
| 紧凑算式，Compute | 9/12 | 12/12 |
| 明确运算步骤 | 8/12 | 4/12 |
| 自然语言 sum/product | 3/12 | 6/12 |
| 控制题：紧凑算式，Calculate | 9/12 | 12/12 |

v2 对 `Calculate 2 * (6 + 5).` 输出 `2*6+5`，结果 17，而期望是 22。仅去除算式空格，改为 `Calculate 2*(6+5).`，便输出正确的 `2*(6+5)`。改为 `Compute 2*(6+5).` 也正确。因此在这组输入中可以观察到空格敏感性，而不能将成功单独归因于 Calculate/Compute 前缀变化。

右侧括号结构 `a*(b+c)` 的数值通过数从 1/12 提升到 9/12，左侧 `(b+c)*a` 两版均为 12/12。但无括号结构和部分步骤描述退步。例如 `Multiply 2 by 6 first, then add 5.`，v1 输出正确的 `2*6+5`，v2 只输出 `2*6`。说明 v2 的改善并非一致的运算顺序理解；本次结果与对模板、空格和措辞的依赖相符。

下一轮应覆盖同一运算的多种措辞、空格和乘数位置，同时成对加入“先加后乘 / 先乘后加”，避免只强化括号格式。使用新的数字和未用于训练的问法做开发评测，再保留另一份未查看的最终测试集。当前实验不能确定模型容量不足，也不能区分增加训练步数与改变训练数据各自的作用。

主实验逐题结果：`artifacts/evaluations/bracket-diagnostic-20261008-204902-76e6b53b/`。

单变量控制逐题结果：`artifacts/evaluations/bracket-diagnostic-20261008-210057-380ef02a/`。

代码回归测试：64 项通过。
