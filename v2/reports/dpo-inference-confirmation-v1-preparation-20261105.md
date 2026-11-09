# DPO 推理一致性确认：固定单LoRA与重复控制

2026-11-05，Asia/Shanghai。运行ID：dpo-inference-confirmation-v1-20261105。training_enabled=false，optimizer_updates=0。

E4返回记录显示续跑CE随机恢复60/84、chosen控制36/84，随机完整任务7/16对8/16，加权SFT16/16。记录重放通过，同时发现相同adapter权重、相同FP32原始评分在多LoRA原生推理中出现greedy恢复5/28条轨迹分叉，2条终局不同。当前先确认推理结果的重复性，C1表现验收未通过，GRPO暂缓。

## 固定协议

4个分支按固定顺序在4个独立子进程执行：weighted_sft、chosen_control、control_repeat、continuation_ce。control_repeat与chosen_control指向同一份真实检查点与权重；每进程仅创建1个推理引擎、加载1个adapter，max_loras=1，LoRA名称fixed_adapter、编号1。

推理仍为vLLM BF16、max_num_seqs=1、eager、无prefix cache、raw_logprobs，沿用E4任务/上下文/温度/种子/max_new_tokens/max_model_len。每个进程先完整任务，再完整恢复，内部顺序一致。prepare将adapter复制至新作业私有目录，仅把adapter_config.json中的target_modules列表排序；原权重字节保持SHA一致，原模型文件保留。所有模型规范化后配置哈希相同，参数权重分别固定。

3个唯一真实模型：加权SFT f8fb95…、chosen控制 ad6a8f…、续跑CE b74af1…。来源为已审E4及加权SFT，prepare固定E4的audit/config/run_status SHA256，逐个核对原模型complete与adapter。模型、软件和输入在作业前后复查。运行环境沿用E4型号/软件/精度，跨作业有效UUID变化单列记录；作业内严格核对同一GPU身份。

各分支：greedy完整任务4条、随机完整任务16条、greedy恢复28条、随机恢复84条，即132条。共80条完整任务+448条恢复=528条；其中chosen_control与control_repeat组成132对相同权重、输入、采样种子和执行顺序的重复评测。

此作业确认固定协议下的结果与重复性。可能来源包括多LoRA加载/编号与进程状态、配置列表顺序和引擎重复性；这些来源依据原始分叉证据及本次固定条件的输出继续定位。

## 通过条件与输出

严格重放全部528条原生轨迹和448个首动作，重算calendar/证据链与配对统计。同权重重复门槛：chosen_control与control_repeat的整条记录在去除分支标签及raw logp浮点值后完全相同，包括prompt/response token IDs、文本、采样种子、工具动作/观察、成功与artifact。原始logp在对齐动作上的最大差异单列报告。

repeatability.json保存相同/不同轨迹数量和键；重复性失败时audit.json的passed=false并报错，作业仍回传完整日志。通过后使用同一固定推理协议比较SFT、chosen控制、续跑CE；随机完整任务和完整恢复分别报告。

不重新训练、不给新模型做参数搜索，不重算已审FP32偏好/专家探针。保留此前完整训练和诊断评分。新作业14400秒总时限、实时进度、已有同名输出保护、失败/超时打包。返回包保存原生记录、运行元数据、规范化配置及源模型哈希；私有adapter复制件留在云端，原模型字节不变。

## 本地验证

4项检查通过，覆盖配置规范化的字段边界、重复控制及篡改拒绝、原生完整重放、单adapter固定编号的实际入口调用、零更新失败归档。编译与3个CLI入口通过。真实输入预检核对3份唯一adapter，确认chosen_control/control_repeat的权重与规范化配置哈希一致，28个development上下文符合token预算；实际GPU推理待云端执行。

验证回执：handoff/dpo-inference-confirmation-local-validation-20261105.json。E4报告：reports/dpo-continuation-ce-paired-v1-review-20261105.md。

## 执行

上传dpo-inference-confirmation-v1-20261105-run.zip至云端v2后：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/dpo-inference-confirmation-v1-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_dpo_inference_confirmation.py
~~~

完成后回传dpo-inference-confirmation-v1-20261105-review.zip。保留E4 paired与加权SFT的原artifacts供模型身份验证。
