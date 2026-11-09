# V2 local artifacts

本目录保留训练数据、权重、Adam/RNG、评分缓存、轨迹与审计回包。运行产物由 Git 忽略；本说明进入 Git。

正式 W 输入为 inputs/grpo-w-20261105/adapter，adapter SHA256 为 f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f。历史纯 SFT 索引位于 baselines/current_sft_baseline.json，历史候选登记位于 candidates/。

最新源码与发布绑定见 ../grpo_alias_release_manifest.json；导入记录见 ../handoff/grpo-source-import-20261109.json。冻结产物按各轮报告中的相对路径定位。Git checkout 后，训练输入与完整证据按各轮归档恢复。
