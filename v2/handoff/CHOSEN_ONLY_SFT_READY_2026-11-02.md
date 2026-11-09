# Chosen-only SFT对照已准备

用户已选择chosen-only SFT。初始同DPO11，169chosen行（123独特context/response）保留原权重，3遍129步、507曝光、lr5e-6、fresh AdamW，时限10800s，实时进度。每遍17581监督token，最长1376。rejected/参考不参与loss。

同次三模型：before DPO11，dpo DPO140，after chosenSFT。96完整轨迹+336开发恢复动作，同原seed。审计对齐原生编码与初始chosen分数。107测试通过。

运行python -u scripts/run_v2_chosen_sft.py；待chosen-only-sft-v1-20261102-review.zip回传。报告：reports/chosen-only-sft-v1-20261102.md。
