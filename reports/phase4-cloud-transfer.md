# Phase 4：云端环境检查与固定产物迁移（2026-10-19）

## 当前验证结果

用户提供的上海3区 4090D 云终端截图确认安装完成：torch 2.8.0+cu128、torch CUDA 12.8、vLLM 0.11.0、transformers 4.57.6、PEFT 0.17.1。pip check 输出 No broken requirements found；torch.cuda.is_available() 为 True；CUDA 张量相加返回 [2.0,2.0,2.0,2.0]。这是依赖导入和基础 CUDA 检查通过，尚未验证 vLLM 引擎初始化、模型加载或 rollout。

云环境与本地 CPU 环境 torch 版本不同。现阶段不执行 pip install -e .，避免项目 torch>=2.14,<2.15 的依赖声明触发不匹配升级；后续用项目根目录的 src 作为 PYTHONPATH 导入现有模块，版本差异记录在产物中。

## 迁移内容与核验

准备一个 ZIP，顶层目录为 AgenticPostTrainingLab，包含 Git 跟踪的代码/配置/数据/报告、完整冻结基线、manifest 对应的 8 个基础模型缓存文件、旧 CPU rollout 目录及 reward 审计 JSON。缓存文件按实际内容写入 ZIP，避免 Windows 链接在 Linux 上失效。排除 Windows .venv、.git、无关 checkpoint 和其他缓存。

cloud_bundle_manifest.json 为本次迁移清单，记录源码提交和每个文件 SHA256；清单自身不自哈希。外层 ZIP 的哈希另外记录。迁移清单不替代历史冻结清单：verify_cloud_bundle.py 在检查迁移文件后，继续调用原 freeze_sft_baseline.verify 核对冻结基线和模型缓存。它不加载模型、不重新证明历史评分或概率。

先在云端 JupyterLab 文件浏览器进入 /root/rivermind-data，上传 ZIP。终端执行以下步骤，解压目标必须尚不存在，避免与旧文件混合：

```bash
conda activate /root/rivermind-data/envs/apt-vllm
cd /root/rivermind-data
test ! -e AgenticPostTrainingLab && python -m zipfile -e phase4-cloud-bundle-20261019.zip .
cd /root/rivermind-data/AgenticPostTrainingLab
python scripts/verify_cloud_bundle.py
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

若目标目录已经存在，不覆盖，先检查来源；若文件校验失败，不继续模型加载。云端解压与校验尚待用户执行。上传不需要重新下载基础模型，不在公网发布 adapter。

## 代码阅读顺序

1. `scripts/verify_cloud_bundle.py` → `file_hash`、`main`：新增入口，流式计算文件哈希，检查路径未越界以及文件完整，之后调用历史冻结核验。
2. `scripts/freeze_sft_baseline.py` → `verify`：已有代码，检查冻结 snapshot 的 files 和固定基础模型的 base_cache_files。注意迁移包保留其相对缓存路径。
3. `cloud_bundle_manifest.json`：迁移包内生成的数据，查看 source_commit 和 files，理解迁移身份与模型推理验证的区别。
4. `src/agentic_post_training/rollouts/hf_sampling.py`：按需回看 CPU 限制；本次尚未把 CPU 采样器改为 GPU，不直接调用它来冒充新引擎验收。

完成上传和核验后，下一步才实现并执行固定 SFT 的 vLLM 最小加载与原生 token 采集。仍属于 Phase 4，不开始真实 GRPO 更新。

## 本地实际迁移验证

已生成 phase4-cloud-bundle-20261019.zip，280241467 字节（约 280MB），包含 245 个有哈希记录的文件；包内源码提交 d14c1331147924de61e41d6a29cfaf02e5d2e7b8。ZIP SHA256 为 d87e010eb28acd12191123827a9996385d6219d39bb4fcc898183bf4b4696f82。

已在独立本地目录完整解压并实际运行 verify_cloud_bundle.py：245 文件校验通过，历史基线 74 个文件与 8 个模型缓存文件通过。额外修改解压副本中的 README 后，核验正确拒绝，随后恢复副本。此验证不依赖 GPU，不代替云端上传后的核验或模型加载验证。包内报告为打包时版本，当前报告补充了这些打包后的检查结果；后续文档提交不改写原 ZIP 的源码身份。
