# C2 GRPO 首轮预检复核与 v2 修复

日期：2026-11-05，Asia/Shanghai。

## 结果

首轮在 NVIDIA L40S 上完成 prepare 和首组4条原生采样，四条轨迹均终局成功。作业在提交 worker 权重核验 RPC 时退出，耗时95.374秒，实际优化器更新0次。

本地使用回传冻结源码重放4条轨迹、28个动作，观察、奖励、证据、终局artifact和原生token校验全部通过。四条轨迹均7动作、回报0.93，组内标准差0，优势均为0；按既定规则，该组应跳过训练更新。

失败原因已定位为RPC载荷中的Python函数对象。v2改为注册worker扩展后，通过方法名字符串调用原有张量核验。108项本地测试通过，修复包已完成归档、逐文件哈希和干净目录检查。

## 回传与身份核对

| 项目 | 结果 |
|---|---|
| 原review ZIP | grpo-outcome-preflight-v1-20261105-review.zip，211个条目，CRC与路径检查通过 |
| ZIP SHA256 | 0cecdf24a6919b569b61ba18ec9f428a97ec923ebd52315ba7f452539f7012c8 |
| 冻结源码 | 181份文件与已发布v1清单逐项匹配 |
| 输入绑定 | 15份返回文件哈希匹配 |
| W adapter SHA256 | f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f |
| 云端GPU | NVIDIA L40S，UUID 92e14e85-628d-e5a4-af16-0db4defb2afb |
| 云端软件 | Python3.11.17、torch2.8.0+cu128、CUDA12.8、vLLM0.11.0、Transformers4.57.6、PEFT0.17.1 |
| 云端SQLite | 3.54.1，已保存当次运行身份 |
| 履历 | 起点SFT513/DPO11；本轮新增RL更新0 |

prepare核对了云端完整基座与tokenizer身份，本地复核了返回的输入绑定、tokenizer和实际W权重。唯一返回adapter为W输入，首轮未创建训练checkpoint和Adam状态。

原ZIP已封存至项目 artifacts/reviews/grpo-outcome-preflight-v1-20261105，并保留原始冻结源码及发布清单副本。

## 原生采样与概率

首组任务为train-000，采样参数K=4、temperature1、top_p1、top_k=-1。原生response共2096个token，最长prompt1137个token。完整轨迹的真实动作、观察、证据绑定、预算、EOS、采样种子和回报均通过本地重放。

首动作概率探针覆盖13个token。processed行为logp与同权重teacher-forced vLLM logp最大绝对差为0.0012126469518989325，通过原定0.25门槛。门槛与采样参数在v2保留原值。

终端显示了逐组rollout与逐动作进度。四条rollout约68秒；collect阶段共90.262秒，包含推理初始化及失败收尾。

## RPC原因与修复

原调用为 engine.collective_rpc(worker_tensor_proof, ...)。默认序列化器在请求发往worker前拒绝函数对象，云端异常为 Object of type function is not serializable。该异常发生在张量核验方法执行前。

v2在LLM初始化时设置 worker_extension_cls=grpo_vllm.GRPOWorkerExtension，调用改为 collective_rpc('grpo_verify_lora_tensors', args=(adapter_path, adapter_sha256))。worker内部继续执行已加载LoRA与目标checkpoint的112个A/B张量比较，回传形状、dtype及哈希。传输载荷使用方法名、路径、SHA字符串和数字，序列化环境设置保持原值。

worker扩展用于向worker注入collective_rpc命名方法，接口依据：[vLLM0.11官方说明](https://docs.vllm.ai/en/v0.11.0/api/vllm/config/parallel.html)。关闭推理引擎后同步清理主进程已初始化的分布式进程组。

## 修复验证

先在实际collector RPC表达式处建立基础类型传输测试，观察到函数对象序列化失败；改为方法名后通过。扩展注册测试先复现缺失注册，再验证修复后的配置与可导入方法。完整collector测试经过真实SQLite动作执行、命名RPC分派、112张量核验及组提交，输出4条成功轨迹和零方差信用。

最终回归运行：test_grpo_rpc_v2.py、test_grpo_v2.py、test_agentic_v2.py、test_hard_recovery_independent_v2.py、test_dpo_sft_start_v2.py，共108项通过，耗时83.65秒。

修复ZIP包含193个条目、192份绑定文件；CRC、逐项SHA256、Python语法、干净目录CLI入口、冷启动扩展导入和动态worker方法注册检查全部通过。下一云端作业将执行真实worker核验、有效梯度更新与刷新后再次采样。

## v2配置与预算

新run_id：grpo-outcome-preflight-v2-20261105。起点W、奖励、K、采样、学习率、KL、TIS、损失归一化与最大实际更新2次均保持原值。

首组零方差已得到实测。v2将有限训练组上限从4调整为8，按原有正常/恢复1:1调度覆盖更多train起点，达到2次实际更新后停止训练组推进，并执行最终版本刷新采样。

首轮已用95.374秒；v2时限3500秒，其中预留120秒打包。两轮配置上限合计3595.374秒，保持一小时预检边界。C2首批3个正式作业、合计12 GPU小时额度保留。

## 上传与运行

将 grpo-outcome-preflight-v2-20261105-run.zip 上传到云端v2目录，执行：

~~~bash
unzip -o /root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-preflight-v2-20261105-run.zip -d /root/rivermind-data/AgenticPostTrainingLab
cd /root/rivermind-data/AgenticPostTrainingLab/v2
/root/rivermind-data/envs/apt-vllm/bin/python -u scripts/run_v2_grpo.py --config configs/grpo-outcome-preflight-v2.json
~~~

该命令创建独立v2作业目录，保留v1目录、轨迹与原review。

结束后返回：

~~~text
/root/rivermind-data/AgenticPostTrainingLab/v2/grpo-outcome-preflight-v2-20261105-review.zip
~~~

修复ZIP SHA256：8a3292623e95725ff03264876e1f3ab026dab418093130cd2212019de6520ee3。

收到v2回传后，继续复核实际更新、Adam/RNG状态、采样版本、worker张量、刷新结果及显存/吞吐，再冻结正式训练配置。
