# C3 matched GRPO contract

日期：2026-11-06。运行：grpo-c3-matched-v1-20261106。用户确认本轮最多 4 小时，包含两支训练、全部评测和审计。

## 共同条件

outcome 与 fine 均从 W 独立初始化新 AdamW，冻结 W reference。W SHA256=f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f，谱系 SFT513/DPO11/RL0。两支各 24 组、K4、normal/recovery 各 12 组，采样 seed20261105、T1/top_p1/top_k-1、最大响应256/上下文8192，沿用 train12 和原12轮预算。

LR1e-6、clip.2、beta.02、TIS cap2、grad clip1；每组最多更新一次，更新后由新进程加载不可变 checkpoint 采样，实际更新数分别记录。终局回报均为 success−.01×总动作数；prefix计入预算，只训练模型续跑动作。损失保持 token→action→episode 均值。每组严格重放、原生 processed q、HF old/ref 重评分、same-path gate、112张量 worker 证明、冻结 reference、Adam/RNG 事务沿用 C2。sampled k3>.05 或 clip fraction>.5 跳过更新。

## 信用变量

outcome 使用同组轨迹回报标准化优势，整条轨迹共享。

fine 方法名 potential_rtg_depth_v1：Phi(s)=.2×已验证指标数，终局 Phi=0。逐动作 shaped reward=outcome_reward+Phi(next)−Phi(current)，G(i,t) 为从该续跑动作至终局的 shaped return-to-go。相对动作位置 t 上，仅取仍有该动作的组内轨迹，计算 population mean/std，A(i,t)=(G(i,t)−mean_t)/(std_t+1e-8)。有效轨迹不足2条或该位置零方差时 A=0；所有位置均无区分信号则整组跳过，不重采样。

完整过程奖励满足 potential telescoping；恢复续跑总 shaped return 相对 outcome 续跑回报相差共同起点的 Phi 常数。第一续跑动作的组内优势与 outcome 对齐，后续动作按可验证进展与剩余回报分别分配信用。fine 允许终局回报相同、过程进展时序不同的组参与更新。比较同时报告终局零方差组与逐动作有效组，因此组数预算相同，实际更新数可不同。本轮为纯 GRPO，未混入 SFT/DPO/CE。

## 同机评测

在新的同一张 L40S 上评测 W、outcome最终、fine最终，每模型132个标准开发episode、14次greedy质量动作，共396个episode/42质量动作。仅评测最终24组节点，不新增中间开发选择。seed、任务、场景、温度与C2协议相同，报告相对W及fine相对outcome的完整配对变化。完整成功率饱和时以sample recovery为主要观察指标；数值提升/回归阈值仍待后续确定，C4执行新任务最终确认。

## SQLite真实依赖扩展

新增 closing-development 四个任务（独立seed8000+，16轮上限）。先核对current/prior/change，close_period检查三个数值、policy/schema和evidence，写入SQLite period_closings表；下一账期SQL读取该结转行与ledger_movements，生成next_balance_cents已验证证据，最终submit绑定closing_id和四项数值/证据。

下一账期SQL在两个独立扰动世界复核；业务SQL为只读，写入仅由服务端close_period执行。脚本专家4/4完成，删除结转行使下一查询无结果，修改结转净额改变下一余额并触发完整性校验。扩展本轮用于开发迁移评测，训练继续使用原任务分布；每个模型另做4条greedy完整扩展episode，共12条。新增业务表现单列，C4新任务确认保留。

## 新L40S实例与预算

允许新实例的L40S UUID与C2不同。prepare核对完整base/adapter/input/software哈希与GPU型号/容量，并采集新身份；两分支在同一新UUID与软件环境执行，作业内保持严格身份。另保存nvidia-smi驱动记录。历史C2输入profile仅作为来源与开销依据。每轮旧成果保持原冻结快照。

总wall-time14400秒，预留120秒打包；包括双分支共48训练组/192训练episode、8条最终刷新、396条标准开发episode、42次质量动作和12条依赖业务episode。按已测L40S开销估计约3–4小时。中断按已提交事务恢复，两支共享已耗时间；超时/异常生成review ZIP。state.pt留云端，review返回哈希和云端复核记录。终端显示104个总阶段及采样、评分、反传、评测进度。
