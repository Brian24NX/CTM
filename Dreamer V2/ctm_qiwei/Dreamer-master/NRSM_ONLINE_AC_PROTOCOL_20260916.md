# NRSM 在线 Actor-Critic 完整链路验证

## 决定与边界

用户明确要求把 AC 接入完整验证。沿用上一轮的一小时预算，但这次不是固定离线
世界模型训练：真实环境采样 → 回放 → NRSM 世界模型 → 想象 Actor-Critic → 环境。
本轮从零初始化，不继承上一轮仅世界模型的 checkpoint。原 CTM 核心、NRSM 适配层、
DreamerV2 训练代码、场景、奖励、MOVE/TURN 自动取货语义均不修改。

采用 project-architecture 的职责分离：

- `nrsm_online_agent.py`：复用已验证世界模型与混合动作分布，增加 AC 与完整模型 checkpoint。
- `train_nrsm_online.py`：在线采样、episode 对齐回放、有限时训练、任务评估、保存与恢复验证。
- `start_nrsm_online.py`：冻结源码后启动独立、限时的 GPU 进程。

备选方案：改写旧 `DreamerV2` 类 / 继承并替换其中 RSSM，会同时影响历史 baseline 的
初始化、直接状态 Actor、BC 和回放契约，风险高。保留离线验证则无法满足完整链路要求。
因此采用独立入口，复用核心与公共数学函数，不迁移或覆盖旧结果。

## 配置

- compact NRSM：D64 / M8 / K2 / sync16 / context128 / stoch8×classes8。
- seed17；batch2；回放是最多 500 个完整 episode 的 FIFO，均匀抽 episode，补齐到101行。
- 随机动作预填至少1000环境步，按完整回合结束，可能略超出；无控制器示范或 BC。
- 世界模型预热50次（单独计数）；此后每5个 Actor 在线环境步联合更新1次。
- 每次均匀有放回抽32个有效非终止 posterior 状态作为想象起点，H=15。
- Actor / Critic：两层128单元。Actor 在执行与想象中均读取 `stoch+deter`，没有真实 vector 旁路。
- 混合动作梯度沿用项目实现：MOVE/TURN 离散分支 REINFORCE，连续参数通过模型 dynamics 梯度。
  imagination_scale=.1，parameter_grad_scale=.1，分支熵=.03，参数熵=.02，unimix=.05。
- gamma=.99、lambda=.95；模型直接预测 gamma×continuation，不再次乘 gamma。
- Critic 目标来自 successor 对齐的 lambda return，权重使用停止梯度的预测存活概率乘积。
  每100次 AC 更新硬同步 slow target，不用 Actor loss 更新 Critic 或世界模型。
- Adam：world3e-4 / actor1e-4 / critic1e-4；各模块梯度范数裁剪100。
- 世界模型目标维持 10×vector NLL + reward NLL + discount BCE + KL(balance=.8, free0)。

以上是完整链路的可运行验证配置，不是官方 DreamerV2 全超参复现。
与历史 DreamerV2 的网络容量、batch、回放、BC、Actor 输入不同，不能据成绩做公平架构比较。

## 数据与时间语义

行0为 reset 观测与零动作。行t的动作生成行t观测、奖励和discount；初始行不参加奖励/
终止训练，padding 不参加任何损失。CTM 的 pre/post traces 随 episode 重置并完整保存于
潜在状态，训练观察全 episode，不从截断历史强行零初始化。想象起点不采终止行或padding。
环境 episode 上限100步，超时是任务失败，discount=0。

计数分别记录环境步、Actor环境步、世界模型更新、AC更新、预填和预热，不能把更新次数
当成环境步数。限时预算包含初始化、预填、预热、定期评估。到期停止更新，保存并最终评估；
正常收尾允许少量超时。外层 timeout 在预算+180秒发 TERM，再给90秒退出。

## 评估与观测

- 每约5分钟 checkpoint 后评估固定20场景（70000–70019），初始/最后也测相同集合。
- 最后另测50个未用于定期评估的场景（71000–71049）。评估数据不写入训练回放。
- 模型latent用argmax、Actor用mode，减小checkpoint间随机评估噪声；训练为随机采样。
- 报告送达、取货、超时、越界、回报、回合长度；保留每局原始记录。
- 每30秒记录最近训练损失、各模块梯度范数、想象回报与discount，episode结束即时记录。
- 20局与单seed仍不足以证明性能收敛，50局最终测试也不能证明架构优势。

## 故障与 checkpoint

每次联合更新在赋值前检查全部梯度和loss非有限值；更新后检查参数。失败标记状态并退出，
不自动重试，不用可疑权重覆盖前一个完整checkpoint。保存包含世界模型、Actor、Critic、
slow Critic、三个优化器、更新计数、已完成episode回放、NumPy RNG和下一场景种子。

恢复需新输出目录且配置完全一致，拒绝离线/旧契约或shape/dtype错误；完成时用新实例
校验所有变量/优化器及连续三步确定性策略输出。TF随机流不承诺bit-exact；保存时尚未
完成的episode不进入回放，恢复从新episode开始（最多损失一个回合的未入库数据）。

源文件在启动前复制到独立目录，子进程只执行冻结版本；manifest记录散列。
无网络服务、额外账户、上传或push。只读取本地可信pickle，不能加载外来checkpoint。
回滚为停止该限时进程并保留输出；旧baseline和离线验证入口未改变。

## 验证记录

8项新增单元测试通过（CPU）：Actor潜在输入一致性、非终止想象起点/动作对齐、lambda return
时序、连续Actor梯度穿过CTM、三模块真实更新及目标网络调度、完整checkpoint恢复及错误契约
拒绝、在线episode动作/终止/padding对齐、固定场景评估只读/可复现/可JSON序列化。
连同NRSM核心14项、离线验证2项，共24项通过；另原版CTM前向/梯度4种fixture的1项测试通过。

GPU短测 `outputs/nrsm_online_ac_smoke_v2_20260916` 完成260环境步（100随机预填+160 Actor步），
世界模型9次、AC8次更新，三模块梯度有限，最终完整checkpoint恢复通过。短测只有2个评估场景，
不用于性能判断。首次尝试发现评估场景编号的NumPy int64不可JSON序列化，已显式转换并加回归测试；
失败目录保留，未覆盖。

GPU续训 `outputs/nrsm_online_ac_resume_smoke_20260916` 从短测完整checkpoint继续，达到280环境步，
优化器迭代由9/8/8变为10/9/9，再次通过恢复验证。不是只加载权重后重置优化器。

原 `nrsm.py`、`ctm_rl_core.py`、`models.py`、`tools.py`、`envs.py` 的SHA256与上一轮
一小时离线验证manifest逐项相同。本次不修改核心模块或场景。

## 已启动的一小时完整链路运行

- 输出：`outputs/nrsm_online_ac_1h_20260916_060621/`。
- 启动前冻结源码：相邻 `nrsm_online_ac_1h_20260916_060621_source/`。
- 程序开始：2026-09-16 06:06:39（Asia/Shanghai）；停止更新目标07:06:39，随后保存与最终评估。
- 随机预填已完成12个回合、1030步（完整回合导致超过1000步）；随后预热50次再联合AC训练。
- 每5分钟保存并测试固定20场景，最后另测50场景；结果写入 `result.json`。
- 无自动续跑、无push；这一小时运行尚不表示已经收敛。
