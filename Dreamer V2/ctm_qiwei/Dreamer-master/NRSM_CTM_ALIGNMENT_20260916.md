# NRSM v2：官方 CTM 模块对齐

## 决定与范围

按用户要求，取消 v1 对 CTM 核心机制的简化改写，改为固定版本官方 RL 向量核心的
TensorFlow 移植。**保留真实官方 PyTorch 模块作为独立可运行参照，不再以手写公式
测试替代官方模块对照。** 不是在原 RSSM 后叠加 CTM，也没有迁移旧 Dreamer 训练框架。

固定源码：SakanaAI `continuous-thought-machines`，commit
`4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`。
官方文件未经修改保存在 `third_party/ctm_reference/models/`，附 Apache-2.0 LICENSE。
模型侧生产运行仍为 TensorFlow，PyTorch 只在独立 CPU 参照环境中使用。

本次使用 deep-research 做源码/运行时核验，使用 project-architecture 划分参照、移植、
世界模型适配三个边界。采用最小独立模块；没有整体重写旧 Dreamer 或改动场景。

## 已替换的模块

| 部分 | v1 改写原型 | 当前 v2 |
|---|---|---|
| 输入 backbone | 一层 condition ELU | 官方 ClassicControlBackbone 的两组 Linear/GLU/LayerNorm |
| 神经元 NLM | 私有 ELU/tanh MLP | SuperLinear/GLU，保留私有权重、可选历史 LN、学习温度除数 T |
| Synapse | 简化 ELU MLP | 官方 RL 的两组 Linear/GLU/LN；另支持 SynapseUNET |
| 初始神经元历史 | 全零 | 可学习的 start_trace 与 start_activated_trace |
| 持久状态 | pre_trace + 最后一个 activation | 完整 pre_trace 与 post_trace |
| 同步窗口 | 每个环境步从零累计 K ticks | 对持久 post_trace 滑动窗口求同步，跨环境步保留 |
| 同步神经元 | 任意随机选对 | RL 支持的 first-last：最后 n 个神经元的上三角，包含自配对 |
| 衰减参数 | softplus(raw_rate) | 官方 exp(-age * clamp(decay,0,4))，初值 0 |
| posterior context | 当前观测融合后重新生成 | 与 prior 共用 CTM 确定性 context；观测只更新 posterior latent |

依据的是 **ctm_rl.py 实际执行路径**，不是混用基础 CTM 和 RL 变体的不同规则。
特别是 RL 同步实际选择最后 n 个神经元，虽然继承的部分 out-index buffer 描述的是
前 n 个；移植遵循实际运算，未擅自修正官方行为。

## 保留的世界模型适配层

`nrsm.py` 把 `(previous_context, previous_z, canonical_action)` 作为向量输入传给
`ctm_rl_core.py`。同步向量经适配投影形成 context；prior 只依赖历史与动作，posterior
额外接收当前 embedding。编码器、categorical latent、KL 与预测头不属于官方 CTM。
这些仍是 NRSM 的适配设计，不能声称被官方 CTM 等价测试验证了算法效果。

额外 posterior fusion 已从默认实现删除；v1 保留在 `nrsm_prototype_v1.py` 和旧运行
冻结源码中。一次性完成这些替换不构成单变量消融，因此不能把将来的成绩变化归因于
其中某一个模块。当前仅验证移植正确性。

## 配置与 carry/checkpoint 契约

默认 D=256、M=16、K=4、sync_neurons=32，对应 **528** 个同步项；compact profile
使用 D=64、M=8、K=2、sync_neurons=16，对应 **136** 项。
旧配置 `pairs`、`pair_seed`、`history_mode=context_only` 不再适用。
deep_nlms=True、nlm_norm=True、synapse_depth=1 是明确选择的有效配置，
不是声称复刻某个论文实验的完整超参数。

carry 为 deter、stoch、logits、pre_trace、post_trace；后两者均为 B,D,M。
reset 恢复学习到的初始神经元 traces，而非全零；其他状态置零。
Padding 冻结完整 carry；跨切片续算必须传递所有 key。

新版核心 checkpoint contract 为 `nrsm_v2_ctm_rl_4a6c9c3_shared_context_fp32`，
同时记录完整 upstream revision。v1 或配置/形状/类型不匹配的 checkpoint 在赋值前
拒绝。旧结果、数据及 checkpoint 未删除；没有自动把旧权重转换成新结构。
核心 snapshot 仍不是完整的 bit-exact 训练恢复机制，TF RNG 恢复尚不在本轮范围。

## 真正的官方对照

`tests/export_ctm_reference.py` 在隔离子进程中导入原版
`ContinuousThoughtMachineRL`，导出实际权重、前向值和 autograd 梯度。
`tests/test_ctm_parity.py` 将相同权重映射到 TF 模块，比较：

- backbone 输出；每个 tick 的 pre/post activation 与同步值；
- 3 个连续环境步的完整历史与最终输出，每步 3 ticks、M=4，因此窗口跨越步边界；
- 输入梯度和全部使用的参数梯度，包括初始历史、NLM 温度、同步衰减；
- decay 区间内外及 0/4 端点，LN 开/关、深/浅 NLM、SynapseUNET；
- 官方源码 SHA256，防止参照被意外修改。

比较结果（CPU FP32；默认 oneDNN 设置）：

| 官方配置 | 对照参数数 | 最大前向绝对误差 | 最大梯度绝对误差 |
|---|---:|---:|---:|
| deep_no_norm | 25 | 8.94e-7 | 2.41e-7 |
| deep_norm | 29 | 1.37e-6 | 1.91e-5 |
| shallow_norm | 24 | 1.79e-6 | 6.42e-6 |
| unet_depth2 | 35 | 8.34e-6 | 7.34e-5 |

前向逐元素 `atol=3e-6, rtol=3e-5`；梯度 `atol=5e-5, rtol=5e-4`。
阈值包含绝对与相对两部分，不要求最大绝对误差低于 atol。不是逐位一致。
初次采用更紧的梯度阈值时，深层链路在接近零的梯度分量上出现数万分之一的差异。
已核对运算顺序，并将 LN 改为显式中心化的标准公式；最终为 FP32 跨框架累积误差
设置上述容差。clamp 区间外严格检查零梯度，端点额外检查非零及数值，
不能借普通容差掩盖梯度路由不同。

**71/71 联合测试通过**：原版 parity + NRSM + 离线数据/梯度 + 既有 KL、自动取货、
工程集成回归。初次 GPU 单测遇到显存预分配失败；正式联合测试在 CPU 完成，
GPU 冒烟另启用按需显存分配，不将 CPU 结果冒称 GPU 全套等价证明。

### GPU 冒烟结果与尚未通过的稳定性门槛

`outputs/nrsm_ctm_aligned_smoke_20260916/result.json`：compact 模型 356,409 参数，
4 个采集训练 episode、2 个留出 episode，完整序列 batch=2，仅 **2 次** optimizer
更新；约 74.6 秒。前后向有限、model/optimizer snapshot 恢复与确定性预测校验通过，
进程已正常结束，没有接 Actor 或继续训练。结果不构成学习质量或优越性的证据。

第 2 次更新的**裁剪前梯度范数为 2,465,413.5**，使用既有 clip=100。
因此数值移植通过不等于训练稳定性通过：原版模块接入全长世界模型 BPTT 后仍可能
出现严重梯度放大。后续必须单独检查序列长度、梯度传播与优化配置；本轮没有为了
掩盖该现象而再次改造 CTM 核心模块。旧 baseline 文件经冻结源码哈希比对未改变。

## 固定参照运行时的重要性

官方 requirements 没有锁定 torch。首次自动安装的 2.14.0+cpu 对
`clamp([-1,0,.1,4,5],0,4)` 的梯度实测为 `[0,0,1,0,0]`；
明确选择并固定的 **2.7.0+cpu** 为 `[0,1,1,1,0]`，与当前 TF clip 边界语义一致。
因此四组 fixture 全部用 2.7.0 重新生成，导出器拒绝不同 torch 版本。
这是选定的可复现参照运行时，**并非已证实的论文原始训练环境**；不宣称与任意
PyTorch 版本都等价。参照 requirements 和 NPZ 版本元数据保留在项目中。

## 复核与后续

```bash
# Dreamer venv：直接读取已保存的官方 fixture，不需要安装 PyTorch
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=tests /home/madao/.venvs/ctm-dreamer/bin/python \
  -m unittest test_ctm_parity test_nrsm test_nrsm_validation test_kl_balance test_auto_pickup test_engineering -v

# 参照 venv：重建 fixture，默认拒绝覆盖
/home/madao/.venvs/ctm-reference/bin/python tests/export_ctm_reference.py \
  --output outputs/ctm_reference/new_deep_norm.npz
```

当前等价范围：RL 向量输入、FP32、dropout=0、first-last、测试覆盖的模块配置。
不包括图像注意力版 CTM、非零 dropout、任意规模/任意运行时的数值一致性。
未接 Actor/Critic、未做正式长训练、未声称收敛或配送性能改善。
下一步仍应先做预算匹配的短世界模型实验，查看 prior/多步误差及梯度稳定性。

来源：

- [官方 RL 核心](https://github.com/SakanaAI/continuous-thought-machines/blob/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466/models/ctm_rl.py)
- [官方 CTM/NLM 组装](https://github.com/SakanaAI/continuous-thought-machines/blob/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466/models/ctm.py)
- [官方基础模块](https://github.com/SakanaAI/continuous-thought-machines/blob/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466/models/modules.py)
