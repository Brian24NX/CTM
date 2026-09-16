# NRSM 首轮工程验证结果 — 2026-09-16

> 本文是 v1 改写原型的历史实验，不能代表后续官方 CTM 对齐版。
> 当前实现说明见 `NRSM_CTM_ALIGNMENT_20260916.md`；本文的数值与 checkpoint 保留不变。

## 结论

P0 核心契约、P1 独立实现与测试、P2 短程离线验证已落地。
**工程链路通过；prior / imagination 的预测质量尚未过关，不进入 Actor 或长跑。**
当前有证据表明 posterior 重建在学习，但不能把它说成世界模型已经能可靠想象未来。
本次没有修改旧 Dreamer 入口、环境、动作集或旧 checkpoint，也没有 push。

## 本轮新增

- `nrsm.py`：独立 NRSM，私有神经元历史 MLP、K tick、同步统计、离散 prior/posterior、
  跨环境步 carry、逐行 reset、padding mask、FP32、严格核心 checkpoint。
- `validate_nrsm.py`：不接 Actor 的离线世界模型验证，完整 episode 输入，独立留出种子。
- `diagnose_nrsm_context.py`：对保存模型做 context / latent 交叉特征诊断。
- `tests/test_nrsm.py`、`tests/test_nrsm_validation.py`：15 项新增测试。
- `NRSM_CORE_PROTOCOL_20260916.md`：论文解释、实现差异、数据与验收契约。

按照 project-architecture skill 的独立边界和可回退原则新增模块，没有直接替换旧 RSSM。

## 测试与运行范围

联合运行 NRSM、离线验证、既有 KL、自动取货和工程集成测试：**69/69 通过**，
其中新增 15 项，既有回归 54 项。包括默认尺寸模型的单步前向/反向，
动态序列编译、完整状态传递、prior 观测隔离和 checkpoint 错误拒绝。
运行命令：

```bash
PYTHONPATH=tests bash run_wsl.sh -m unittest test_nrsm test_nrsm_validation test_kl_balance test_auto_pickup test_engineering -v
bash run_wsl.sh validate_nrsm.py --output outputs/nrsm_p2_20260916 --updates 100
bash run_wsl.sh diagnose_nrsm_context.py outputs/nrsm_p2_20260916
```

输出目录已经存在，训练命令不会覆盖；重做必须指定新的目录。

短验证：RTX 5070 Ti Laptop / TensorFlow 2.21，compact NRSM 331,151 个训练参数；
16 个训练 episode、8 个留出 episode，控制器与随机轨迹各半；batch=2，100 次更新。
约 131.8 秒（脚本内采集、训练、评估、恢复检查；不含启动检查）。不是 100 个环境步，
也不是 NRSM 策略训练。初始轨迹 collection 的成功标记属于采集控制器，不属于 NRSM。

## 留出集预测结果

固定的 8 个未参与训练的 episode，categorical argmax；向量为现有归一化 13 维观测，
RMSE 不是米。后验重建看到了当前观测；prior 和开环预测没有看到目标时刻的观测。

| 指标，越低越好 | 初始化 | 更新 50 | 更新 100 |
|---|---:|---:|---:|
| Posterior 向量 RMSE | 0.7566 | 0.4225 | **0.2118** |
| 一步 prior 向量 RMSE | 1.0424 | 0.9963 | **0.9929** |
| Posterior reward RMSE | 0.4739 | 0.1425 | 0.1179 |
| Prior reward RMSE | 1.1890 | 1.0281 | 1.0454 |
| Posterior discount BCE | 1.0144 | 0.1138 | 0.1104 |
| 开环 5 步向量 RMSE | 1.0101 | 1.1652 | **1.1078** |
| 开环 15 步向量 RMSE | 1.0473 | 0.9184 | **1.0584** |

相同有效窗口下，直接保持最后观测不变的参考误差为：1 步 0.1079，5 步 0.2637，
15 步 0.4484。当前 learned prior 明显未达到实用水平。这个参考无需学习，但不能
据此单独诊断 NRSM 架构的长期上限；许多观测维度短时间本来就变化不大。

## Context / latent 差距诊断

在更新 100 的同一个模型、相同留出样本上，保持预测头权重不变，交叉使用两种特征：

| 输入预测头的 context | 输入预测头的 categorical latent | 向量 RMSE |
|---|---|---:|
| Posterior | Posterior | 0.2118 |
| Posterior | Prior | 0.2191 |
| Prior | Posterior | 0.9863 |
| Prior | Prior | 0.9929 |

**该 checkpoint 的直接预测误差对 context 来源非常敏感，对 latent 来源相对不敏感。**
留出集 categorical KL=0.3037，而 posterior/prior context 差的 RMS=1.1469。
这些交叉特征不是合法的独立生成轨迹，属于诊断探针，不等于完整训练消融或因果证明。

代码解释：`nrsm.py:155` 的 prior 使用同步投影后的 context；`nrsm.py:160`
的 posterior 又经过包含当前 embedding 的 fusion。`validate_nrsm.py:81` 的重建、
reward、discount 监督来自 posterior 特征；KL 对齐的是 categorical 分布，
**没有直接保证两种 deterministic context 能被同一个预测头等价使用**。
因此 KL 和重建都下降，并不能保证 imagination 的预测头输入已匹配。

这符合此前明确登记的 posterior-fusion 风险。仍不能仅凭 100 次更新断言“训练更久
绝对无效”，也不能把这个 NRSM 新实现的问题归因于旧 DreamerV2 或场景本身。

## 稳定性、checkpoint 与边界

- 梯度和参数检查未发现 NaN/Inf；第 10 次更新记录的裁剪前梯度范数达到 42,598，
  使用全局 clip=100。第 100 次为 42.31。短程有限不代表长跑稳定性已经通过。
- 0/50/100 更新分别保留完整诊断模型和 optimizer snapshot，每份约 4 MB；
  最终新建模型恢复后，确定性预测及 optimizer 张量一致性检查通过。
- 源码冻结及 SHA256、采集数据、split seeds、progress/result JSON 均已保存。
- 这些是可信本地 pickle；不要加载外来不可信 checkpoint。
- 尚无 bit-exact 训练恢复承诺：TF 随机状态未捕获，恢复训练 CLI 尚未实现。
- 默认大尺寸只做了前向/反向测试；短训练使用 compact profile，不能外推成本和收敛。
- 当前场景观测充分，未验证长时记忆优势；未训练 Actor，未衡量 NRSM 配送成功率。

原始证据：`outputs/nrsm_p2_20260916/result.json`、`progress.json`、
`context_probe.json`、`manifest.json`、`episodes.json`。

## 下一步建议（本轮未执行）

先做共享数据、初始化规则和预算的 **posterior-fusion / prediction-feature 对照**：
保留当前实现作为 A，另设一个明确标注的 context 一致性方案作为 B，先检验一步 prior
和多步误差是否改善，并跟踪长序列梯度。不要仅凭更低 KL 选择方案，也不要把改动偷渡
到 DreamerV2 baseline。通过这个模型侧门槛后再接 latent Actor/Critic。
