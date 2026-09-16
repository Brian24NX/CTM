# NRSM 官方对齐版：一小时运行

用户明确要求先运行一小时。本轮只运行 NRSM 世界模型，不接 Actor/Critic、
不改变场景、动作、模型结构或训练目标；不是端到端配送策略训练。

- 新目录：`outputs/nrsm_ctm_1h_20260916_0403`，从零初始化，seed=17。
- 使用当前官方 CTM-RL 对齐的 compact profile（D64/M8/K2，sync_neurons16）。
- 固定离线数据：16 个训练 episode，8 个独立留出 episode；控制器/随机轨迹各半。
- batch=2，完整 padded 101 行 episode，Adam 3e-4，gradient clip=100。
- 启动时间：2026-09-16 04:03:10（Asia/Shanghai）。
- 更新停止目标：2026-09-16 05:03:10；到时完成当前更新后保存、评估、恢复校验。
  一小时预算包含脚本内初始化/评估时间；最后收尾可略超时。
- 每 30 秒记录最新指标、窗口梯度峰值、被裁剪的更新比例。
- 每约 5 分钟保存 model+optimizer checkpoint，再进行留出评估；结束再保存。
- 发现非有限 loss、梯度范数、梯度或更新后参数则停止，不自动重试。
- 独立进程脱离终端运行，外层 timeout 在 3720 秒发送 TERM，另给 90 秒清理后
  强制停止；只约束本次进程组，不操作其他训练或 WSL 服务。

本轮仅修改验证程序的限时、状态、日志与数值检查；官方 CTM 核心未修改。
新增 `start_nrsm_timed.py` 作为后台启动器。既有无 duration 模式仍限定 1..200 更新。
先完成两项离线数据/梯度回归测试，并实测了 15 秒限时流程：2 次更新、到时停止、
checkpoint 恢复校验成功。此前观察到的大梯度风险仍未宣称解决。

## 查询结果

- `status.json`：状态、更新数、deadline、最近 checkpoint；运行中约 30 秒刷新。
- `progress.json`：初始化基准、训练记录、5 分钟评估和最终评估。
- `result.json`：正常收尾后的结果与 checkpoint 恢复校验标志。
- `manifest.json`、`source/`、`data.npz`：固定配置、冻结源码和确切轨迹。
- 目录旁的 `.launch.json` / `.launch.log`：后台 supervisor PID、命令、标准输出。

异常退出时先查启动日志和最后完整 checkpoint；不要把旧 `status=running`
单独当作进程仍存活的证据。Checkpoint 不包含完整 TF RNG 状态，不能保证
bit-exact 训练续跑。没有设置自动追加训练、重启、推送 Git 或修改 baseline。
