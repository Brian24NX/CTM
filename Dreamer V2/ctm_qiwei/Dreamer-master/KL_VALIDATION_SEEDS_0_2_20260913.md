# KL 修复验证：seed 0、seed 2

用户要求 seed 0、seed 2 重跑。本轮与修复版 seed 1 使用同一训练协议：从原第一阶段约 10 万步 checkpoint 续到累计 30 万步，不从零重训。

- 新目录：`/home/madao/ctm-runs/dreamerv2_klfix_seed0_seed2_20260913`。
- 顺序：seed 0 从 100,050 步至 300,000 步，然后 seed 2 从 100,004 步至 300,000 步；完整 episode 可能略超预算。
- 与已完成的修复版 seed 1 的 15 个冻结源码文件散列完全一致。相对旧实验，训练源码仅 `models.py` 的 KL 直接梯度权重修正为 prior 0.8、posterior 0.2。
- MOVE/TURN + 自动取货保持不变，无显式 CATCH。
- 复制并逐文件核验 replay、64 条永久示范和 checkpoint；保留优化器与已完成的预热状态。旧实验和修复版 seed 1 不变。
- 最新 checkpoint 约每 1,000 步保存并保留上一份，每跨 10,000 步及终点保留快照；另保存各自起点快照。
- 每 5,000 步在线测试 5 局；两个 seed 串行完成后 runner 自动退出。不启动其他种子、额外训练或独立评测；遇失败不自动跳过。
- 首约 10 万步的日志和权重来自旧 KL。进程重启会重建 RNG、环境和采样器，不保证逐位复现；单 seed 在线滑动结果不等于独立固定场景评测。

## 准备与验证

`prepare_kl_validation.py` 新增 `--seeds` 选项，默认仍为 seed 1。新增 5 项测试均通过：指定种子及顺序、父文件保全、默认行为、非法种子、损坏第二个父 checkpoint 和已存在目标拒绝（部分合并在同一测试中）。实际准备核验了两份 checkpoint 的散列、有限性、步数、预热、replay 步数及复制一致性。

```bash
/home/madao/.venvs/ctm-dreamer/bin/python -B prepare_kl_validation.py \
  --previous /home/madao/ctm-runs/dreamerv2_stage1_20260910 \
  --outdir /home/madao/ctm-runs/dreamerv2_klfix_seed0_seed2_20260913 --seeds 0 2
```

主状态为新目录 `batch_status.json`，出处记录为 `continuation_provenance.json`。原自动跟进保持暂停，本轮无定时模型调用。新准备脚本与本说明尚未再次 push。
