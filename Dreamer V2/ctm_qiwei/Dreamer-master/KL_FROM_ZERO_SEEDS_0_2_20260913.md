# Seed 0、seed 2：正确 KL 从零训练

用户明确要求从零开始，取代此前从旧 KL 的 10 万步 checkpoint 续训安排。

- 旧续训目录 `dreamerv2_klfix_seed0_seed2_20260913` 已停止；seed 0 留下 103,193 步的有限 checkpoint，seed 2 尚未启动。旧结果不删除、不覆盖，也不再自动继续。
- 新目录：`/home/madao/ctm-runs/dreamerv2_klfix_fromzero_seed0_seed2_20260913`。
- 顺序：seed 0 → seed 2，各从零训练至累计 300,000 环境步（完整 episode 可能略超）；全部完成后自动停止。
- 不复制或加载任何旧 checkpoint、优化器状态、replay 或历史 metrics。模型随机初始化，重新生成 64 条控制器示范，重新采集 prefill，执行 1,000 次世界模型预热和 3,000 次 Actor 预热。
- 从第一次梯度更新就使用正确 KL：prior 0.8、posterior 0.2。15 个冻结训练源文件与已验证的修复版 seed 1 完全相同，其他超参数不变。
- 动作保持 MOVE/TURN + 自动取货，无显式 CATCH。
- 每约 1,000 步保存最新/上一份 checkpoint，每跨 10,000 步及终点保留快照；预热每 100 次更新保存进度。
- 每 5,000 环境步在线测试 5 局；没有自动独立评测、其他 seed 或额外训练阶段。
- 已完成的 seed 1 仍是“旧 KL 10 万步起点 + 修复续训”，本次不重跑它，后续比较应明确这一差别。

## 启动与状态

使用已冻结的修复版 runner 创建全新批次，不传 `--resume`。runner 的启动位置不意味着加载该目录的 checkpoint：训练输出目录与新批次命令共同决定数据来源。

```bash
/home/madao/.venvs/ctm-dreamer/bin/python -B -u \
  /home/madao/ctm-runs/dreamerv2_klfix_seed1_20260912/source/run_batch.py \
  --outdir /home/madao/ctm-runs/dreamerv2_klfix_fromzero_seed0_seed2_20260913 \
  --seeds 0 2 --steps 300000
```

主状态为新目录 `batch_status.json`；训练初始化标记为各 seed 的 `run_metadata.jsonl` 中 `resumed_checkpoint: false`。旧自动跟进仍暂停；没有新增定时模型调用。
