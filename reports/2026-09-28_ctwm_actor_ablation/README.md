# CT-WM actor ablation: `imagination_scale` 0.1 → 1.0 (2026-09-28)

Follow-up to [`../2026-09-28_first_run_mac/`](../2026-09-28_first_run_mac/README.md). In that 1-hour run the
CT-WM world model learned, but the actor stayed a random policy for all 1,540 policy episodes.

## Hypothesis and pre-registered criteria

_Written at 02:21 CDT, after launch and before any result beyond the step-0 evaluation was available._

**Hypothesis.** The CT-WM actor loss is DreamerV2's actor loss without its 5× behaviour-cloning term:

```
loss = −imagination_scale · (REINFORCE + 0.1 · dynamics) − (0.03 · H_discrete + 0.02 · H_parameter)
```

- With `imagination_scale = 0.1`, the RL term is 10× down-weighted relative to the entropy bonus.
- Adam's update direction depends on the relative weights, not on the overall loss scale. So the entropy bonus
  (a push toward randomness) may dominate, and the actor never leaves the random policy.
- Setting `imagination_scale = 1.0` restores the RL term's weight. **This is the only change.**

**Setup.**

- The unmodified `train_nrsm_online.py`, with every default kept (compact NRSM, no demonstrations, latent-input
  actor, batch 2, H = 15, train every 5 steps, and so on).
- `run_ablation.py` (in this folder) only replaces the default `ACConfig`. The trainer's own `manifest.json`
  confirms `imagination_scale = 1.0` with all other actor-critic settings at their defaults.
- **Budget:** exactly 137,680 environment steps (`--steps`), which is what the baseline run reached. Learning
  happens per environment step, so wall-clock speed does not affect it.
- **Seed 17** is paired with the baseline. It has the same initial weights (its step-0 evaluation is identical:
  0% / 5% / 70% / 30%), the same random prefill and the same training-scene sequence.
- **Seed 18** is a replicate of the fix. There is **no seed-18 baseline**, which is a known gap.
- Evaluation is unchanged: 20 fixed scenes (70000–70019) every ~5 min, then 20 fixed + 50 fresh scenes
  (71000–71049) at the end.

**Criteria**, judged on the last quarter of policy episodes against the baseline's last quarter (1,024 m final
distance, 2.1% pickups, MOVE share 0.50):

- **The actor learns (H1)** if, in the fix runs, the mean distance to the active goal at the end of an episode is
  at least 20% lower (< 820 m) **or** the training pickup rate is at least 10%.
- **No effect (H0)** if the behaviour statistics stay at random-prefill levels: MOVE share ≈ 0.5, |parameter| ≈ 0.5,
  final distance ≈ 1,000 m, pickups ≈ 1–2%.
- **Secondary:** delivery and pickup in the deterministic evaluations, and world-model prediction error (the same
  sweep as the first report).

## Results

_Added when the runs finish._
