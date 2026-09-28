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

## Second experiment: learned-variance reward head (pre-registered)

_Written at 02:37 CDT, while the actor-fix runs were at ~23k of 137,680 steps. Only their 5-minute evaluations
(all 0% delivery) had been seen._

**Motivation (measured on the baseline's final checkpoint, `baseline_reward_signal.json`).** On ordinary
(non-terminal) steps, the goal_safe_v1 reward is −0.01 plus potential shaping. Its informative variation is tiny:
std 0.0015 on the agent's own flights. The world model's reward predictions are much worse than that:

| Data | Actual std | Prediction RMSE | Correlation with actual |
|---|---:|---:|---:|
| Agent's last 100 training episodes | 0.0015 | 0.012 (8× the std) | +0.05 to +0.07 |
| Held-out random flights | 0.0017 | 0.008–0.009 | +0.33 |
| Held-out controller flights | 0.0028 | 0.016 | −0.21 to −0.23 |

So imagination gives the actor almost no information about which steps are better. The reward head is a
**fixed unit-variance Gaussian** (`models.DenseHead`, `Normal(x, 1)`), and it has to fit both these ~0.001-scale
differences and the rare ±1 terminal rewards.

**Hypothesis.** A reward head that predicts its own per-state standard deviation can be precise on ordinary steps
and uncertain on terminal steps. That should raise the informativeness of imagined rewards and give the actor a
usable signal.

**Change.** `ablation_patches.LearnedStdHead`, with `std = 0.01 + softplus(raw)`, is the only change. It is
applied *on top of the actor fix* (`imagination_scale = 1.0`) and compared seed-for-seed with the actor-fix runs.

- Everything else is identical: the task and reward (goal_safe_v1, shaping 1.0), the loss (still the reward NLL),
  budget 137,680 steps, seeds 17 and 18.
- The team's manifest does not record this change. `<run>.ablation.json` does.

**Criteria**, compared with the actor-fix run of the same seed:

- **Mechanism (primary for this change):** on the agent's own last 100 episodes, the correlation between predicted
  and actual non-terminal rewards is **≥ 0.3**, and RMSE / std is **≤ 3** (the fixed head gives 0.05–0.07 and ~8).
- **Behaviour:** the same thresholds as the first experiment, on the last quarter of policy episodes: final
  distance < 820 m **or** pickups ≥ 10%. Also better than the actor-fix run of the same seed on both measures.
- **Guardrail:** the world model's 15-step open-loop vector RMSE is not more than 10% worse than the actor-fix run's.

## Results

_Added when the runs finish._
