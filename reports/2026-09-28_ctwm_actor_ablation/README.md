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

## Results: experiment 1, actor fix (`imagination_scale = 1.0`)

Both runs completed exactly 137,680 environment steps (27,330 / 27,336 actor-critic updates, ~82 min each while
sharing the CPU), and checkpoint restore was verified.

**Verdict against the pre-registered criteria: H1 is not met, so H0 holds.** The actor fix alone does not make
the actor learn.

Policy (training) episodes, by quarter. The baseline is the first report's run.

| Run | Quarter | MOVE share | Mean MOVE param | Mean speed | Final distance to active goal | Pickup | Delivery | Out of bounds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, seed 17 | Q4 | 0.50 | +0.01 | 6.6 m/step | 1,024 m | 2.1% | 0% | 24% |
| Actor fix, seed 17 | Q1 → Q4 | 0.49 → 0.51 | +0.02 | 7.1 → 7.0 m/step | 1,046 → **1,054 m** | 1.3 → **2.3%** | 0–0.3% | 26 → 33% |
| Actor fix, seed 18 | Q1 → Q4 | 0.50 → 0.50 | +0.03 → +0.02 | 7.2 → 6.7 m/step | 1,028 → **1,044 m** | 1.0 → **1.2%** | 0% | 29 → 39% |

The thresholds were < 820 m final distance or ≥ 10% pickups. Neither run comes close; both stay at the random
prefill's level (~1,150 m, 0–2% pickups).

**The change did take effect.**

- The actor's gradient-norm median rose from 0.012 (baseline) to 0.034 / 0.037.
- But the extra push produced no progress toward the goal. It came with *more* out-of-bounds endings (24% → 33% /
  39% in the last quarter).
- Deterministic evaluation: 0% delivery at every one of the 18 fixed-scene checkpoints in both seeds. Pickups peaked
  at 10% / 20% at single checkpoints, and the final fixed and fresh scenes show 0% pickup. Over 3,237 training
  episodes there was one delivery (seed 17).

**Why: the signal it amplifies is not informative** (`reward_signal_check.py`, final checkpoints):

| Run | Own flights: RMSE / actual std | Own flights: correlation | Controller flights: correlation |
|---|---:|---:|---:|
| Baseline, seed 17 | 8.1× | +0.07 | −0.23 |
| Actor fix, seed 17 | 8.1× | +0.25 | −0.34 |
| Actor fix, seed 18 | 37.6× | +0.06 | −0.19 |

On ordinary steps, the world model still cannot tell better moves from worse ones. On goal-directed flights its
predictions point the wrong way.

- Seed 18, with the most −1 out-of-bounds endings in its replay, has the largest errors. That fits the
  "one fixed-width head is pulled by rare ±1 rewards" diagnosis.
- Its state predictions are unaffected by the actor change: the 15-step open-loop RMSE at the end is 0.441 / 0.403,
  against 0.444 for the baseline and 0.448 for persistence.

**Conclusion.** Down-weighting of the RL term is not the only reason the actor fails. Weighting it up does not
help, because imagined rewards carry almost no usable signal. This makes experiment 2, which targets the reward
head, the critical test.

## Results: experiment 2, actor fix + learned reward std

Both runs completed exactly 137,680 steps (~85 min each), and checkpoint restore was verified.

**Verdict against the pre-registered criteria: all three fail.**

| Run | Own flights: correlation (≥ 0.3?) | RMSE / std (≤ 3?) | Final distance, Q4 (< 820 m?) | Pickups, Q4 (≥ 10%?) | 15-step open-loop RMSE (guardrail: ≤ 1.1× actor fix) |
|---|---:|---:|---:|---:|---:|
| Actor fix, seed 17 | +0.25 | 8.1× | 1,054 m | 2.3% | 0.441 |
| **+ learned reward std, seed 17** | **+0.08** ✗ | **9.9×** ✗ | **1,012 m** ✗ | **1.8%** ✗ | **0.563** ✗ (+28%) |
| Actor fix, seed 18 | +0.06 | 37.6× | 1,044 m | 1.2% | 0.403 |
| **+ learned reward std, seed 18** | **−0.02** ✗ | **9.5×** ✗ | **1,019 m** ✗ | **1.8%** ✗ | **0.662** ✗ (+64%) |

Deterministic evaluation: 0% delivery at every checkpoint of both seeds. Final fixed / fresh pickups were 0% / 2%
(seed 17) and 5% / 0% (seed 18).

**What happened** (training logs):

- The head became confident: the reward NLL fell from +0.75 to −3.3, which means its predicted std sat at the 0.01
  floor. Its errors on ordinary steps did not shrink (RMSE ≈ 0.009–0.015).
- A confident head with unchanged errors produces large gradients. The world-model gradient norm had a median of
  **174 after the first 1,000 updates, against 23 in the baseline**. Almost every update was therefore clipped
  to 100.
- Under global-norm clipping, the reward term then dominates each update. The state predictions degrade (1-step
  open-loop RMSE 0.43 / 0.65, against 0.23 / 0.20 for the actor fix), and the policy still does not improve.

**Root cause: the world model cannot see the effect of a single step.** A per-dimension check on the 8 held-out
episodes:

| Final checkpoint | 1-step prior position error (median) | Active-goal offset error (median) |
|---|---:|---:|
| Baseline, seed 17 | 221 m | 206 m |
| Actor fix, seed 18 | 211 m | 213 m |
| Actual movement per step in those flights | 22 m (max 40 m) | n/a |

- The per-step shaping reward depends on how much one step (≤ 40 m) changes the remaining route. The model's
  position estimate is uncertain by ~10× that, so no reward head, whether fixed or learned-width, can recover the
  signal from these features.
- The team's DreamerV2 diagnosis shows the same pattern: 1-step position error of 273–294 m, against 15–17 m for
  persistence (`STAGE1_RESULTS_20260910.md` §4). This looks like a shared world-model bottleneck, not something
  specific to CTM.

## Figures

One panel per seed. Colour follows the condition: blue = baseline, orange = actor fix, aqua = actor fix + learned
reward std. Seed 18 has no baseline run.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="final_distance_to_goal_dark.png">
  <img alt="Two-panel line chart (seeds 17 and 18) of the rolling mean distance to the active goal at the end of training episodes. All five runs stay between about 950 and 1,200 m for the whole 137,680 steps; none trends down toward the 820 m threshold." src="final_distance_to_goal_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="training_pickup_rate_dark.png">
  <img alt="Two-panel line chart of the rolling pickup rate in training episodes. Every condition fluctuates between 0% and 7% with no upward trend; the pre-registered threshold was 10%." src="training_pickup_rate_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="world_model_15step_dark.png">
  <img alt="Two-panel log-scale line chart of the world model's 15-step open-loop error divided by the persistence error over world-model updates. The baseline and actor-fix runs fall from about x2.2 to about x1.0; the learned-reward-std runs plateau around x1.3 to x1.5." src="world_model_15step_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="actor_gradient_norm_dark.png">
  <img alt="Two-panel log-scale line chart of the actor gradient norm over actor-critic updates. The baseline's median is about 0.012; the actor-fix runs sit about 3 times higher (median about 0.035) and the reward-fix runs about 1.6 times higher (median about 0.02)." src="actor_gradient_norm_light.png">
</picture>

## Overall conclusions

1. **Neither fix makes the CT-WM actor learn within 137,680 steps.** All five runs (baseline, 2 × actor fix,
   2 × actor fix + reward fix) end at 0% delivery, with behaviour indistinguishable from random.
2. **The imagination-scale hypothesis is rejected as the sole cause.** Up-weighting the RL term increases actor
   gradients ~3× but amplifies an uninformative signal (with slightly more out-of-bounds endings).
3. **The learned reward-std head, in this form, is rejected and harmful.** It does not improve reward prediction,
   and it degrades the world model through gradient clipping. If it is retried, the reward term needs its own
   weighting or a separate clip, or a larger `min_std`.
4. **The bottleneck is the world model's precision at single-step scale.** Its position error (~200 m) is about 10×
   one step of motion. Until it can resolve single steps, imagined per-step rewards cannot guide the actor.

**Suggested next experiments** (not run; one variable each):

- Predict state changes (Δ-vectors, i.e. residual decoding) instead of absolute states, or up-weight the position
  and goal-offset dimensions. Measure the position error in metres against the 22 m/step scale.
- Give the actor the exact observation in the real environment, as the team's DreamerV2 does. Then imagination only
  has to rank actions, not localise the drone.
- Derive shaping in imagination from a predicted potential Φ(s) (remaining route length), rather than predicting each
  per-step reward difference.
- Any positive result needs ≥ 3 seeds per condition. Seed 18 in these experiments has no baseline control.

## Files in this folder

| File | Content |
|---|---|
| `run_ablation.py`, `ablation_patches.py` | Wrapper and runtime patches (the team's code is not modified) |
| `*_ablation.json` | Per-run record of the changes and the wrapper/patch SHA-256 |
| `imagscale1_seed{17,18}_*`, `imagscale1_rewardstd_seed{17,18}_*` | Per-run evaluations, training log, episodes, action stats, world-model sweep, reward check, `result.json`, `manifest.json` |
| `baseline_reward_signal.json` | Reward check on the first report's final checkpoint |
| `sweep_world_model.py`, `reward_signal_check.py`, `make_figures.py` | Analysis scripts (reproduce the files and figures) |
| `*_light.png`, `*_dark.png` | Comparison figures |

The git history shows each script's revision: `82ec9cd` is the wrapper the actor-fix runs used, and `4f09dfd` is
the one the reward-fix runs used.
