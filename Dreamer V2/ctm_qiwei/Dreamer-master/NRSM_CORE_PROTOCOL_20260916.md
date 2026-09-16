# NRSM P0–P2 contract and implementation

> Historical v1 protocol. The active implementation was superseded by the
> official-CTM-aligned v2. See `NRSM_CTM_ALIGNMENT_20260916.md`.
> The v1 source is retained as `nrsm_prototype_v1.py`; do not interpret the
> zero-reset, per-step sync or posterior-fusion rules below as v2 behavior.

Status: implemented core; bounded offline validation only. No Actor integration,
no long training run, no task changes. See the companion validation report for
actual measured results; this document defines the contract, not a success claim.

## Scope and architecture decision

NRSM means **Neuronal Recurrent Space Model**, the core of CT-WM in the new
manuscript `Docs/CT_WM__A_Continuous_Thought_World_Model_for_Long_Distance_Sparse_Rewards_Autonomous_UAV_Navigation.pdf`.
Its placeholder experimental/task claims are not implementation requirements.

Keep the current relay environment and its two-action automatic-pickup contract.
`nrsm.py` depends on existing `models.py`, `tools.py`, and `run_support.py`;
`validate_nrsm.py` adds an offline encoder and three prediction heads.
Neither imports the Dreamer agent, mutates the environment, or loads old weights.
Existing Dreamer files and checkpoints remain untouched by this change.

Alternatives: directly replacing RSSM in the legacy agent is less code initially
but mixes core errors with Actor/input/objective issues. An independent core
costs one adapter later but gives an isolated, testable prior/posterior boundary.
Choose the independent core. Do not introduce another service or training stack.
Rollback is simply not invoking the new runner; no legacy migration is required.

## State and timing contract

All tensors and trainable computation are FP32. B is batch, D neurons, M trace
length in **internal ticks**, C context, Z categorical variables, Q classes.

| Carry key | Shape | Meaning |
|---|---|---|
| deter | B,C | Posterior-fused context after observation; prior context during imagination |
| stoch | B,Z,Q | Straight-through categorical latent |
| logits | B,Z,Q | Categorical distribution parameters |
| pre_trace | B,D,M | Persistent per-neuron pre-activation history |
| activation | B,D | Final neuron activations from the previous environment step |

`initial(B)` zeros every key. `reset(state,is_first)` clears every key only for
the indicated rows. `observe(embed, action, state, is_first, valid)` accepts B,T
sequences; padded rows freeze every state key and receive zero input gradient.
Actions at row t produced observation t; row zero is an initial observation.
Full episodes have at most 101 rows. Chunked callers must pass the complete carry;
passing only deter/stoch is not valid. No mutable per-episode state lives in modules.

Transition order:

1. Condition on previous posterior context, previous latent, and canonical action.
2. Perform K ticks: synapse network → append pre-activation → private per-neuron
   MLP → selected-pair synchronization → projection and LayerNorm.
3. Compute the prior distribution from the final projected context, **without
   the current observation**.
4. Only then read the current embedding, compute posterior latent, and fuse the
   context with embedding and posterior latent.
5. Imagination repeats steps 1–3 only, carrying the full neuronal state.

Private MLP weights have dimensions D,M,H and D,H: not a shared temporal MLP.
Fixed, seeded, unique unordered neuron pairs include possible self-pairs.
Per-pair rates are softplus parameters. Synchronization implements manuscript
Eq. 9: `N <- exp(-r)*N + activation_i*activation_j`,
`mass <- exp(-r)*mass + 1`, `S = N/sqrt(mass)`.
N and mass reset at **each environment step**, not across episodes only.
S is not Pearson correlation and is not guaranteed to lie in [-1,1].

## Explicit interpretations and deviations

- Persistent pre-traces/activations across environment steps are an explicit
  engineering extension where the manuscript's history initialization is
  underspecified. `history_mode=context_only` is the reinitialization ablation.
- Posterior context fusion follows the intended manuscript mechanism but creates
  a possible context distribution gap: heads trained on posterior features need
  not work well on prior features. Measure both, not just reconstruction loss.
- Omit optional null-token attention in both profiles. K=1 is a tick ablation,
  **not** a claim of equivalence to RSSM/GRU.
- Tick weighting loss is OFF. For `L=sum softmax(-lambda*l)_i*l_i`, full gradient
  is `w_j*(1-lambda*(l_j-L))`, not simply `w_j`; high-loss ticks can have negative
  gradients. Detached weights are a different objective. Do not silently select
  either interpretation or call the paper's gradient expression verified.
- Current shaped rewards are continuous, so use a Gaussian reward head, not the
  manuscript's sparse Bernoulli reward description. No auxiliary task heads.

## Profiles and validation protocol

| Parameter | Core default | P2 compact diagnostic |
|---|---:|---:|
| D / M / K / selected pairs | 256 / 16 / 4 / 512 | 64 / 8 / 2 / 128 |
| Context / synapse hidden / embedding | 400 / 400 / 400 | 128 / 128 / 128 |
| Private MLP hidden | 16 | 8 |
| Categorical variables / classes | 32 / 32 | 8 / 8 |

P1 tests cover prior leakage, private weights, explicit-sum synchronization,
chunk carry, padding gradients, reset, parameter gradients, action gradients,
compiled dynamic scans, checkpoint roundtrip/rejection, history ablation,
single tick, and a default-profile forward/backward pass.

P2 uses 16 training episodes (seeds 20000–20015) and 8 held-out episodes
(40000–40007), alternating the existing feedback controller and random actions.
These are diagnostic model-training trajectories, not behavioral cloning or an
end-to-end policy evaluation. Episodes are padded to 101 and masked; no context
is silently discarded by short random crops. Batch 2, 100 gradient updates,
Adam 3e-4, global gradient clip 100; runner refuses more than 200 updates.

Loss: 10×vector Gaussian NLL + reward Gaussian NLL + discount BCE + KL,
KL(post||prior), balance .8, free 0. Initial row has no reward/discount loss.
This is a declared diagnostic objective, **not audited official DreamerV2
hyperparameters**. No tick loss, Actor, Critic, BC, or rare-event sampler.

Fixed held-out evaluation uses categorical argmax (not an MC expectation):
posterior/prior normalized-vector RMSE, posterior/prior reward RMSE, posterior
discount BCE, and open-loop vector RMSE at 1/5/15 steps from every valid posterior
anchor using actual future actions. Windows never cross an episode boundary.
Include last-observation persistence error as a simple reference. Reduction in
reconstruction loss alone is not evidence of good dynamics or policy usefulness.

Each new output directory refuses overwrite and contains frozen source files
and SHA256 manifest, exact collected data, split metadata, atomic progress JSON,
and retained model+optimizer snapshots at 0/50/100. Disk guards precede snapshots.
Core checkpoint loading rejects architecture/dtype/nonfinite/pair-index mismatch
before assigning weights. Pickle files are **trusted-local only**.
The diagnostic verifies full model and optimizer restoration plus deterministic
prediction equality, but does not promise bit-exact training resume: TF random
state is not captured and there is no resumable trainer CLI yet.

## Remaining work and gates

P0/P1 success means a specified, tested core; P2 success means a finite bounded
training pipeline with honest held-out diagnostics. It does not establish
long-memory benefit, stable latent planning, or superiority over DreamerV2.
Before P3, inspect prior/posterior and multistep gaps. Any objective change needs
its own comparison rather than being silently folded into this run.

P3 later: latent-only Actor/Critic adapter, posterior-vs-imagined input audit,
full restart/RNG/data-sampler checkpoint contract, memory/cost measurements,
and bounded integration tests. Formal comparison requires an independently
audited DreamerV2 baseline and matching observation/action/reward/data budgets.
Current fully exposed vector scenario can test engineering but cannot by itself
establish a long-term-memory advantage.

Research references informing the design (not claims of identical code):

- [Sakana CTM RL core](https://github.com/SakanaAI/continuous-thought-machines/blob/main/models/ctm_rl.py)
- [Sakana neuron-level modules](https://github.com/SakanaAI/continuous-thought-machines/blob/main/models/modules.py)
- [Official DreamerV2 agent](https://github.com/danijar/dreamerv2/blob/main/dreamerv2/agent.py)
