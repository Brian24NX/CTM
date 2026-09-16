# Pinned official CTM reference

The files in `models/` are unmodified SakanaAI Continuous Thought Machines
source pinned to commit `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`.
The upstream license is retained in `LICENSE`.

Upstream: <https://github.com/SakanaAI/continuous-thought-machines/tree/4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466>

This reference is test-only. Production Dreamer remains TensorFlow and must not
import these PyTorch modules. Reference execution requires a separate CPU
PyTorch environment; do not install PyTorch into the Dreamer environment.

## Explicit reference runtime

These fixtures use **PyTorch 2.7.0+cpu**, NumPy 2.5.3 and huggingface_hub
1.31.0. This is a pinned implementation-comparison runtime; it is **not a
claim that these exact packages reproduce the original paper's runtime**.
Install into the isolated reference environment with:

```bash
/home/madao/.venvs/ctm-reference/bin/python -m pip install \
  -r third_party/ctm_reference/requirements-reference.txt
```

The upstream repository does not pin PyTorch. Newer versions are not necessarily
interchangeable: the locally checked CPU PyTorch 2.14.0 gives zero gradients at
the exact endpoints of `clamp(x, 0, 4)`, whereas CPU PyTorch 2.7.0 gives one.
For inputs `[-1, 0, 0.1, 4, 5]`, the observed gradients of the summed clamp are
`[0, 0, 1, 0, 0]` versus `[0, 1, 1, 1, 0]`, respectively. The official CTM
initializes decay parameters to exactly zero, so this affects a trainable
parameter, not merely an irrelevant numerical corner. TensorFlow's
`clip_by_value` endpoint convention matches the selected 2.7.0 runtime.
The exporter rejects a different PyTorch version rather than silently changing
the reference gradient contract. PyTorch 2.14 fixtures generated during initial
setup were replaced, not mixed into the final reference fixture set.

## Reproduce an independent reference fixture

From the project root in WSL:

```bash
/home/madao/.venvs/ctm-reference/bin/python tests/export_ctm_reference.py \
  --output tests/fixtures/ctm_reference/deep_norm.npz
```

Repeat with `--deep-nlms 0`, `--layernorm 0`, or `--synapse-depth 2` to cover
the shallow NLM, unnormalized NLM, and upstream SynapseUNET variants.
The exporter spawns an isolated interpreter and imports the actual
`ContinuousThoughtMachineRL` class. No handwritten reference model is used.
It records upstream source hashes and the PyTorch version in each NPZ.

## Fixture contract

All calculations are CPU float32, dropout is zero, and the network is in
evaluation mode. Defaults are 3 environment steps, 2 batch rows, 3 ticks per
step, 8 neurons, history length 4, input width 7, backbone width 6, NLM hidden
width 5, and the last 4 neurons' 10 upper-triangular pairs. Thus history crosses
environment-step boundaries. Both initial traces are learned parameters.

- `config__NAME`: upstream constructor settings.
- `weight__NAME`: the complete upstream `state_dict`; names/shapes unchanged.
- `grad__NAME`: gradients of every upstream trainable parameter, including
  learned initial traces. Non-trainable index/mask buffers have no gradient.
- `inputs`, `input_grad`: `[environment_step, batch, input_size]`.
- `result`, `loss_weights`: `[environment_step, batch, pair]`.
- `backbone`: `[environment_step, batch, backbone_width]`.
- `pre_ticks`, `post_ticks`: `[environment_step, tick, batch, neuron]` from
  official `track=True`.
- `sync_ticks`: `[environment_step, tick, batch, pair]`, obtained by calling
  the official synchronisation method on the tracked activation windows.
- `pre_states`, `post_states`: `[environment_step, batch, neuron, history]`.
- `initial_pre`, `initial_post`, `final_pre`, `final_post`: `[batch, neuron, history]`.
- `loss`: `sum(result * loss_weights) + 0.03 * sum(final_pre**2) +
  0.07 * sum(final_post**2)`.

Decay parameters include values below zero, exactly zero, interior values,
exactly four, and above four to test clamp endpoints and gradients. NLM
temperatures are deliberately non-unit to test division and its gradient.
The fixture verifies numerical fidelity of these configurations, not task
performance or fidelity of the separate world-model adapter.
