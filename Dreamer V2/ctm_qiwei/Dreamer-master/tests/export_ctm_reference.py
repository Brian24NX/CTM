"""Export forward/backward fixtures from the unmodified upstream CTM RL model.

Run with the isolated CPU PyTorch environment, NOT the Dreamer TensorFlow venv.
The parent launches ``python -I`` and the worker prepends only the vendored
upstream root, preventing the local ``models.py`` from shadowing upstream code.
No handwritten CTM equations are used to produce the reference outputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


UPSTREAM_SHA = "4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466"
REFERENCE_TORCH_VERSION = "2.7.0+cpu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-nlms", type=int, choices=(0, 1), default=1)
    parser.add_argument("--layernorm", type=int, choices=(0, 1), default=1)
    parser.add_argument("--synapse-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def export(args):
    import numpy as np
    import torch

    if torch.__version__ != REFERENCE_TORCH_VERSION:
        raise RuntimeError(
            f"Reference fixtures require torch=={REFERENCE_TORCH_VERSION}; "
            f"found {torch.__version__}. Clamp endpoint gradients differ in "
            "other releases, so changing this runtime changes the contract."
        )

    project_root = Path(__file__).resolve().parents[1]
    reference_root = project_root / "third_party" / "ctm_reference"
    sys.path.insert(0, str(reference_root))
    from models.ctm_rl import ContinuousThoughtMachineRL
    import models.ctm_rl as upstream_module

    # Fail rather than accidentally comparing against a locally rewritten model.
    if Path(upstream_module.__file__).resolve() != (
        reference_root / "models" / "ctm_rl.py"
    ).resolve():
        raise RuntimeError("Reference import did not resolve to vendored upstream source")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = dict(
        iterations=3, d_model=8, d_input=6, n_synch_out=4,
        synapse_depth=args.synapse_depth, memory_length=4,
        deep_nlms=bool(args.deep_nlms), memory_hidden_dims=5,
        do_layernorm_nlm=bool(args.layernorm),
        backbone_type="classic-control-backbone", dropout=0.0,
        neuron_select_type="first-last",
    )
    model = ContinuousThoughtMachineRL(**config).cpu().eval()
    steps, batch, input_size = 3, 2, 7
    inputs = torch.randn(steps, batch, input_size, requires_grad=True)

    def initial_states():
        # Retain links to learned parameters: initial traces are not detached.
        return (
            model.start_trace.unsqueeze(0).expand(batch, -1, -1),
            model.start_activated_trace.unsqueeze(0).expand(batch, -1, -1),
        )

    # Materialize LazyLinear weights before recording or changing parameters.
    with torch.no_grad():
        model(inputs[0], initial_states())
        model.decay_params_out.copy_(torch.tensor(
            [-0.5, 0.0, 0.001, 0.15, 0.7, 1.5, 3.9, 4.0, 4.5, 2.0]
        ))
        temperature_index = 0
        for name, parameter in model.named_parameters():
            if name.endswith(".T"):
                parameter.fill_(1.3 if temperature_index == 0 else 0.7)
                temperature_index += 1

    def array(value):
        return value.detach().cpu().numpy().copy()

    initial_pre, initial_post = initial_states()
    payload = {
        "upstream_sha": np.asarray(UPSTREAM_SHA),
        "fixture_contract": np.asarray("ctm_rl_official_forward_backward_v1"),
        "torch_version": np.asarray(torch.__version__),
        "numpy_version": np.asarray(np.__version__),
        "seed": np.asarray(args.seed),
        "input_size": np.asarray(input_size),
        "initial_pre": array(initial_pre),
        "initial_post": array(initial_post),
        "inputs": array(inputs),
    }
    payload.update({f"config__{name}": np.asarray(value) for name, value in config.items()})
    payload.update({f"weight__{name}": array(value) for name, value in model.state_dict().items()})
    source_hashes = {
        str(path.relative_to(reference_root)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((reference_root / "models").glob("*.py"))
    }
    payload["source_sha256_json"] = np.asarray(json.dumps(source_hashes, sort_keys=True))

    backbone_outputs = []
    hook = model.backbone.register_forward_hook(
        lambda module, arguments, output: backbone_outputs.append(array(output))
    )
    state = initial_pre, initial_post
    results, pres, posts, pre_states, post_states, sync_ticks = [], [], [], [], [], []
    for step in range(steps):
        previous_post = state[1]
        result, state, pre_tick, post_tick = model(inputs[step], state, track=True)
        results.append(result)
        pres.append(pre_tick)
        posts.append(post_tick)
        pre_states.append(array(state[0]))
        post_states.append(array(state[1]))
        # The official forward returns synchronisation only at the final tick.
        # Rebuild tracked windows by concatenation and invoke its actual method
        # to additionally check the sliding-window synchronisation at every tick.
        tick_window = previous_post.detach()
        tick_sync = []
        with torch.no_grad():
            for activation in post_tick:
                tick_window = torch.cat((
                    tick_window[:, :, 1:], torch.from_numpy(activation).unsqueeze(-1)
                ), dim=-1)
                tick_sync.append(array(model.compute_synchronisation(tick_window)))
        sync_ticks.append(np.stack(tick_sync))
    hook.remove()

    result = torch.stack(results)
    loss_weights = torch.linspace(-0.8, 1.1, result.numel()).reshape_as(result)
    loss = (result * loss_weights).sum() + 0.03 * state[0].square().sum() + 0.07 * state[1].square().sum()
    loss.backward()
    missing_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing_gradients.append(name)
        else:
            payload[f"grad__{name}"] = array(parameter.grad)
    if missing_gradients:
        raise RuntimeError(f"Unexpected unused upstream parameters: {missing_gradients}")
    payload.update(
        result=array(result), pre_ticks=np.stack(pres), post_ticks=np.stack(posts),
        sync_ticks=np.stack(sync_ticks), pre_states=np.stack(pre_states),
        post_states=np.stack(post_states), backbone=np.stack(backbone_outputs),
        final_pre=array(state[0]), final_post=array(state[1]),
        input_grad=array(inputs.grad), loss_weights=array(loss_weights), loss=array(loss),
        parameter_names=np.asarray([name for name, _ in model.named_parameters()]),
    )
    for name, value in payload.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite upstream fixture field: {name}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; use --force explicitly")
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "output": str(output), "config": config,
        "parameters": len(payload["parameter_names"]), "loss": float(loss.detach()),
        "upstream_sha": UPSTREAM_SHA,
    }, sort_keys=True))


def main():
    args = parse_args()
    if args.worker:
        export(args)
        return
    # Isolated interpreter prevents cwd/PYTHONPATH collisions with Dreamer models.
    command = [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
