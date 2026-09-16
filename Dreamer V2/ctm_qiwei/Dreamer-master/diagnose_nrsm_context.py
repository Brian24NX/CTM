"""Read-only model probe; writes a NEW report, never changes saved weights.

Crossed context/latent features are diagnostics, not valid generative states
and not interventions sufficient to establish a training root cause.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

import tools
from nrsm import NRSMConfig
from validate_nrsm import DiagnosticWorldModel, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    output = args.run/'context_probe.json'
    if output.exists():
        raise FileExistsError(output)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('float32')
    manifest = json.loads((args.run/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed since the run: ' + name)
    with np.load(args.run/'data.npz') as archive:
        data = {k.removeprefix('heldout_'): tf.constant(archive[k]) for k in archive.files if k.startswith('heldout_')}
    model = DiagnosticWorldModel(NRSMConfig(**manifest['config']))
    model.forward({k: v[:1, :2] for k, v in data.items()}, False)
    checkpoint = args.run/f"step_{manifest['args']['updates']:04d}.pkl"
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)  # Own trusted local artifact only.
    if saved['contract'] != model.core.contract or saved['config'] != manifest['config']:
        raise ValueError('Checkpoint contract mismatch')
    if len(saved['weights']) != len(model.variables):
        raise ValueError('Checkpoint weight count mismatch')
    for variable, value in zip(model.variables, saved['weights']):
        if tuple(variable.shape) != value.shape or not np.isfinite(value).all():
            raise ValueError('Checkpoint weight mismatch')
    for variable, value in zip(model.variables, saved['weights']):
        variable.assign(value)
    post, prior, _, _, _ = model.forward(data, False)
    valid = data['valid'] * (1-data['is_first'])
    result = {}
    for context_name, context_state in [('post', post), ('prior', prior)]:
        for latent_name, latent_state in [('post', post), ('prior', prior)]:
            crossed = dict(deter=context_state['deter'], stoch=latent_state['stoch'])
            pred = model.vector(model.core.get_feat(crossed)).mean()
            result[f'{context_name}_context_{latent_name}_latent_rmse'] = float(tf.sqrt(
                tools.masked_mean(tf.reduce_mean((pred-data['vector'])**2, -1), valid)))
    _, kl = model.core.kl_loss(post, prior, valid=valid)
    result['categorical_kl'] = float(kl)
    result['context_rms_difference'] = float(tf.sqrt(tools.masked_mean(
        tf.reduce_mean((post['deter']-prior['deter'])**2, -1), valid)))
    result['scope'] = 'Same held-out rows and weights; diagnostic crossed features, not a policy or training ablation'
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
