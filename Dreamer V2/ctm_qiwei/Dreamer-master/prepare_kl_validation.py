"""Prepare isolated, bounded KL-fixed continuations; never launch training."""
import argparse
import contextlib
import datetime
import json
import pathlib
import pickle
import shutil

import numpy as np

from prepare_stage2 import digest, write_json
from run_support import RunLock, require_free_space


def prepare(previous, root, seeds=(1,)):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0, 1, 2) for seed in seeds):
        raise ValueError('Use distinct seeds from 0, 1, 2')
    previous, root = pathlib.Path(previous).resolve(), pathlib.Path(root).resolve()
    repository = pathlib.Path(__file__).resolve().parent
    if root == previous or root.is_relative_to(previous) or previous.is_relative_to(root):
        raise ValueError('Destination must be separate from the parent')
    if root.exists():
        raise FileExistsError('Refusing to overwrite existing experiment')
    with contextlib.ExitStack() as locks:
        locks.enter_context(RunLock(previous / '.stage.lock'))
        locks.enter_context(RunLock(previous / '.batch.lock'))
        for seed in sorted(seeds):
            locks.enter_context(RunLock(previous / f'seed_{seed}' / '.run.lock'))
        batch = json.loads((previous / 'batch_status.json').read_text())
        stage = json.loads((previous / 'stage_status.json').read_text())
        manifest = json.loads((previous / 'checkpoint_manifest.json').read_text())
        if batch['status'] != 'completed' or stage['status'] != 'completed':
            raise ValueError('Parent stage must be completed')
        hashes = {}
        for name, expected in batch['source_sha256'].items():
            if digest(previous / 'source' / name) != expected:
                raise ValueError(f'Parent source changed: {name}')
            hashes[name] = digest(repository / name)
            if name != 'models.py' and hashes[name] != expected:
                raise ValueError(f'Unexpected non-KL source change: {name}')
        if hashes['models.py'] != 'c95064dd88f7e6af4d327484b27754a4e15a9939717846c80fabacc4818b59b4':
            raise ValueError('models.py is not the tested KL fix')
        parents = []
        for seed in seeds:
            old_run = previous / f'seed_{seed}'
            entry = next(item for item in batch['runs'] if item['seed'] == seed)
            checkpoint = old_run / 'variables.pkl'
            parent_hash = digest(checkpoint)
            if parent_hash != manifest[str(checkpoint.relative_to(previous))]['sha256']:
                raise ValueError('Parent checkpoint hash mismatch')
            with checkpoint.open('rb') as stream:
                payload = pickle.load(stream)
            if payload['step'] != {0: 100050, 1: 100094, 2: 100004}[seed] or payload['step'] != entry['final_step']:
                raise ValueError('Unexpected parent step')
            if payload['warmup'] != dict(model=1000, actor=3000):
                raise ValueError('Incomplete parent warmup')
            if not all(np.isfinite(value).all() for value in payload['variables']):
                raise ValueError('Non-finite parent checkpoint')
            episodes = list((old_run / 'episodes').glob('*.npz'))
            if sum(int(path.stem.rsplit('-', 1)[1]) - 1 for path in episodes) != payload['step']:
                raise ValueError('Replay/checkpoint step mismatch')
            if len(list((old_run / 'episodes' / 'demonstrations').glob('*.npz'))) != 64:
                raise ValueError('Missing demonstrations')
            parents.append(dict(seed=seed, step=payload['step'], checkpoint_sha256=parent_hash))
        require_free_space(previous, 5 * len(seeds) * 2**30)
        root.mkdir(parents=True)
        source = root / 'source'
        source.mkdir()
        for name, expected in hashes.items():
            shutil.copy2(repository / name, source / name)
            if digest(source / name) != expected:
                raise ValueError(f'Copied source mismatch: {name}')
        runs = []
        for parent in parents:
            seed = parent['seed']
            old_run, run = previous / f'seed_{seed}', root / f'seed_{seed}'
            entry = next(item for item in batch['runs'] if item['seed'] == seed)
            run.mkdir()
            shutil.copytree(old_run / 'episodes', run / 'episodes')
            for name in ('variables.pkl', 'metrics.jsonl', 'run_metadata.jsonl'):
                shutil.copy2(old_run / name, run / name)
                if digest(old_run / name) != digest(run / name):
                    raise ValueError(f'Copied file mismatch: {name}')
            for path in (old_run / 'episodes').rglob('*.npz'):
                if digest(path) != digest(run / path.relative_to(old_run)):
                    raise ValueError('Copied replay mismatch')
            (run / 'checkpoints').mkdir()
            shutil.copy2(run / 'variables.pkl', run / 'checkpoints' / f"step_{parent['step']:09d}.pkl")
            command = list(entry['command'])
            command[command.index('--steps') + 1] = '300000'
            command[command.index('--logdir') + 1] = str(run)
            runs.append(dict(seed=seed, status='pending', outdir=str(run), command=command, attempts=0))
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        provenance = dict(
            protocol='kl_fixed_selected_seeds_100k_to_300k_v1', previous_stage=str(previous),
            seeds=seeds, target_steps=300000, parent_checkpoints=parents,
            changed_source_files=['models.py'], source_sha256=hashes,
            preparer_sha256=digest(__file__),
            note='Only KL gradient mixing changed. Copied optimizer, warmup, replay and historical metrics. '
                 'History up to each parent step uses old KL. RNG/environment/sampler restart. '
                 'Run selected seeds sequentially, then stop; no automatic final evaluation jobs.')
        write_json(root / 'continuation_provenance.json', provenance)
        shutil.copy2(__file__, root / pathlib.Path(__file__).name)
        write_json(root / 'batch_status.json', dict(
            status='prepared', runner_contract='long_run_safe_v1', started_at=now,
            source_sha256=hashes, runs=runs))
        for parent in parents:
            if digest(previous / f"seed_{parent['seed']}" / 'variables.pkl') != parent['checkpoint_sha256']:
                raise ValueError('Parent checkpoint changed during preparation')
        return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=pathlib.Path, required=True)
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous, args.outdir, args.seeds), indent=2))
