"""Bounded P2 offline world-model diagnostic. Never imports the Dreamer agent.

Full episodes, unchanged environment, held-out seeds, no Actor or task changes.
Diagnostic controller data is NOT a demonstration-enabled policy benchmark.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import signal
import time
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

import models
import tools
from controller_baseline import controller_action
from envs import DreamerV2UAVEnv
from nrsm import NRSM, NRSMConfig
from ctm_rl_core import UPSTREAM_REVISION, REFERENCE_TORCH_VERSION
from run_support import atomic_write, require_free_space


def write_json(path, value):
    atomic_write(path, lambda f: f.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def collect(count, seed_start):
    arrays = {k: np.zeros((count, 101) + shape, np.float32) for k, shape in
              dict(vector=(13,), action=(4,), reward=(), discount=(), valid=(), is_first=()).items()}
    records = []
    for i in range(count):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        env = DreamerV2UAVEnv('relay', max_episode_steps=100, seed=seed,
                             reward_mode='goal_safe_v1', reward_discount=.99, shaping_scale=1.)
        try:
            obs = env.reset()
            arrays['vector'][i, 0] = obs['vector']
            arrays['discount'][i, 0] = .99
            arrays['valid'][i, 0] = arrays['is_first'][i, 0] = 1
            policy = 'controller' if i % 2 == 0 else 'random'
            for t in range(1, 101):
                if policy == 'controller':
                    action = controller_action(env)
                else:
                    action = np.zeros(4, np.float32)
                    branch = int(rng.integers(2))
                    action[branch], action[branch+2] = 1., rng.uniform(-1, 1)
                obs, reward, done, info = env.step(action)
                arrays['vector'][i, t] = obs['vector']
                arrays['action'][i, t] = action
                arrays['reward'][i, t] = reward
                arrays['discount'][i, t] = .99 * float(info['discount'])
                arrays['valid'][i, t] = 1
                if done:
                    break
            records.append(dict(seed=seed, policy=policy, length=t,
                                success=bool(obs['is_success'])))
        finally:
            env.close()
    return arrays, records


class DiagnosticWorldModel(tools.Module):
    def __init__(self, config):
        super().__init__()
        self.core = NRSM(config)
        self.encoder = models.VectorEncoder(config.embed)
        self.vector = models.DenseHead((13,), layers=2, units=config.hidden)
        self.reward = models.DenseHead((), layers=2, units=config.hidden)
        self.discount = models.DenseHead((), layers=2, units=config.hidden, dist='binary')

    def forward(self, data, sample):
        post, prior = self.core.observe(self.encoder(data), data['action'],
                                        is_first=data['is_first'], valid=data['valid'], sample=sample)
        feat = self.core.get_feat(post)
        return post, prior, self.vector(feat), self.reward(feat), self.discount(feat)

    def objective(self, data):
        post, prior, vector, reward, discount = self.forward(data, True)
        valid = data['valid']
        # Initial observation has no preceding transition/reward target.
        transition = valid * (1. - data['is_first'])
        kl, raw_kl = self.core.kl_loss(post, prior, balance=.8, free=0., valid=valid)
        vector_nll = tools.masked_mean(-vector.log_prob(data['vector']), valid)
        reward_nll = tools.masked_mean(-reward.log_prob(data['reward']), transition)
        discount_bce = tools.masked_mean(-discount.log_prob(data['discount']), transition)
        loss = 10. * vector_nll + reward_nll + discount_bce + kl
        return loss, dict(loss=loss, kl=raw_kl, vector_nll=vector_nll,
                          reward_nll=reward_nll, discount_bce=discount_bce)


def evaluate(model, data):
    post, prior, vector, reward, discount = model.forward(data, False)
    valid = data['valid'] * (1. - data['is_first'])
    def rmse(pred, target, mask):
        squared = (pred - target)**2
        if squared.shape.rank == 3:
            squared = tf.reduce_mean(squared, -1)
        return float(tf.sqrt(tools.masked_mean(squared, mask)))
    result = dict(posterior_vector_rmse=rmse(vector.mean(), data['vector'], valid),
                  prior_vector_rmse=rmse(model.vector(model.core.get_feat(prior)).mean(), data['vector'], valid),
                  posterior_reward_rmse=rmse(reward.mean(), data['reward'], valid),
                  prior_reward_rmse=rmse(model.reward(model.core.get_feat(prior)).mean(), data['reward'], valid),
                  posterior_discount_bce=float(tools.masked_mean(-discount.log_prob(data['discount']), valid)))
    # All valid windows; no reset/terminal crossing. Actual future actions only,
    # never future observations. Start from the posterior at each anchor.
    for horizon in (1, 5, 15):
        length = data['action'].shape[1] - horizon
        anchors = {k: tf.reshape(v[:, :length], [-1] + list(v.shape[2:])) for k, v in post.items()}
        future = tf.stack([data['action'][:, j:j+length] for j in range(1, horizon+1)], 2)
        future = tf.reshape(future, [-1, horizon, 4])
        imagined = model.core.imagine(future, anchors, sample=False)
        final = {k: v[:, -1] for k, v in imagined.items()}
        pred = tf.reshape(model.vector(model.core.get_feat(final)).mean(), [-1, length, 13])
        mask = data['valid'][:, :length] * data['valid'][:, horizon:]
        result[f'open_loop_{horizon}_vector_rmse'] = rmse(pred, data['vector'][:, horizon:], mask)
        result[f'persistence_{horizon}_vector_rmse'] = rmse(data['vector'][:, :length], data['vector'][:, horizon:], mask)
    return result


def save_snapshot(path, model, optimizer, step, rng):
    require_free_space(path.parent, 256 * 1024**2)
    payload = dict(contract=model.core.contract, config=asdict(model.core.config), step=step,
                   weights=[v.numpy() for v in model.variables],
                   optimizer=[v.numpy() for v in optimizer.variables],
                   numpy_rng=rng.bit_generator.state,
                   scope='Offline diagnostic weights and optimizer; TF RNG not bit-exact resumable')
    if not all(np.isfinite(x).all() for x in payload['weights'] + payload['optimizer']):
        raise ValueError('Non-finite snapshot refused')
    atomic_write(path, lambda f: pickle.dump(payload, f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--train-episodes', type=int, default=16)
    parser.add_argument('--eval-episodes', type=int, default=8)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--duration-seconds', type=int, default=0,
                        help='Explicit timed validation, 1..3600 seconds; supersedes --updates')
    args = parser.parse_args()
    if not 0 <= args.duration_seconds <= 3600:
        parser.error('Timed validation is capped at one hour')
    if (not args.duration_seconds and not 1 <= args.updates <= 200) or min(args.train_episodes, args.eval_episodes, args.batch) < 1:
        parser.error('Require positive sizes and either 1..200 updates or an explicit timed budget')
    if args.train_episodes > 1000 or args.eval_episodes > 1000:
        parser.error('Diagnostic collection is capped at 1000 episodes per split')
    tf.keras.mixed_precision.set_global_policy('float32')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_unix = time.time()
    stop_requested = []
    def request_stop(signum, frame):
        stop_requested.append(signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def status(state, step=0, **extra):
        write_json(args.output/'status.json', dict(status=state, updates=step,
            started_at=datetime.fromtimestamp(started_unix, timezone.utc).isoformat(),
            deadline_at=(datetime.fromtimestamp(started_unix+args.duration_seconds, timezone.utc).isoformat()
                         if args.duration_seconds else None),
            elapsed_seconds=time.monotonic()-started, **extra))
    status('initializing')
    require_free_space(args.output, 512 * 1024**2)
    config = NRSMConfig(neurons=64, memory=8, ticks=2, sync_neurons=16, context=128,
                        hidden=128, nlm_hidden=8, stoch=8, classes=8, embed=128)
    hashes = {}
    for name in ('nrsm.py', 'ctm_rl_core.py', 'validate_nrsm.py', 'models.py', 'tools.py', 'envs.py',
                 'uav_actions.py', 'controller_baseline.py', 'run_support.py'):
        content = Path(__file__).with_name(name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        atomic_write(args.output/'source'/name, lambda f, b=content: f.write(b))
    manifest = dict(config=asdict(config), core_contract=NRSM.contract,
                    upstream_revision=UPSTREAM_REVISION, reference_torch_version=REFERENCE_TORCH_VERSION,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256=hashes, tensorflow=tf.__version__,
                    gpu=[x.name for x in tf.config.list_physical_devices('GPU')],
                    evaluation='categorical argmax, normalized vector RMSE; not Monte Carlo expectation',
                    environment='unchanged relay/goal_safe_v1, horizon=100, discount=.99, shaping=1',
                    objective='10*vector Gaussian NLL + reward Gaussian NLL + discount BCE + balanced KL (.8/free0)',
                    limitations='Compact profile, offline diagnostic, no Actor, no task performance claim; TF RNG not bit-exact resumable')
    write_json(args.output/'manifest.json', manifest)
    train, train_records = collect(args.train_episodes, 20000)
    heldout, heldout_records = collect(args.eval_episodes, 40000)
    write_json(args.output/'episodes.json', dict(train=train_records, heldout=heldout_records))
    atomic_write(args.output/'data.npz', lambda f: np.savez_compressed(f,
                 **{'train_'+k: v for k, v in train.items()}, **{'heldout_'+k: v for k, v in heldout.items()}))
    heldout = {k: tf.constant(v) for k, v in heldout.items()}
    model = DiagnosticWorldModel(config)
    model.forward({k: tf.constant(v[:1, :2]) for k, v in train.items()}, False)
    variables = model.trainable_variables
    optimizer = tf.keras.optimizers.Adam(3e-4)
    optimizer.build(variables)

    @tf.function
    def update(batch):
        with tf.GradientTape() as tape:
            loss, metrics = model.objective(batch)
        gradients = tape.gradient(loss, variables)
        for var, gradient in zip(variables, gradients):
            if gradient is None:
                raise ValueError('Missing gradient: ' + var.name)
            tf.debugging.assert_all_finite(gradient, 'Non-finite gradient')
        gradients, norm = tf.clip_by_global_norm(gradients, 100.)
        tf.debugging.assert_all_finite(loss, 'Non-finite loss')
        tf.debugging.assert_all_finite(norm, 'Non-finite gradient norm')
        optimizer.apply_gradients(zip(gradients, variables))
        for var in variables:
            tf.debugging.assert_all_finite(var, 'Non-finite parameter after update')
        return dict(metrics, gradient_norm=norm)

    before = evaluate(model, heldout)
    history = [dict(update=0, heldout=before)]
    save_snapshot(args.output/'step_0000.pkl', model, optimizer, 0, rng)
    write_json(args.output/'progress.json', history)
    print(json.dumps(history[-1]), flush=True)
    step = 0
    window_norms = []
    last_checkpoint = last_log = time.monotonic()
    checkpoint_path = args.output/'step_0000.pkl'
    status('running', checkpoint=str(checkpoint_path))
    def remaining():
        return (time.monotonic()-started < args.duration_seconds if args.duration_seconds else step < args.updates)
    try:
      while remaining() and not stop_requested:
        indices = rng.integers(args.train_episodes, size=args.batch)
        metrics = update({k: tf.constant(v[indices]) for k, v in train.items()})
        step += 1
        window_norms.append(float(metrics['gradient_norm']))
        end = not remaining() or bool(stop_requested)
        now = time.monotonic()
        checkpoint_due = end or (now-last_checkpoint >= 300 if args.duration_seconds else step % 50 == 0)
        log_due = end or checkpoint_due or (now-last_log >= 30 if args.duration_seconds else step % 10 == 0)
        if log_due:
            record = dict(update=step, elapsed_seconds=time.monotonic()-started,
                          train={k: float(v) for k, v in metrics.items()})
            record['train'].update(gradient_norm_window_max=max(window_norms),
                                   clipped_fraction=float(np.mean(np.asarray(window_norms) > 100.)))
            if checkpoint_due:
                checkpoint_path = args.output/f'step_{step:04d}.pkl'
                save_snapshot(checkpoint_path, model, optimizer, step, rng)
                # Save before evaluation, so evaluation failure cannot lose training progress.
                record['heldout'] = evaluate(model, heldout)
                last_checkpoint = time.monotonic()
            history.append(record)
            write_json(args.output/'progress.json', history)
            print(json.dumps(record), flush=True)
            last_log = time.monotonic()
            window_norms.clear()
            status('running', step, checkpoint=str(checkpoint_path), latest=record)
    except Exception as error:
        status('failed', step, checkpoint=str(checkpoint_path), error=repr(error))
        raise
    # Covers a signal/deadline arriving between iterations or during evaluation.
    checkpoint_path = args.output/f'step_{step:04d}.pkl'
    save_snapshot(checkpoint_path, model, optimizer, step, rng)
    after = evaluate(model, heldout)
    history.append(dict(update=step, elapsed_seconds=time.monotonic()-started, heldout=after, final=True))
    write_json(args.output/'progress.json', history)
    status('finalizing', step, checkpoint=str(checkpoint_path))
    # Verify a fresh instance reproduces the final deterministic predictions,
    # and that optimizer slots can be restored with identical shape and values.
    clone = DiagnosticWorldModel(config)
    clone.forward({k: v[:1, :2] for k, v in heldout.items()}, False)
    clone_opt = tf.keras.optimizers.Adam(3e-4)
    clone_opt.build(clone.trainable_variables)
    with checkpoint_path.open('rb') as f:
        saved = pickle.load(f)
    for dest, source in ((clone.variables, saved['weights']), (clone_opt.variables, saved['optimizer'])):
        if len(dest) != len(source):
            raise AssertionError('Checkpoint variable count mismatch')
        for variable, value in zip(dest, source):
            if tuple(variable.shape) != value.shape:
                raise AssertionError('Checkpoint variable shape mismatch')
            variable.assign(value)
            np.testing.assert_array_equal(variable.numpy(), value)
    restored = evaluate(clone, heldout)
    for key in after:
        np.testing.assert_allclose(after[key], restored[key], rtol=1e-5, atol=1e-6)
    result = dict(before=before, after=after, checkpoint_restore_verified=True,
                  updates=step, trainable_parameters=int(sum(np.prod(v.shape) for v in variables)),
                  elapsed_seconds=time.monotonic()-started, status='stopped' if stop_requested else 'completed',
                  stop_reason='signal' if stop_requested else 'duration' if args.duration_seconds else 'updates',
                  warning='P2 engineering diagnostic only. Improvement does not establish a better policy or long-memory advantage.')
    write_json(args.output/'result.json', result)
    status(result['status'], step, checkpoint=str(checkpoint_path), stop_reason=result['stop_reason'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
