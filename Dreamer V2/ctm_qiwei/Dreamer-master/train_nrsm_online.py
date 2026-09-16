"""Bounded online NRSM Actor-Critic integration validation, separate run outputs."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import tensorflow as tf

from envs import DreamerV2UAVEnv
from nrsm_online_agent import ACConfig, OnlineAgent, compact_config
from nrsm_run_budget import RunBudget
from run_support import atomic_write, append_jsonl, require_free_space


SOURCE_FILES = ('train_nrsm_online.py', 'nrsm_online_agent.py', 'nrsm.py', 'ctm_rl_core.py',
                'validate_nrsm.py', 'models.py', 'tools.py', 'envs.py', 'uav_actions.py',
                'controller_baseline.py', 'run_support.py', 'run_wsl.sh', 'gpu_check.py', 'nrsm_run_budget.py')


def write_json(path, value):
    content = json.dumps(value, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda f: f.write(content))


def environment(seed):
    return DreamerV2UAVEnv('relay', seed=seed, max_episode_steps=100,
                           reward_mode='goal_safe_v1', reward_discount=.99, shaping_scale=1.)


def new_episode(obs):
    episode = dict(vector=np.zeros((101, 13), np.float32), action=np.zeros((101, 4), np.float32),
                   reward=np.zeros(101, np.float32), discount=np.zeros(101, np.float32),
                   valid=np.zeros(101, np.float32), is_first=np.zeros(101, np.float32))
    episode['vector'][0] = obs['vector']
    episode['valid'][0] = episode['is_first'][0] = 1
    episode['discount'][0] = .99
    return episode


def add_transition(episode, index, action, obs, reward, info):
    episode['vector'][index] = obs['vector']
    episode['action'][index] = action
    episode['reward'][index] = reward
    episode['discount'][index] = .99 * float(info['discount'])
    episode['valid'][index] = 1


def random_action(rng):
    action = np.zeros(4, np.float32)
    branch = int(rng.integers(2))
    action[branch], action[branch+2] = 1, rng.uniform(-1, 1)
    return action


def episode_record(obs, info, length, total_return, pickup, scene):
    record = dict(scene=int(scene), length=int(length), return_=float(total_return),
                  success=bool(obs['is_success']), pickup=bool(pickup),
                  timeout=bool(info['truncated']) and not bool(obs['is_success']) and not bool(obs['out_of_bounds']),
                  oob=bool(obs['out_of_bounds']))
    if sum(int(record[k]) for k in ('success', 'timeout', 'oob')) != 1:
        raise ValueError('Unexpected terminal outcome')
    return record


def evaluate(agent, seed_start, count, cancelled=lambda: False):
    """Same fixed layouts every checkpoint; no replay writes or optimizer calls."""
    envs = [environment(seed_start + i) for i in range(count)]
    try:
        observations = [e.reset() for e in envs]
        state = agent.world.core.initial(count)
        previous = tf.zeros([count, 4])
        active = np.ones(count, bool)
        pickup = np.zeros(count, bool)
        returns = np.zeros(count)
        records = []
        for step in range(1, 101):
            if cancelled():
                return dict(cancelled=True, completed_episodes=len(records), episodes=records)
            actions, state = agent.policy(tf.constant(np.stack([o['vector'] for o in observations])),
                                          previous, state, False)
            previous = actions
            actions = actions.numpy()
            for i in np.flatnonzero(active):
                observations[i], reward, done, info = envs[i].step(actions[i])
                returns[i] += reward
                pickup[i] |= bool(observations[i]['carrying_supply']) or bool(observations[i]['relay_reached'])
                if done:
                    records.append(episode_record(observations[i], info, step, float(returns[i]),
                                                  bool(pickup[i]), seed_start+i))
                    active[i] = False
            if not active.any():
                break
        assert len(records) == count
        return dict(episodes=sorted(records, key=lambda r: r['scene']), count=count,
                    **{key: float(np.mean([r[key] for r in records]))
                       for key in ('success', 'pickup', 'timeout', 'oob', 'length', 'return_')})
    finally:
        for env in envs:
            env.close()


def verify_restore(agent, checkpoint):
    clone = OnlineAgent(agent.core_config, agent.c)
    restored = clone.load(checkpoint)
    for name, group in agent.groups().items():
        for original, copied in zip(group, clone.groups()[name]):
            np.testing.assert_array_equal(original.numpy(), copied.numpy())
    # Carry across multiple decisions, not just a one-step weight shape check.
    states = [a.world.core.initial(2) for a in (agent, clone)]
    actions = [tf.zeros([2, 4]), tf.zeros([2, 4])]
    for step in range(3):
        for i, a in enumerate((agent, clone)):
            actions[i], states[i] = a.policy(tf.ones([2, 13]) * (step*.1), actions[i], states[i], False)
        np.testing.assert_allclose(actions[0], actions[1], atol=1e-6)
        for key in states[0]:
            np.testing.assert_allclose(states[0][key], states[1][key], atol=1e-5, rtol=1e-5)
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-seconds', type=int, default=None)
    parser.add_argument('--steps', type=int, default=0,
                        help='Cumulative environment-step target, INCLUDING restored progress; no time limit')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--prefill-steps', type=int, default=1000)
    parser.add_argument('--warmup-updates', type=int, default=50)
    parser.add_argument('--train-every', type=int, default=5)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--replay-episodes', type=int, default=500)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--final-eval-episodes', type=int, default=50)
    parser.add_argument('--final-eval-seed-start', type=int, default=71000)
    parser.add_argument('--checkpoint-seconds', type=int, default=300)
    parser.add_argument('--max-ac-updates', type=int, default=0, help='Optional extra cap for smoke tests')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    try:
        budget = RunBudget.from_args(args.duration_seconds, args.steps)
    except ValueError as error:
        parser.error(str(error))
    args.duration_seconds = budget.seconds
    if (min(args.prefill_steps, args.train_every, args.batch, args.replay_episodes,
            args.eval_episodes, args.final_eval_episodes, args.checkpoint_seconds) < 1 or
        args.warmup_updates < 0 or args.max_ac_updates < 0):
        parser.error('Invalid sizes')
    if args.output.exists():
        raise FileExistsError('New output required, including on resume')
    args.output.mkdir(parents=True)
    require_free_space(args.output, 1024**3)
    tf.keras.mixed_precision.set_global_policy('float32')
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.monotonic()
    wall_start = time.time()
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, _: stopped.append(signum))
    signal.signal(signal.SIGINT, lambda signum, _: stopped.append(signum))
    agent = OnlineAgent()
    counts = dict(env_steps=0, policy_env_steps=0, model_updates=0, ac_updates=0,
                  prefill_steps=0, warmup_updates=0, episodes=0)
    replay, next_scene = [], 80000 + args.seed * 10000
    training_config = {key: getattr(args, key) for key in
        ('seed', 'prefill_steps', 'warmup_updates', 'train_every', 'batch', 'replay_episodes')}
    if args.resume:
        state = agent.load(args.resume)
        if state['training_config'] != training_config:
            raise ValueError('Resume training protocol mismatch')
        counts, replay, next_scene = state['counts'], state['replay'], state['next_scene']
        rng.bit_generator.state = state['rng']
    if budget.steps and counts['env_steps'] >= budget.steps:
        raise ValueError('Step target must exceed restored environment steps')
    if (int(agent.model_opt.iterations.numpy()) != counts['model_updates'] or
        int(agent.actor_opt.iterations.numpy()) != counts['ac_updates'] or
        int(agent.critic_opt.iterations.numpy()) != counts['ac_updates'] or
        int(agent.ac_updates.numpy()) != counts['ac_updates']):
        raise ValueError('Optimizer counters do not match saved progress')
    initial_counts = dict(counts)
    core_hashes = {}
    for name in SOURCE_FILES:
        content = Path(__file__).with_name(name).read_bytes()
        core_hashes[name] = hashlib.sha256(content).hexdigest()
        atomic_write(args.output/'source'/name, lambda f, data=content: f.write(data))
    manifest = dict(contract=agent.contract, core_contract=agent.world.core.contract,
                    core_config=asdict(agent.core_config), ac_config=asdict(agent.c),
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    started_at=datetime.fromtimestamp(wall_start, timezone.utc).isoformat(),
                    source_sha256=core_hashes, tensorflow=tf.__version__,
                    gpu=[g.name for g in tf.config.list_physical_devices('GPU')],
                    actor_input='NRSM latent stoch+deter for online and imagination, no true-vector shortcut',
                    replay='Uniform complete-episode replay; padded 101 rows, valid/reset masks; bounded FIFO',
                    protocol='Random prefill; model warmup; joint world+actor+critic updates every 5 online steps by default; no BC/demo',
                    environment='unchanged MOVE/TURN auto-pickup; relay/goal_safe_v1; horizon100; gamma.99; shaping1',
                    evaluation=f'Deterministic latent argmax + actor mode, fixed scenes 70000+; final scenes {args.final_eval_seed_start}+',
                    budget=asdict(budget), initial_counts=initial_counts,
                    resume_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                    limitation='End-to-end engineering validation, not a matched DreamerV2 benchmark. '
                               'Checkpoint resumes at a new episode, not bit-exact TF RNG.')
    write_json(args.output/'manifest.json', manifest)
    latest_checkpoint = None

    def status(phase, **extra):
        write_json(args.output/'status.json', dict(status=phase, **counts,
            elapsed_seconds=time.monotonic()-start, started_at=manifest['started_at'],
            deadline_at=(datetime.fromtimestamp(wall_start+budget.seconds, timezone.utc).isoformat()
                         if budget.seconds else None),
            target_env_steps=budget.steps or None,
            remaining_env_steps=max(0, budget.steps-counts['env_steps']) if budget.steps else None,
            checkpoint=str(latest_checkpoint) if latest_checkpoint else None, **extra))

    def log(kind, **extra):
        record = dict(kind=kind, **counts, elapsed_seconds=time.monotonic()-start, **extra)
        append_jsonl(args.output/'metrics.jsonl', record, durable=True)
        print(json.dumps(record, allow_nan=False), flush=True)

    def expired():
        return bool(stopped) or budget.reached(time.monotonic()-start, counts['env_steps'])

    def update_allowed():
        # In step-budget mode, finish the update due on the last collected step.
        # A wall-clock deadline or signal still prevents another update.
        return not stopped and (not budget.seconds or time.monotonic()-start < budget.seconds)

    def save():
        nonlocal latest_checkpoint
        require_free_space(args.output, 1024**3)
        latest_checkpoint = args.output/f'checkpoint_env{counts["env_steps"]:07d}_ac{counts["ac_updates"]:06d}.pkl'
        agent.save(latest_checkpoint, dict(counts=dict(counts), replay=replay,
            next_scene=next_scene, rng=rng.bit_generator.state, training_config=training_config))
        status('running')

    def batch():
        selected = rng.integers(len(replay), size=args.batch)
        return {k: tf.constant(np.stack([replay[i][k] for i in selected])) for k in replay[0]}

    status('initializing')
    env = None
    before = None
    latest_metrics = {}
    last_log = last_checkpoint = time.monotonic()
    try:
        save()
        before = evaluate(agent, 70000, args.eval_episodes, expired)
        log('evaluation', split='fixed', metrics=before)
        while not expired():
            if args.max_ac_updates and counts['ac_updates'] >= args.max_ac_updates:
                break
            prefill = counts['prefill_steps'] < args.prefill_steps
            status('prefill' if prefill else 'online_training', latest_train=latest_metrics)
            env = environment(next_scene)
            scene = next_scene
            next_scene += 1
            observation = env.reset()
            episode = new_episode(observation)
            state = agent.world.core.initial(1)
            previous_action = tf.zeros([1, 4])
            total_return, pickup = 0., False
            for t in range(1, 101):
                if expired() or (args.max_ac_updates and counts['ac_updates'] >= args.max_ac_updates):
                    break
                if prefill:
                    action = random_action(rng)
                else:
                    sampled, state = agent.policy(tf.constant(observation['vector'][None]),
                                                   previous_action, state, True)
                    action = sampled.numpy()[0]
                observation, reward, done, info = env.step(action)
                previous_action = tf.constant(action[None])
                add_transition(episode, t, action, observation, reward, info)
                counts['env_steps'] += 1
                counts['prefill_steps' if prefill else 'policy_env_steps'] += 1
                total_return += reward
                pickup |= bool(observation['carrying_supply']) or bool(observation['relay_reached'])
                if done:
                    replay.append(episode)
                    del replay[:-args.replay_episodes]
                    counts['episodes'] += 1
                    log('episode', phase='prefill' if prefill else 'policy',
                        metrics=episode_record(observation, info, t, total_return, pickup, scene))
                    atomic_write(args.output/'episodes'/f'{scene}.npz',
                                 lambda f, e=episode: np.savez_compressed(f, **e))
                if not prefill and replay and counts['policy_env_steps'] % args.train_every == 0:
                    # Warmup is counted separately and is not represented as AC learning.
                    while counts['warmup_updates'] < args.warmup_updates and update_allowed():
                        status('model_warmup')
                        latest_metrics = {k: float(v) for k, v in agent.update(batch(), True).items()}
                        counts['warmup_updates'] += 1
                        counts['model_updates'] += 1
                        if time.monotonic()-last_log >= 30:
                            log('train', phase='model_warmup', metrics=latest_metrics)
                            last_log = time.monotonic()
                    if update_allowed():
                        latest_metrics = {k: float(v) for k, v in agent.update(batch()).items()}
                        counts['model_updates'] += 1
                        counts['ac_updates'] += 1
                        assert int(agent.ac_updates.numpy()) == counts['ac_updates']
                if time.monotonic()-last_log >= 30:
                    log('train', phase='prefill' if prefill else 'joint_ac', metrics=latest_metrics)
                    status('prefill' if prefill else 'online_training', latest_train=latest_metrics)
                    last_log = time.monotonic()
                if time.monotonic()-last_checkpoint >= args.checkpoint_seconds and not expired():
                    save()  # Save before evaluation; only completed replay episodes included.
                    status('evaluating', latest_train=latest_metrics)
                    evaluation = evaluate(agent, 70000, args.eval_episodes, expired)
                    log('evaluation', split='fixed', metrics=evaluation)
                    last_checkpoint = time.monotonic()
                if done:
                    break
            env.close()
            env = None
        save()
        status('finalizing')
        # Allow bounded final evaluation after training deadline. Signals cancel it.
        after = evaluate(agent, 70000, args.eval_episodes, lambda: bool(stopped))
        log('evaluation', split='fixed_final', metrics=after)
        fresh = evaluate(agent, args.final_eval_seed_start, args.final_eval_episodes, lambda: bool(stopped))
        log('evaluation', split='fresh_final', metrics=fresh)
        verify_restore(agent, latest_checkpoint)
        result = dict(status='stopped' if stopped else 'completed', **counts,
                      stop_reason='signal' if stopped else 'update_cap' if args.max_ac_updates and
                          counts['ac_updates'] >= args.max_ac_updates else budget.stop_reason,
                      target_env_steps=budget.steps or None,
                      before=before, after=after, fresh=fresh, latest_train=latest_metrics,
                      elapsed_seconds=time.monotonic()-start, checkpoint=str(latest_checkpoint),
                      checkpoint_restore_verified=True,
                      optimizer_iterations=dict(world=int(agent.model_opt.iterations.numpy()),
                          actor=int(agent.actor_opt.iterations.numpy()), critic=int(agent.critic_opt.iterations.numpy())),
                      scope='Online world+Actor+Critic integration; not a convergence or matched-baseline claim')
        write_json(args.output/'result.json', result)
        status(result['status'], stop_reason=result['stop_reason'], checkpoint_restore_verified=True)
        print(json.dumps(result), flush=True)
    except Exception as error:
        # Preserve previous good snapshot. Never overwrite it with suspect weights.
        status('failed', error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
