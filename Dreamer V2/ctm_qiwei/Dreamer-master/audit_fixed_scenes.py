"""Paired, inference-only UAV diagnostic; never changes checkpoint or replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


MODES = ('baseline', 'unstall_actor_move', 'unstall_positive_move',
         'reconstructed', 'reconstructed_context20', 'controller')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unstall(action, move_parameter, speed, streak, positive=False):
    action = np.array(action, dtype=np.float32, copy=True)
    stalled_turn = speed <= .1 and int(np.argmax(action[:2])) == 1
    streak = streak + 1 if stalled_turn else 0
    triggered = streak >= 5
    if triggered:
        action[:] = [1., 0., 1. if positive else move_parameter, 0.]
        streak = 0
    return action, streak, triggered


def paired(base, other):
    assert [r['scene'] for r in base] == [r['scene'] for r in other]
    result = {}
    for field in ('success', 'pickup', 'timeout', 'oob'):
        a = np.array([r[field] for r in base], int)
        b = np.array([r[field] for r in other], int)
        gain, loss = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
        n = gain + loss
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / 2**n) if n else 1.
        rng = np.random.RandomState(42)
        samples = (b - a)[rng.randint(0, len(a), size=(4000, len(a)))].mean(axis=1)
        result[field] = dict(delta_pp=float(100 * (b - a).mean()), gained=gain, lost=loss,
                             paired_bootstrap95_pp=(100 * np.percentile(samples, [2.5, 97.5])).tolist(),
                             mcnemar_exact_p=p)
    return result


def prediction_errors(pred, true, persistence, reward, continuation, pred_reward, pred_discount):
    if not len(true):
        return dict(n=0)
    delta = pred - true
    angle = np.arctan2(pred[:, 3], pred[:, 4]) - np.arctan2(true[:, 3], true[:, 4])
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    nonterminal = continuation > 0
    return dict(n=len(true), position_mae_m=float(np.linalg.norm(delta[:, :2] * 1000, axis=1).mean()),
        active_goal_mae_m=float(np.linalg.norm(delta[:, 11:13] * 2000, axis=1).mean()),
        persistence_position_mae_m=float(np.linalg.norm((persistence[:, :2] - true[:, :2]) * 1000, axis=1).mean()),
        speed_mae=float(np.abs(delta[:, 2] * 20).mean()),
        heading_mae_deg=float(np.abs(angle).mean() * 180 / np.pi),
        time_mae_steps=float(np.abs(delta[:, 9] * 50).mean()),
        phase_accuracy=float(((pred[:, 10] > 0) == (true[:, 10] > 0)).mean()),
        reward_mae=float(np.abs(pred_reward - reward).mean()),
        continuation_brier=float(np.square(pred_discount - .99 * continuation).mean()),
        nonterminal_count=int(nonterminal.sum()),
        false_terminal_fraction=float((pred_discount[nonterminal] < .5).mean()) if nonterminal.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=60000)
    args = parser.parse_args()
    if args.episodes < 1 or args.outdir.exists():
        raise ValueError('Need a positive budget and a new output directory')
    run = args.run.resolve()
    source = run.parent / 'source'
    state = json.loads((run.parent / 'batch_status.json').read_text())
    entry = next(r for r in state['runs'] if Path(r['outdir']).resolve() == run)
    if entry['status'] != 'completed':
        raise ValueError('Only diagnose completed, immutable checkpoints')
    for name, expected in state['source_sha256'].items():
        assert digest(source / name) == expected, name
    checkpoint_hash = digest(run / 'variables.pkl')
    replay_before = {str(p.relative_to(run)): (p.stat().st_size, p.stat().st_mtime_ns)
                     for p in (run / 'episodes').rglob('*.npz')}
    sys.path.insert(0, str(source))
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import dreamer
    import envs
    from controller_baseline import controller_action
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('float32')
    config = argparse.Namespace(**json.loads((run / 'run_metadata.jsonl').read_text().splitlines()[-1])['config'])
    config.logdir, config.batch_size, config.dataset_prefetch = run, 1, 1
    assert config.action_contract == 'move_turn_autopickup_v3'
    assert config.time_limit == 100 and config.imag_horizon == 15
    def make_env(seed):
        return envs.DreamerV2UAVEnv('relay', seed=seed, max_episode_steps=100,
            reward_mode=config.reward_mode, reward_discount=config.discount,
            shaping_scale=config.shaping_scale)
    environment = make_env(args.seed_start)
    agent = dreamer.DreamerV2(config, run / 'episodes', environment.action_space,
                              environment.observation_space, None)
    agent.load(run / 'variables.pkl')
    observation = environment.reset()
    original, _ = agent.policy({k: v[None] for k, v in observation.items()}, None, False)
    np.testing.assert_allclose(original.numpy(), agent.actor(observation['vector'][None]).mode().numpy(), atol=1e-7)
    environment.close()
    def variable_hash():
        return hashlib.sha256(b''.join(v.numpy().tobytes() for v in agent.variables)).hexdigest()
    before = variable_hash()
    iterations = [int(o._opt.iterations.numpy()) for o in (agent.model_opt, agent.actor_opt, agent.critic_opt)]
    args.outdir.mkdir(parents=True)
    protocol = dict(checkpoint=str(run / 'variables.pkl'), checkpoint_sha256=checkpoint_hash,
        checkpoint_step=int(agent.step.numpy()), seed_start=args.seed_start, episodes=args.episodes,
        horizon=100, modes=MODES, stochastic_seed=20260914,
        turn_intervention='On the fifth consecutive proposed TURN with observed speed <=0.1, force MOVE. '
                          'Either use actor MOVE mean, or parameter +1 (adds 4 speed). No heading correction.',
        actor_comparison='Same frozen actor; true vector vs posterior-decoded vector, without retraining. '
                         'Context20 resets only latent filtering every 20 observations.',
        prediction='Baseline and controller trajectories only; identical future actions at horizons 1,5,15. '
                   'Anchors every five steps, plus immediately before pickup/termination. No future observations. '
                   'Full and context20 use identical target indices; one seeded latent sample, not MC expectation.',
        caveats=['Diagnostic set 60000+ has been used before; not a fresh holdout',
                 'Inference intervention is not an ablation of retraining with consistent Actor inputs',
                 'Paired scenario bootstrap and exact McNemar p values are exploratory, not multiplicity corrected',
                 'Changing actions changes visited states; offline same-state disagreement is reported separately'],
        frozen_source_sha256=state['source_sha256'], script_sha256=digest(Path(__file__).resolve()))
    (args.outdir / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    @tf.function(reduce_retracing=True)
    def direct(vector):
        dist = agent.actor(vector)
        return dist.mode(), tf.tanh(dist.mean_tensor)[:, 0]

    @tf.function(reduce_retracing=True)
    def filtered(vector, action, latent):
        post, _ = agent.rssm.obs_step(latent, action, agent.encoder(dict(vector=vector)))
        predicted = agent.vector(agent.rssm.get_feat(post)).mean()
        return post, agent.actor(predicted).mode(), predicted

    result = dict(protocol=protocol, policy={}, paired_vs_baseline={}, actor_input={}, multistep={})
    trajectories = {}
    scenarios = None
    for mode in MODES:
        tf.random.set_seed(20260914)
        environments = [make_env(seed) for seed in range(args.seed_start, args.seed_start + args.episodes)]
        observations = [e.reset() for e in environments]
        layouts = [dict(start=e.env.state.position.tolist(), relay=e.env.relay_goal.tolist(),
                        final=e.env.final_goal.tolist(), heading=e.env.state.heading) for e in environments]
        if scenarios is None:
            scenarios = layouts
            (args.outdir / 'scenarios.json').write_text(json.dumps(layouts, indent=2), encoding='utf-8')
        else:
            assert layouts == scenarios
        n = len(environments)
        active = np.ones(n, bool)
        latent = agent.rssm.initial(n)
        prev_action = np.zeros((n, 4), np.float32)
        streak = np.zeros(n, int)
        records = [dict(scene=args.seed_start+i, success=False, pickup=False, timeout=False,
                        oob=False, length=0, interventions=0, stationary_steps=0,
                        move_steps=0, turn_steps=0, return_=0., pickup_step=None) for i in range(n)]
        saved = [dict(vector=[o['vector']], action=[np.zeros(4, np.float32)], reward=[0.], discount=[1.])
                 for o in observations]
        for step in range(100):
            vector = np.stack([o['vector'] for o in observations])
            actions, move_params = direct(vector)
            actions, move_params = actions.numpy(), move_params.numpy()
            if mode.startswith('reconstructed'):
                if mode.endswith('context20') and step % 20 == 0:
                    latent = agent.rssm.initial(n)
                latent, decoded_actions, _ = filtered(vector, prev_action, latent)
                actions = decoded_actions.numpy()
            for i in np.flatnonzero(active):
                env, record = environments[i], records[i]
                action = actions[i]
                if mode.startswith('unstall'):
                    action, streak[i], hit = unstall(action, move_params[i], env.env.state.speed,
                        streak[i], positive=mode == 'unstall_positive_move')
                    record['interventions'] += int(hit)
                elif mode == 'controller':
                    action = controller_action(env)
                record['stationary_steps'] += int(env.env.state.speed <= .1)
                record['move_steps' if np.argmax(action[:2]) == 0 else 'turn_steps'] += 1
                obs, reward, done, info = env.step(action)
                prev_action[i] = action
                observations[i] = obs
                record.update(length=step+1, success=bool(obs['is_success']),
                              timeout=bool(obs['truncated']), oob=bool(obs['out_of_bounds']))
                record['return_'] += reward
                if bool(obs['carrying_supply']) and not record['pickup']:
                    record['pickup'], record['pickup_step'] = True, step+1
                if mode in ('baseline', 'controller'):
                    for key, value in dict(vector=obs['vector'], action=action.copy(),
                                           reward=reward, discount=float(info['discount'])).items():
                        saved[i][key].append(value)
                active[i] = not done
            if not active.any():
                break
        for e in environments:
            e.close()
        summary = {field+'_rate': float(np.mean([r[field] for r in records]))
                   for field in ('success', 'pickup', 'timeout', 'oob')}
        summary.update(all_turn_episodes=sum(r['move_steps']==0 for r in records),
            mean_length=float(np.mean([r['length'] for r in records])),
            stationary_step_fraction=sum(r['stationary_steps'] for r in records)/sum(r['length'] for r in records),
            intervened_episodes=sum(r['interventions']>0 for r in records),
            intervention_count=sum(r['interventions'] for r in records))
        result['policy'][mode] = dict(summary=summary, records=records)
        if mode != 'baseline':
            result['paired_vs_baseline'][mode] = paired(result['policy']['baseline']['records'], records)
        else:
            stalled = {r['scene'] for r in records if r['move_steps']==0}
        if mode.startswith('unstall'):
            subset = [r for r in records if r['scene'] in stalled]
            result['policy'][mode]['baseline_allturn_subset'] = dict(n=len(subset),
                success=sum(r['success'] for r in subset), pickup=sum(r['pickup'] for r in subset),
                timeout=sum(r['timeout'] for r in subset), oob=sum(r['oob'] for r in subset))
        if mode in ('baseline', 'controller'):
            trajectories[mode] = [{k: np.asarray(v, np.float32) for k,v in ep.items()} for ep in saved]
        print(mode, json.dumps(summary), flush=True)
        (args.outdir / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

    @tf.function(reduce_retracing=True)
    def decode(latent):
        features = agent.rssm.get_feat(latent)
        return agent.vector(features).mean(), agent.reward(features).mean(), agent.discount(features).mean()

    @tf.function(reduce_retracing=True)
    def roll15(latent, actions):
        outputs = []
        for t in range(15):
            latent = agent.rssm.img_step(latent, actions[:, t])
            if t in (0,4,14):
                outputs.append(decode(latent))
        return outputs

    for kind, episodes in trajectories.items():
        packed = {key: np.stack([np.pad(ep[key], [(0, 101-len(ep[key]))]+[(0,0)]*(ep[key].ndim-1), mode='edge')
                                for ep in episodes]) for key in episodes[0]}
        np.savez_compressed(args.outdir / f'{kind}_trajectories.npz',
                            **packed, lengths=np.array([len(e['vector'])-1 for e in episodes]))
        for context in (0,20):
            tf.random.set_seed(20260914)
            latent = agent.rssm.initial(len(episodes))
            post, predictions = [], []
            for t in range(101):
                if context and t % context == 0:
                    latent = agent.rssm.initial(len(episodes))
                latent, _, predicted = filtered(packed['vector'][:,t], packed['action'][:,t], latent)
                post.append({k:v.numpy() for k,v in latent.items()})
                predictions.append(predicted.numpy())
            predictions = np.stack(predictions, axis=1)
            true_flat = np.concatenate([e['vector'][:-1] for e in episodes])
            predicted_flat = np.concatenate([predictions[i,:len(e['vector'])-1] for i,e in enumerate(episodes)])
            true_action = direct(true_flat)[0].numpy()
            predicted_action = direct(predicted_flat)[0].numpy()
            same = np.argmax(true_action[:,:2],1) == np.argmax(predicted_action[:,:2],1)
            input_key = f'{kind}_context{context}'
            result['actor_input'][input_key] = dict(n=len(same), branch_disagreement=float((~same).mean()),
                parameter_mae_same_branch=float(np.abs(true_action[same,2:]-predicted_action[same,2:]).sum(axis=1).mean()) if same.any() else None,
                state_normalized_rmse=float(np.sqrt(np.square(true_flat-predicted_flat).mean())),
                decoded_outside_normalized_range_fraction=float((np.abs(predicted_flat)>1).mean()))
            descriptions, starts, windows = [], [], []
            for i, ep in enumerate(episodes):
                length = len(ep['vector'])-1
                anchors = set(range(0,length,5)) | {length-1}
                anchors.update((np.flatnonzero(np.diff(ep['vector'][:,10])>0)).tolist())
                for t in sorted(anchors):
                    starts.append({k:v[i] for k,v in post[t].items()})
                    actions = ep['action'][t+1:t+16]
                    windows.append(np.pad(actions, [(0,15-len(actions)),(0,0)]))
                    descriptions.append((i,t))
            collected = {h:[] for h in (1,5,15)}
            for offset in range(0,len(starts),64):
                batch = starts[offset:offset+64]
                latent = {k: np.stack([r[k] for r in batch]) for k in batch[0]}
                outputs = roll15(latent,np.asarray(windows[offset:offset+64],np.float32))
                for h, output in zip((1,5,15),outputs):
                    values = [v.numpy() for v in output]
                    for j,(i,t) in enumerate(descriptions[offset:offset+64]):
                        ep = episodes[i]
                        if t+h>=len(ep['vector']):
                            continue
                        collected[h].append((values[0][j],ep['vector'][t+h],ep['vector'][t],
                            ep['reward'][t+h],ep['discount'][t+h],values[1][j],values[2][j],
                            ep['vector'][t,10]<=0 and ep['vector'][t+h,10]>0))
            result['multistep'][input_key] = {}
            for h, data in collected.items():
                arrays = [np.asarray(x) for x in zip(*data)]
                report = prediction_errors(*arrays[:7])
                pickup = arrays[7].astype(bool)
                report['pickup_crossing_count'] = int(pickup.sum())
                report['pickup_phase_recall'] = float((arrays[0][pickup,10]>0).mean()) if pickup.any() else None
                result['multistep'][input_key][str(h)] = report
            print('predictions',input_key,json.dumps(result['actor_input'][input_key]),flush=True)
            (args.outdir / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False),encoding='utf-8')
    replay_after = {str(p.relative_to(run)): (p.stat().st_size,p.stat().st_mtime_ns)
                    for p in (run / 'episodes').rglob('*.npz')}
    assert before == variable_hash()
    assert iterations == [int(o._opt.iterations.numpy()) for o in (agent.model_opt,agent.actor_opt,agent.critic_opt)]
    assert checkpoint_hash == digest(run/'variables.pkl')
    assert replay_before == replay_after
    assert all(digest(source/n)==h for n,h in state['source_sha256'].items())
    result['invariants'] = dict(checkpoint_unchanged=True, parameters_unchanged=True,
        optimizer_iterations_unchanged=True, replay_inventory_sizes_mtimes_unchanged=True,
        source_unchanged=True, scenarios_identical_all_modes=True, direct_actor_matches_original_policy=True)
    (args.outdir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('COMPLETE',str(args.outdir),flush=True)


if __name__ == '__main__':
    main()
