"""End-to-end AC boundary, gradient ownership, replay and restore regression."""
import pickle
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import tools
from nrsm import NRSMConfig
from nrsm_online_agent import ACConfig, OnlineAgent
from train_nrsm_online import add_transition, environment, new_episode, random_action, verify_restore, evaluate


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.mixed_precision.set_global_policy('float32')
        tf.keras.utils.set_random_seed(42)
        cls.agent = OnlineAgent(NRSMConfig(neurons=8, memory=4, ticks=2,
            sync_neurons=4, context=10, hidden=12, nlm_hidden=5, stoch=3, classes=4, embed=7),
            ACConfig(horizon=3, starts=4, units=12, layers=2, slow_update=2))
        cls.data = dict(vector=tf.ones([2, 5, 13])*.1,
            action=tf.constant([[[0., 0., 0., 0.]] + [[1., 0., .2, 0.]]*2 + [[0., 0., 0., 0.]]*2]*2),
            reward=tf.constant([[0., -.01, -1., 0., 0.]]*2),
            discount=tf.constant([[.99, .99, 0., 0., 0.]]*2),
            valid=tf.constant([[1., 1., 1., 0., 0.]]*2),
            is_first=tf.constant([[1., 0., 0., 0., 0.]]*2))

    def test_actor_uses_same_latent_as_online_policy(self):
        a = self.agent
        action, post = a.policy(tf.ones([2, 13])*.1, tf.zeros([2, 4]), a.world.core.initial(2), False)
        np.testing.assert_allclose(action, a.actor(a.world.core.get_feat(post)).mode(), atol=1e-7)
        self.assertEqual(action.shape, (2, 4))
        arr = action.numpy()
        np.testing.assert_array_equal(arr[:, :2].sum(-1), 1)
        np.testing.assert_array_equal(arr[:, 2:]*(1-arr[:, :2]), 0)

    def test_imagination_and_nonterminal_start_alignment(self):
        a = self.agent
        post, *_ = a.world.forward(self.data, False)
        starts, indices = a.select_starts(post, self.data)
        self.assertTrue(set(indices.numpy()).issubset({0, 1, 5, 6}))
        source, successor, actions = a.imagine(starts)
        self.assertEqual(source.shape[:2], (3, 4))
        np.testing.assert_array_equal(source[0], a.world.core.get_feat(starts))
        np.testing.assert_array_equal(source[1:], successor[:-1])
        np.testing.assert_array_equal(actions.numpy()[..., 2:]*(1-actions.numpy()[..., :2]), 0)

    def test_fixed_evaluation_is_readonly_reproducible_and_json_serializable(self):
        a = self.agent
        before = {k: [v.numpy().copy() for v in values] for k, values in a.groups().items()}
        first = evaluate(a, 78000, 2)
        second = evaluate(a, 78000, 2)
        self.assertEqual(first, second)
        self.assertEqual(first['count'], 2)
        json.dumps(first, allow_nan=False)
        for key, variables in a.groups().items():
            for v, old in zip(variables, before[key]):
                np.testing.assert_array_equal(v.numpy(), old)

    def test_successor_returns_do_not_shift_rewards(self):
        returns = tools.lambda_return_from_next_value(tf.constant([[1.], [2.], [3.]]),
            tf.constant([[10.], [20.], [30.]]), tf.constant([[.9], [.9], [0.]]), 1.)
        np.testing.assert_allclose(returns[:, 0], [5.23, 4.7, 3.], atol=1e-6)

    def test_continuous_actor_gradient_passes_through_ctm(self):
        a = self.agent
        post, *_ = a.world.forward(self.data, False)
        starts, _ = a.select_starts(post, self.data)
        with tf.GradientTape() as tape:
            _, _, _, _, metrics = a.actor_objective(starts)
            imagined_return = metrics['imagined_return']
        gradients = tape.gradient(imagined_return, a.actor.trainable_variables)
        connected = [g.numpy() for g in gradients if g is not None]
        self.assertTrue(connected)
        self.assertTrue(all(np.isfinite(g).all() for g in connected))
        self.assertGreater(sum(float(np.abs(g).sum()) for g in connected), 0.)

    def test_joint_update_changes_all_three_modules_and_target_schedule(self):
        a = self.agent
        before = {name: [v.numpy().copy() for v in getattr(a, name).trainable_variables]
                  for name in ('world', 'actor', 'critic')}
        slow_before = [v.numpy().copy() for v in a.slow_critic.variables]
        # Warmup must not update actor/critic or their optimizer iterations.
        a.update(self.data, True)
        self.assertEqual(int(a.actor_opt.iterations.numpy()), 0)
        for name in ('actor', 'critic'):
            for v, old in zip(getattr(a, name).trainable_variables, before[name]):
                np.testing.assert_array_equal(v.numpy(), old)
        metrics = a.update(self.data)
        self.assertTrue(all(np.isfinite(float(v)) for v in metrics.values()))
        for name, old in before.items():
            self.assertTrue(any(not np.array_equal(v.numpy(), b)
                                for v, b in zip(getattr(a, name).trainable_variables, old)), name)
        self.assertEqual(int(a.actor_opt.iterations.numpy()), 1)
        self.assertEqual(int(a.critic_opt.iterations.numpy()), 1)
        for v, old in zip(a.slow_critic.variables, slow_before):
            np.testing.assert_array_equal(v.numpy(), old)
        a.update(self.data)
        for v, slow in zip(a.critic.variables, a.slow_critic.variables):
            np.testing.assert_array_equal(v.numpy(), slow.numpy())

    def test_checkpoint_restores_optimizers_and_rejects_incompatible(self):
        a = self.agent
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'agent.pkl'
            a.save(checkpoint, dict(marker='full-ac'))
            self.assertEqual(verify_restore(a, checkpoint)['marker'], 'full-ac')
            with checkpoint.open('rb') as f:
                payload = pickle.load(f)
            payload['contract'] = 'offline-world-only'
            with checkpoint.open('wb') as f:
                pickle.dump(payload, f)
            with self.assertRaises(ValueError):
                a.load(checkpoint)

    def test_online_episode_transition_and_terminal_masks(self):
        env = environment(78123)
        rng = np.random.default_rng(17)
        try:
            obs = env.reset()
            episode = new_episode(obs)
            for t in range(1, 101):
                action = random_action(rng)
                obs, reward, done, info = env.step(action)
                add_transition(episode, t, action, obs, reward, info)
                if done:
                    break
            self.assertTrue(done)
            self.assertEqual(episode['discount'][t], 0)
            self.assertEqual(episode['valid'].sum(), t+1)
            self.assertEqual(episode['is_first'].sum(), 1)
            np.testing.assert_array_equal(episode['action'][0], 0)
            np.testing.assert_array_equal(episode['action'][t+1:], 0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
