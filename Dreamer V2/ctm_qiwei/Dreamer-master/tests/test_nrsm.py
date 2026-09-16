"""Core contract tests; no environment training or legacy checkpoint writes."""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from nrsm import NRSM, NRSMConfig
from ctm_rl_core import SuperLinear


class NRSMTests(unittest.TestCase):
    def setUp(self):
        tf.keras.mixed_precision.set_global_policy('float32')
        tf.keras.utils.set_random_seed(41)
        self.c = NRSMConfig(neurons=8, memory=4, ticks=3, sync_neurons=4, context=10,
                            hidden=12, nlm_hidden=5, stoch=3, classes=4, embed=7)
        self.m = NRSM(self.c)
        self.action = tf.constant([[1., 0., .3, 0.], [0., 1., 0., -.4]])
        self.embed = tf.random.normal([2, 7])

    def assert_state_equal(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            np.testing.assert_allclose(a[k], b[k], rtol=2e-5, atol=2e-6, err_msg=k)

    def test_prior_cannot_see_current_observation(self):
        a, p = self.m.obs_step(self.m.initial(2), self.action, self.embed, seed=[4, 5])
        b, q = self.m.obs_step(self.m.initial(2), self.action, self.embed+1, seed=[4, 5])
        self.assert_state_equal(p, q)
        np.testing.assert_array_equal(a['deter'], b['deter'])
        np.testing.assert_array_equal(a['deter'], p['deter'])
        self.assertGreater(float(tf.norm(a['logits']-b['logits'])), .001)

    def test_private_neuron_weights(self):
        layer = SuperLinear(3, 2, 2, False)
        weights = np.zeros([3, 2, 2], np.float32)
        weights[0, 0] = [1., 2.]
        layer.w1.assign(weights)
        y = layer(tf.ones([1, 2, 3]))
        np.testing.assert_allclose(y, [[[1., 0.], [2., 0.]]], rtol=1e-6)

    def test_sync_matches_explicit_sum(self):
        core = self.m.ctm
        acts = tf.random.normal([2, self.c.neurons, self.c.memory])
        rates = np.linspace(-1., 5., core.output_size).astype(np.float32)
        core.decay_params_out.assign(rates)
        weight = np.exp(-np.arange(self.c.memory-1, -1, -1)[None]*np.clip(rates, 0, 4)[:, None])
        product = tf.gather(acts, core.left, axis=1)*tf.gather(acts, core.right, axis=1)
        expected = (product.numpy()*weight[None]).sum(-1)/np.sqrt(weight.sum(-1))[None]
        np.testing.assert_allclose(core.synchronise(acts), expected, rtol=2e-6, atol=2e-6)

    def test_chunk_carry_matches_full_observe_and_imagine(self):
        x, a = tf.random.normal([2, 5, 7]), tf.repeat(self.action[:, None], 5, axis=1)
        full, _ = self.m.observe(x, a, sample=False)
        prefix, _ = self.m.observe(x[:, :2], a[:, :2], sample=False)
        carry = {k: v[:, -1] for k, v in prefix.items()}
        suffix, _ = self.m.observe(x[:, 2:], a[:, 2:], carry, sample=False)
        self.assert_state_equal({k: v[:, 2:] for k, v in full.items()}, suffix)
        whole = self.m.imagine(a, carry, sample=False)
        first = self.m.imagine(a[:, :2], carry, sample=False)
        last = self.m.imagine(a[:, 2:], {k: v[:, -1] for k, v in first.items()}, sample=False)
        self.assert_state_equal({k: v[:, 2:] for k, v in whole.items()}, last)

    def test_padding_freezes_state_and_has_zero_gradient(self):
        x = tf.Variable(tf.random.normal([2, 4, 7]))
        a = tf.repeat(self.action[:, None], 4, axis=1)
        with tf.GradientTape() as tape:
            post, _ = self.m.observe(x, a, valid=tf.constant([[1, 1, 0, 0]]*2), sample=False)
            loss = tf.reduce_sum(self.m.get_feat(post))
        grad = tape.gradient(loss, x)
        np.testing.assert_array_equal(grad[:, 2:], 0)
        for v in post.values():
            np.testing.assert_array_equal(v[:, 1], v[:, 3])

    def test_reset_is_per_row_and_clears_every_trace(self):
        post, _ = self.m.obs_step(self.m.initial(2), self.action, self.embed, sample=False)
        reset = self.m.reset(post, [True, False])
        for k, v in reset.items():
            np.testing.assert_array_equal(v[0], self.m.initial(2)[k][0])
            np.testing.assert_array_equal(v[1], post[k][1])
        x = tf.stack([self.embed, self.embed], 1)
        a = tf.repeat(self.action[:, None], 2, 1)
        scanned, _ = self.m.observe(x, a, post, is_first=[[False, True]]*2, sample=False)
        expected, _ = self.m.obs_step(self.m.initial(2), self.action, self.embed, sample=False)
        self.assert_state_equal({k: v[:, -1] for k, v in scanned.items()}, expected)

    def test_all_parameter_groups_receive_finite_gradients(self):
        with tf.GradientTape() as tape:
            post, prior = self.m.obs_step(self.m.initial(2), self.action, self.embed, sample=False)
            kl, _ = self.m.kl_loss(post, prior)
            loss = kl + tf.reduce_mean(post['deter']**2) + tf.reduce_mean(prior['deter']**2)
        gradients = tape.gradient(loss, self.m.trainable_variables)
        for v, g in zip(self.m.trainable_variables, gradients):
            self.assertIsNotNone(g, v.name)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(g))), v.name)
        for variable in (self.m.ctm.nlm1.w1, self.m.ctm.synapse1.kernel, self.m.ctm.decay_params_out,
                         self.m.prior_logits.kernel, self.m.post_logits.kernel,
                         self.m.ctm.start_trace, self.m.ctm.start_activated_trace):
            i = next(i for i, v in enumerate(self.m.trainable_variables) if v is variable)
            self.assertGreater(float(tf.norm(gradients[i])), 0., variable.name)

    def test_imagination_action_gradient_and_compiled_forward(self):
        action = tf.Variable(self.action)
        with tf.GradientTape() as tape:
            state = self.m.img_step(self.m.initial(2), action, sample=False)
            loss = tf.reduce_sum(state['deter'][:, :3])
        grad = tape.gradient(loss, action)
        self.assertGreater(float(tf.norm(grad[:, 2:])), 0.)
        fn = tf.function(lambda a: self.m.img_step(self.m.initial(2), a, sample=False))
        self.assert_state_equal(state, fn(action))

    def test_checkpoint_roundtrip_and_architecture_rejection(self):
        state, _ = self.m.obs_step(self.m.initial(2), self.action, self.embed, sample=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'core.pkl'
            self.m.save_checkpoint(path, state)
            other = NRSM(self.c)
            restored = other.load_checkpoint(path)
            self.assert_state_equal(state, restored)
            self.assert_state_equal(self.m.img_step(state, self.action, seed=[3, 2]),
                                    other.img_step(restored, self.action, seed=[3, 2]))
            wrong = NRSM(replace(self.c, ticks=2))
            before = [v.numpy().copy() for v in wrong.variables]
            with self.assertRaises(ValueError):
                wrong.load_checkpoint(path)
            for a, b in zip(before, wrong.variables):
                np.testing.assert_array_equal(a, b)

    def test_persistent_pre_post_history_and_single_tick(self):
        previous = self.m.initial(2)
        changed = dict(previous, pre_trace=tf.ones_like(previous['pre_trace']))
        self.assertGreater(float(tf.norm(self.m.img_step(previous, self.action, False)['deter'] -
                                         self.m.img_step(changed, self.action, False)['deter'])), .001)
        changed = dict(previous, post_trace=tf.ones_like(previous['post_trace']))
        self.assertGreater(float(tf.norm(self.m.img_step(previous, self.action, False)['deter'] -
                                         self.m.img_step(changed, self.action, False)['deter'])), .001)
        one = NRSM(replace(self.c, ticks=1))
        for v in one.img_step(previous, self.action).values():
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(v))))

    def test_compiled_dynamic_length_scan(self):
        @tf.function(input_signature=[tf.TensorSpec([None, None, 7], tf.float32),
                                      tf.TensorSpec([None, None, 4], tf.float32)])
        def scan(x, a):
            return self.m.observe(x, a, sample=False)
        for length in (2, 6):
            x = tf.random.normal([2, length, 7])
            a = tf.repeat(self.action[:, None], length, 1)
            eager, _ = self.m.observe(x, a, sample=False)
            compiled, _ = scan(x, a)
            self.assert_state_equal(eager, compiled)

    def test_invalid_checkpoint_rejected_before_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'core.pkl'
            self.m.save_checkpoint(path, self.m.initial(2))
            with path.open('rb') as stream:
                original = pickle.load(stream)
            for corruption in ('weight_nan', 'carry_dtype', 'pair_index'):
                payload = pickle.loads(pickle.dumps(original))
                if corruption == 'weight_nan':
                    index = next(i for i, x in enumerate(payload['values']) if x.dtype == np.float32)
                    payload['values'][index].flat[0] = np.nan
                elif corruption == 'carry_dtype':
                    payload['state']['deter'] = payload['state']['deter'].astype(np.float64)
                else:
                    index = next(i for i, v in enumerate(self.m.variables) if v is self.m.ctm.left)
                    payload['values'][index][0] = self.c.neurons + 1
                with path.open('wb') as stream:
                    pickle.dump(payload, stream)
                before = [v.numpy().copy() for v in self.m.variables]
                with self.assertRaises(ValueError):
                    self.m.load_checkpoint(path)
                for a, b in zip(before, self.m.variables):
                    np.testing.assert_array_equal(a, b)

    def test_default_profile_forward_backward(self):
        model = NRSM()
        with tf.GradientTape() as tape:
            post, prior = model.obs_step(model.initial(1), self.action[:1], tf.zeros([1, 400]))
            loss, _ = model.kl_loss(post, prior)
            loss += tf.reduce_mean(post['deter']**2)
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and bool(tf.reduce_all(tf.math.is_finite(g))) for g in grads))

    def test_v1_checkpoint_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'v1.pkl'
            self.m.save_checkpoint(path)
            with path.open('rb') as f:
                payload = pickle.load(f)
            payload['contract'] = 'nrsm_core_v1_fp32_step_sync'
            with path.open('wb') as f:
                pickle.dump(payload, f)
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                self.m.load_checkpoint(path)


if __name__ == '__main__':
    unittest.main()
