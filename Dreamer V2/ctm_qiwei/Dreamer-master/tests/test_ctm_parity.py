"""Compare TF port to fixtures executed by the unmodified PyTorch upstream.

No PyTorch dependency in the Dreamer runtime. Regeneration uses the isolated
exporter. Matching actual weights (not merely RNG seeds) is mandatory.
"""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from ctm_rl_core import CTMRLCore, UPSTREAM_REVISION, REFERENCE_TORCH_VERSION


class CTMParityTests(unittest.TestCase):
    def test_official_forward_backward_and_cross_step_state(self):
        tf.keras.mixed_precision.set_global_policy('float32')
        fixture_root = Path(__file__).with_name('fixtures')/'ctm_reference'
        upstream = Path(__file__).resolve().parents[1]/'third_party'/'ctm_reference'
        self.assertEqual(len(list(fixture_root.glob('*.npz'))), 4)
        for fixture in sorted(fixture_root.glob('*.npz')):
            with self.subTest(fixture=fixture.name), np.load(fixture, allow_pickle=False) as f, tf.device('/CPU:0'):
                self.assertEqual(str(f['upstream_sha']), UPSTREAM_REVISION)
                self.assertEqual(str(f['torch_version']), REFERENCE_TORCH_VERSION)
                for name, digest in json.loads(str(f['source_sha256_json'])).items():
                    self.assertEqual(hashlib.sha256((upstream/name).read_bytes()).hexdigest(), digest, name)
                cfg = {k.removeprefix('config__'): f[k].item() for k in f.files if k.startswith('config__')}
                core = CTMRLCore(int(f['input_size']), cfg['d_model'], cfg['memory_length'], cfg['iterations'],
                    cfg['d_input'], cfg['n_synch_out'], cfg['memory_hidden_dims'], cfg['deep_nlms'],
                    cfg['do_layernorm_nlm'], cfg['synapse_depth'])
                mapping = core.torch_named_variables()
                self.assertEqual(set(mapping), set(f['parameter_names']))
                self.assertEqual(len(mapping), len(core.trainable_variables))
                for name, (variable, transpose) in mapping.items():
                    value = f['weight__'+name]
                    variable.assign(value.T if transpose else value)
                max_forward, max_gradient = 0., 0.
                def compare(actual, expected, name, gradient=False):
                    nonlocal max_forward, max_gradient
                    actual = np.asarray(actual)
                    error = float(np.max(np.abs(actual-expected)))
                    if gradient:
                        max_gradient = max(max_gradient, error)
                    else:
                        max_forward = max(max_forward, error)
                    # Recurrent FP32 reverse-mode reductions differ across
                    # frameworks; allow small cancellation error near zero.
                    # Forward checks stay strict; clamp boundaries separately
                    # require the actual reference's zero/nonzero routing.
                    np.testing.assert_allclose(actual, expected,
                        rtol=5e-4 if gradient else 3e-5, atol=5e-5 if gradient else 3e-6,
                        err_msg=fixture.name+':'+name)
                raw = tf.Variable(f['inputs'])
                with tf.GradientTape() as tape:
                    pre, post = core.initial(raw.shape[1])
                    compare(pre, f['initial_pre'], 'initial_pre')
                    compare(post, f['initial_post'], 'initial_post')
                    result = []
                    for step in range(raw.shape[0]):
                        previous_post = post
                        sync, pre, post, pt, at = core(raw[step], pre, post)
                        result.append(sync)
                        compare(core.backbone(raw[step]), f['backbone'][step], 'backbone')
                        compare(pt, f['pre_ticks'][step].transpose(1, 0, 2), 'pre_ticks')
                        compare(at, f['post_ticks'][step].transpose(1, 0, 2), 'post_ticks')
                        compare(pre, f['pre_states'][step], 'pre_state')
                        compare(post, f['post_states'][step], 'post_state')
                        tick_post = previous_post
                        for tick in range(core.ticks):
                            tick_post = tf.concat([tick_post[:, :, 1:], at[:, tick, :, None]], -1)
                            compare(core.synchronise(tick_post), f['sync_ticks'][step, tick], 'sync_tick')
                    result = tf.stack(result)
                    loss = tf.reduce_sum(result*f['loss_weights']) + .03*tf.reduce_sum(pre**2) + .07*tf.reduce_sum(post**2)
                compare(result, f['result'], 'result')
                compare(loss, f['loss'], 'loss')
                gradients = tape.gradient(loss, [raw] + [v for v, _ in mapping.values()])
                compare(gradients[0], f['input_grad'], 'input_gradient', True)
                for (name, (_, transpose)), gradient in zip(mapping.items(), gradients[1:]):
                    self.assertIsNotNone(gradient, name)
                    compare(gradient.numpy().T if transpose else gradient.numpy(), f['grad__'+name], name, True)
                    if name == 'decay_params_out':
                        np.testing.assert_array_equal(gradient.numpy()[[0, 8]], 0.)
                        for index in (1, 7):
                            self.assertNotEqual(float(gradient[index]), 0.)
                            np.testing.assert_allclose(gradient[index], f['grad__'+name][index], rtol=5e-4, atol=1e-7)
                print('CTM_PARITY '+json.dumps(dict(case=fixture.stem, parameters=len(mapping),
                      max_forward_abs_error=max_forward, max_gradient_abs_error=max_gradient)), flush=True)


if __name__ == '__main__':
    unittest.main()
