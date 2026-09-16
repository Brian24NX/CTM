"""Check offline data alignment and the compiled full-sequence training path."""
import unittest

import numpy as np
import tensorflow as tf

from nrsm import NRSMConfig
from validate_nrsm import DiagnosticWorldModel, collect


class NRSMValidationTests(unittest.TestCase):
    def test_collection_is_reproducible_and_episode_aligned(self):
        first, records = collect(2, 26000)
        second, again = collect(2, 26000)
        self.assertEqual(records, again)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for row, record in enumerate(records):
            last = record['length']
            self.assertEqual(first['valid'][row].sum(), last+1)
            np.testing.assert_array_equal(first['valid'][row, :last+1], 1)
            np.testing.assert_array_equal(first['action'][row, 0], 0)
            self.assertEqual(first['is_first'][row].sum(), 1)
            self.assertEqual(first['discount'][row, last], 0)
            np.testing.assert_array_equal(first['action'][row, last+1:], 0)
            action = first['action'][row, 1:last+1]
            np.testing.assert_array_equal(action[:, :2].sum(-1), 1)
            np.testing.assert_array_equal(action[:, 2:] * (1-action[:, :2]), 0)

    def test_compiled_objective_ignores_padding_and_backpropagates(self):
        tf.keras.mixed_precision.set_global_policy('float32')
        model = DiagnosticWorldModel(NRSMConfig(neurons=8, memory=4, ticks=2,
                sync_neurons=4, context=10, hidden=12, nlm_hidden=5, stoch=3, classes=4, embed=7))
        batch = dict(vector=tf.ones([2, 5, 13])*.1, action=tf.zeros([2, 5, 4]),
                     reward=tf.zeros([2, 5]), discount=tf.ones([2, 5])*.99,
                     valid=tf.constant([[1., 1., 1., 0., 0.]]*2),
                     is_first=tf.constant([[1., 0., 0., 0., 0.]]*2))
        model.forward(batch, False)
        @tf.function
        def measure(data):
            with tf.GradientTape() as tape:
                tape.watch(data['vector'])
                loss, _ = model.objective(data)
            return loss, tape.gradient(loss, [data['vector']] + model.trainable_variables)
        loss, gradients = measure(batch)
        self.assertTrue(bool(tf.math.is_finite(loss)))
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(gradient))))
        np.testing.assert_array_equal(gradients[0][:, 3:], 0)


if __name__ == '__main__':
    unittest.main()
