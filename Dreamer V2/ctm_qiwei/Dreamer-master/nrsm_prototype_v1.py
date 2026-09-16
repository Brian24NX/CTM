"""NRSM core v1. Independent of the legacy Dreamer training entry point.

Persistent pre-activation traces are an explicit extension of the manuscript.
Synchronization accumulators reset at each environment transition (Eq. 9).
No task heads, Actor, auxiliary tick loss, or global mutable episode state here.
All numerical state is FP32. Actions have the existing four-channel contract.
"""
from dataclasses import asdict, dataclass
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

import models
import tools
from run_support import atomic_write


@dataclass(frozen=True)
class NRSMConfig:
    neurons: int = 256
    memory: int = 16
    ticks: int = 4
    pairs: int = 512
    context: int = 400
    hidden: int = 400
    nlm_hidden: int = 16
    stoch: int = 32
    classes: int = 32
    embed: int = 400
    pair_seed: int = 0
    history_mode: str = 'persistent'

    def __post_init__(self):
        sizes = ('neurons', 'memory', 'ticks', 'pairs', 'context', 'hidden',
                 'nlm_hidden', 'stoch', 'classes', 'embed')
        if any(not isinstance(getattr(self, k), int) or getattr(self, k) < 1 for k in sizes):
            raise ValueError('NRSM dimensions must be positive integers')
        if self.classes < 2 or self.pairs > self.neurons * (self.neurons + 1) // 2:
            raise ValueError('Invalid categorical size or pair count')
        if self.history_mode not in ('persistent', 'context_only'):
            raise ValueError('Unknown history mode')


class NeuronTemporalModel(tools.Module):
    """Private M -> H -> 1 MLP for every neuron, vectorized, no shared weights."""
    def __init__(self, neurons, memory, hidden):
        super().__init__()
        self.w1 = tf.Variable(tf.random.normal([neurons, memory, hidden], stddev=memory**-.5), name='nlm_w1')
        self.b1 = tf.Variable(tf.zeros([neurons, hidden]), name='nlm_b1')
        self.w2 = tf.Variable(tf.random.normal([neurons, hidden], stddev=hidden**-.5), name='nlm_w2')
        self.b2 = tf.Variable(tf.zeros([neurons]), name='nlm_b2')

    def __call__(self, history):
        hidden = tf.nn.elu(tf.einsum('bdm,dmh->bdh', history, self.w1) + self.b1)
        return tf.tanh(tf.einsum('bdh,dh->bd', hidden, self.w2) + self.b2)


def synchronize_step(numerator, mass, activation, left, right, rate):
    """Eq. 9 recurrence. S is not a bounded correlation coefficient."""
    decay = tf.exp(-rate)
    product = tf.gather(activation, left, axis=-1) * tf.gather(activation, right, axis=-1)
    numerator = decay * numerator + product
    mass = decay * mass + 1.
    return numerator, mass, numerator / tf.sqrt(tf.maximum(mass, 1e-8))


class NRSM(tools.Module):
    contract = 'nrsm_core_v1_fp32_step_sync'

    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or NRSMConfig()
        if tf.keras.mixed_precision.global_policy().compute_dtype != 'float32':
            raise ValueError('NRSM v1 requires the shared FP32 policy')
        # Fixed unique unordered pairs; indices are model constants, NOT episode state.
        left, right = np.triu_indices(c.neurons)
        chosen = np.random.default_rng(c.pair_seed).choice(len(left), c.pairs, replace=False)
        self.left = tf.Variable(left[chosen].astype(np.int32), trainable=False, name='pair_left')
        self.right = tf.Variable(right[chosen].astype(np.int32), trainable=False, name='pair_right')
        self.raw_rate = tf.Variable(tf.fill([c.pairs], -2.), name='sync_raw_rate')
        self.nlm = NeuronTemporalModel(c.neurons, c.memory, c.nlm_hidden)

        def dense(name, ins, outs, activation=None):
            layer = tf.keras.layers.Dense(outs, activation=activation, name=name, dtype='float32')
            layer.build((None, ins))
            return layer

        self.condition = dense('condition', c.context + c.stoch*c.classes + 4, c.hidden, 'elu')
        self.synapse1 = dense('synapse1', c.neurons + c.hidden, c.hidden, 'elu')
        self.synapse2 = dense('synapse2', c.hidden, c.neurons)
        self.project = dense('sync_project', c.pairs, c.context)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype='float32')
        self.norm.build((None, c.context))
        self.prior_hidden = dense('prior_hidden', c.context, c.hidden, 'elu')
        self.prior_logits = dense('prior_logits', c.hidden, c.stoch*c.classes)
        self.post_hidden = dense('post_hidden', c.context + c.embed, c.hidden, 'elu')
        self.post_logits = dense('post_logits', c.hidden, c.stoch*c.classes)
        # Explicit posterior fusion, not part of the prior; must audit prior/head gap.
        self.fusion = dense('posterior_fusion', c.context+c.stoch*c.classes+c.embed, c.context, 'tanh')
        self.init_trace = None
        self.init_activation = None
        if c.history_mode == 'context_only':
            self.init_trace = dense('init_trace', c.context, c.neurons*c.memory, 'tanh')
            self.init_activation = dense('init_activation', c.context, c.neurons, 'tanh')

    def initial(self, batch_size):
        c = self.config
        return dict(deter=tf.zeros([batch_size, c.context]),
                    stoch=tf.zeros([batch_size, c.stoch, c.classes]),
                    logits=tf.zeros([batch_size, c.stoch, c.classes]),
                    pre_trace=tf.zeros([batch_size, c.neurons, c.memory]),
                    activation=tf.zeros([batch_size, c.neurons]))

    def reset(self, state, is_first):
        return {k: tf.where(tf.reshape(tf.cast(is_first, tf.bool), [-1]+[1]*(v.shape.rank-1)),
                            tf.zeros_like(v), v) for k, v in state.items()}

    def transition(self, previous, action):
        c = self.config
        action = models.canonical_action(tf.cast(action, tf.float32))
        flat_z = tf.reshape(previous['stoch'], [-1, c.stoch*c.classes])
        condition = self.condition(tf.concat([previous['deter'], flat_z, action], -1))
        trace, activation = previous['pre_trace'], previous['activation']
        if c.history_mode == 'context_only':
            trace = tf.reshape(self.init_trace(previous['deter']), [-1, c.neurons, c.memory])
            activation = self.init_activation(previous['deter'])
        numerator = tf.zeros([tf.shape(action)[0], c.pairs])
        mass = tf.zeros_like(numerator)
        rate = tf.nn.softplus(self.raw_rate)
        contexts = []
        for _ in range(c.ticks):
            pre = self.synapse2(self.synapse1(tf.concat([activation, condition], -1)))
            trace = tf.concat([trace[:, :, 1:], pre[:, :, None]], -1)
            activation = self.nlm(trace)
            numerator, mass, sync = synchronize_step(numerator, mass, activation, self.left, self.right, rate)
            contexts.append(self.norm(self.project(sync)))
        return contexts[-1], trace, activation, tf.stack(contexts, axis=1)

    def _stats(self, logits, sample, seed):
        c = self.config
        logits = tf.reshape(logits, [-1, c.stoch, c.classes])
        flat = tf.reshape(logits, [-1, c.classes])
        if sample:
            index = (tf.random.categorical(flat, 1) if seed is None else
                     tf.random.stateless_categorical(flat, 1, seed))
            index = tf.reshape(index, [-1, c.stoch])
        else:
            index = tf.argmax(logits, -1)
        hard = tf.one_hot(index, c.classes, dtype=tf.float32)
        probs = tf.nn.softmax(logits, -1)
        return dict(logits=logits, stoch=hard + probs - tf.stop_gradient(probs))

    def img_step(self, previous, action, sample=True, seed=None):
        context, trace, activation, _ = self.transition(previous, action)
        stats = self._stats(self.prior_logits(self.prior_hidden(context)), sample, seed)
        return dict(**stats, deter=context, pre_trace=trace, activation=activation)

    def obs_step(self, previous, action, embed, sample=True, seed=None):
        prior = self.img_step(previous, action, sample, seed)
        post_seed = None if seed is None else tf.random.experimental.stateless_fold_in(seed, 1)
        stats = self._stats(self.post_logits(self.post_hidden(tf.concat([prior['deter'], embed], -1))), sample, post_seed)
        z = tf.reshape(stats['stoch'], [-1, self.config.stoch*self.config.classes])
        context = self.fusion(tf.concat([prior['deter'], z, embed], -1))
        return dict(**stats, deter=context, pre_trace=prior['pre_trace'], activation=prior['activation']), prior

    def observe(self, embed, action, state=None, is_first=None, valid=None, sample=True):
        """Episode-aligned scan. Padding freezes state; reset is per batch row."""
        batch = tf.shape(action)[0]
        tf.debugging.assert_positive(tf.shape(action)[1])
        state = self.initial(batch) if state is None else state
        first = tf.zeros(tf.shape(action)[:2], tf.bool) if is_first is None else tf.cast(is_first, tf.bool)
        valid = tf.ones(tf.shape(action)[:2], tf.bool) if valid is None else tf.cast(valid, tf.bool)
        def step(carry, inputs):
            previous_state = carry[0]
            x, a, reset, mask_valid = inputs
            previous = self.reset(previous_state, reset)
            post, prior = self.obs_step(previous, a, x, sample)
            def mask(candidate):
                return {k: tf.where(tf.reshape(mask_valid, [-1]+[1]*(v.shape.rank-1)), v, previous_state[k])
                        for k, v in candidate.items()}
            return mask(post), mask(prior)
        inputs = (tf.transpose(embed, [1, 0, 2]), tf.transpose(action, [1, 0, 2]),
                  tf.transpose(first), tf.transpose(valid))
        result = tf.scan(step, inputs, initializer=(state, state), parallel_iterations=1)
        return tuple({k: tf.transpose(v, [1, 0]+list(range(2, v.shape.rank))) for k, v in rows.items()}
                     for rows in result)

    def imagine(self, action, state=None, sample=True):
        state = self.initial(tf.shape(action)[0]) if state is None else state
        tf.debugging.assert_positive(tf.shape(action)[1])
        result = tf.scan(lambda prev, a: self.img_step(prev, a, sample),
                         tf.transpose(action, [1, 0, 2]), initializer=state, parallel_iterations=1)
        return {k: tf.transpose(v, [1, 0]+list(range(2, v.shape.rank))) for k, v in result.items()}

    get_feat = models.RSSM.get_feat
    get_dist = models.RSSM.get_dist
    kl_loss = models.RSSM.kl_loss

    def save_checkpoint(self, path, state=None):
        """Core-only trusted local snapshot; not a trainer/optimizer checkpoint."""
        values = [v.numpy() for v in self.variables]
        saved_state = None if state is None else {k: v.numpy() for k, v in state.items()}
        if not all(np.isfinite(v).all() for v in values + list((saved_state or {}).values())):
            raise ValueError('Non-finite core checkpoint refused')
        payload = dict(contract=self.contract, config=asdict(self.config), values=values, state=saved_state)
        atomic_write(Path(path), lambda f: pickle.dump(payload, f))

    def load_checkpoint(self, path):
        # Pickle is allowed only for our own trusted local artifacts.
        with Path(path).open('rb') as f:
            payload = pickle.load(f)
        if payload.get('contract') != self.contract or payload.get('config') != asdict(self.config):
            raise ValueError('NRSM architecture contract mismatch')
        values, variables = payload['values'], self.variables
        if len(values) != len(variables) or any(tuple(v.shape) != x.shape or
                                               np.dtype(tf.as_dtype(v.dtype).as_numpy_dtype) != x.dtype or not np.isfinite(x).all()
                                               for v, x in zip(variables, values)):
            raise ValueError('Invalid NRSM weights')
        for variable, value in zip(variables, values):
            if (variable is self.left or variable is self.right) and not np.array_equal(variable.numpy(), value):
                raise ValueError('Fixed synchronization pair indices do not match config')
        state = payload.get('state')
        if state is not None:
            if not isinstance(state, dict) or 'deter' not in state or state['deter'].ndim != 2:
                raise ValueError('Invalid NRSM carry state')
            expected = self.initial(len(state['deter']))
            if state.keys() != expected.keys() or any(state[k].shape != tuple(v.shape) or
                                                     state[k].dtype != np.float32 or
                                                     not np.isfinite(state[k]).all() for k, v in expected.items()):
                raise ValueError('Invalid NRSM carry state')
        # No assignments before ALL validation succeeds.
        for variable, value in zip(variables, values):
            variable.assign(value)
        return None if state is None else {k: tf.convert_to_tensor(v) for k, v in state.items()}

