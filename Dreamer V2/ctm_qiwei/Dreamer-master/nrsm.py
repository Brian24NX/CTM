"""NRSM v2: pinned official CTM RL core plus a separate Dreamer-style adapter.

Unlike the archived v1 prototype, this uses full learned pre/post traces, GLU
NLM/synapses, windowed sync, and shared prior/posterior deterministic context.
Old NRSM v1 checkpoints are intentionally incompatible. Legacy Dreamer unchanged.
"""
from dataclasses import asdict, dataclass
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

import models
import tools
from ctm_rl_core import CTMRLCore, UPSTREAM_REVISION, dense, norm
from run_support import atomic_write


@dataclass(frozen=True)
class NRSMConfig:
    neurons: int = 256
    memory: int = 16
    ticks: int = 4
    sync_neurons: int = 32
    context: int = 400
    hidden: int = 400
    nlm_hidden: int = 16
    stoch: int = 32
    classes: int = 32
    embed: int = 400
    deep_nlms: bool = True
    nlm_norm: bool = True
    synapse_depth: int = 1

    def __post_init__(self):
        sizes = ('neurons', 'memory', 'ticks', 'sync_neurons', 'context', 'hidden',
                 'nlm_hidden', 'stoch', 'classes', 'embed', 'synapse_depth')
        if any(type(getattr(self, k)) is not int or getattr(self, k) < 1 for k in sizes):
            raise ValueError('NRSM dimensions must be positive integers')
        if self.classes < 2 or self.sync_neurons > self.neurons:
            raise ValueError('Invalid categorical size or sync neuron count')
        if type(self.deep_nlms) is not bool or type(self.nlm_norm) is not bool:
            raise ValueError('NLM switches must be booleans')


class NRSM(tools.Module):
    contract = 'nrsm_v2_ctm_rl_4a6c9c3_shared_context_fp32'

    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or NRSMConfig()
        if tf.keras.mixed_precision.global_policy().compute_dtype != 'float32':
            raise ValueError('NRSM v2 requires the shared FP32 policy')
        self.ctm = CTMRLCore(c.context+c.stoch*c.classes+4, c.neurons, c.memory, c.ticks,
                             c.hidden, c.sync_neurons, c.nlm_hidden, c.deep_nlms,
                             c.nlm_norm, c.synapse_depth)
        # Adapter projections/distributions are NOT claimed to be official CTM.
        self.project = dense(self.ctm.output_size, c.context)
        self.norm = norm(c.context)
        self.prior_hidden = dense(c.context, c.hidden)
        self.prior_logits = dense(c.hidden, c.stoch*c.classes)
        self.post_hidden = dense(c.context+c.embed, c.hidden)
        self.post_logits = dense(c.hidden, c.stoch*c.classes)

    def initial(self, batch_size):
        c = self.config
        pre, post = self.ctm.initial(batch_size)
        return dict(deter=tf.zeros([batch_size, c.context]),
                    stoch=tf.zeros([batch_size, c.stoch, c.classes]),
                    logits=tf.zeros([batch_size, c.stoch, c.classes]),
                    pre_trace=pre, post_trace=post)

    def reset(self, state, is_first):
        initial = self.initial(tf.shape(state['deter'])[0])
        return {k: tf.where(tf.reshape(tf.cast(is_first, tf.bool), [-1]+[1]*(v.shape.rank-1)),
                            initial[k], v) for k, v in state.items()}

    def transition(self, previous, action):
        c = self.config
        action = models.canonical_action(tf.cast(action, tf.float32))
        flat_z = tf.reshape(previous['stoch'], [-1, c.stoch*c.classes])
        raw = tf.concat([previous['deter'], flat_z, action], -1)
        sync, pre, post, _, _ = self.ctm(raw, previous['pre_trace'], previous['post_trace'])
        return self.norm(self.project(sync)), pre, post

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
        context, pre, post = self.transition(previous, action)
        stats = self._stats(self.prior_logits(tf.nn.elu(self.prior_hidden(context))), sample, seed)
        return dict(**stats, deter=context, pre_trace=pre, post_trace=post)

    def obs_step(self, previous, action, embed, sample=True, seed=None):
        prior = self.img_step(previous, action, sample, seed)
        post_seed = None if seed is None else tf.random.experimental.stateless_fold_in(seed, 1)
        stats = self._stats(self.post_logits(tf.nn.elu(self.post_hidden(tf.concat([prior['deter'], embed], -1)))), sample, post_seed)
        # Shared deterministic state: current observation changes ONLY posterior z.
        return dict(**stats, deter=prior['deter'], pre_trace=prior['pre_trace'], post_trace=prior['post_trace']), prior

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
        payload = dict(contract=self.contract, upstream_revision=UPSTREAM_REVISION,
                       config=asdict(self.config), values=values, state=saved_state)
        atomic_write(Path(path), lambda f: pickle.dump(payload, f))

    def load_checkpoint(self, path):
        # Pickle is allowed only for our own trusted local artifacts.
        with Path(path).open('rb') as f:
            payload = pickle.load(f)
        if (payload.get('contract') != self.contract or payload.get('config') != asdict(self.config)
                or payload.get('upstream_revision') != UPSTREAM_REVISION):
            raise ValueError('NRSM architecture contract mismatch')
        values, variables = payload['values'], self.variables
        if len(values) != len(variables) or any(tuple(v.shape) != x.shape or
                                               np.dtype(tf.as_dtype(v.dtype).as_numpy_dtype) != x.dtype or not np.isfinite(x).all()
                                               for v, x in zip(variables, values)):
            raise ValueError('Invalid NRSM weights')
        for variable, value in zip(variables, values):
            if (variable is self.ctm.left or variable is self.ctm.right) and not np.array_equal(variable.numpy(), value):
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
