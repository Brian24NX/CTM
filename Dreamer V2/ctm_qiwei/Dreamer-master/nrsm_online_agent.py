"""NRSM world model + latent-input hybrid Actor/Critic, isolated from DreamerV2.

No demonstrations or direct-observation actor shortcut. Reuses the validated
CTM core, canonical hybrid distribution, and successor-aligned lambda returns.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

import models
import tools
from nrsm import NRSMConfig
from run_support import atomic_write, require_free_space
from validate_nrsm import DiagnosticWorldModel


@dataclass(frozen=True)
class ACConfig:
    horizon: int = 15
    starts: int = 32
    units: int = 128
    layers: int = 2
    discount: float = .99
    lambda_: float = .95
    model_lr: float = 3e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    clip: float = 100.
    imagination_scale: float = .1
    parameter_grad_scale: float = .1
    discrete_entropy: float = .03
    parameter_entropy: float = .02
    slow_update: int = 100

    def __post_init__(self):
        if min(self.horizon, self.starts, self.units, self.layers, self.slow_update) < 1:
            raise ValueError('Positive AC dimensions and target interval required')
        if not 0 < self.discount <= 1 or not 0 <= self.lambda_ <= 1:
            raise ValueError('Invalid return discounts')


def compact_config():
    return NRSMConfig(neurons=64, memory=8, ticks=2, sync_neurons=16,
                      context=128, hidden=128, nlm_hidden=8, stoch=8, classes=8, embed=128)


class OnlineAgent:
    contract = 'nrsm_online_ac_v1_latent_input_no_bc'

    def __init__(self, core_config=None, ac_config=None):
        self.core_config = core_config or compact_config()
        self.c = ac_config or ACConfig()
        self.world = DiagnosticWorldModel(self.core_config)
        self.actor = models.HybridActionDecoder(2, layers=self.c.layers, units=self.c.units,
                                                 min_std=.1, unimix=.05)
        self.critic = models.DenseHead((), layers=self.c.layers, units=self.c.units)
        self.slow_critic = models.DenseHead((), layers=self.c.layers, units=self.c.units)
        dummy = dict(vector=tf.zeros([1, 2, 13]), action=tf.zeros([1, 2, 4]),
                     valid=tf.ones([1, 2]), is_first=tf.constant([[1., 0.]]))
        post, *_ = self.world.forward(dummy, False)
        feat = self.world.core.get_feat(post)
        self.actor(feat); self.critic(feat); self.slow_critic(feat)
        self.model_opt = tf.keras.optimizers.Adam(self.c.model_lr)
        self.actor_opt = tf.keras.optimizers.Adam(self.c.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(self.c.critic_lr)
        for optimizer, module in ((self.model_opt, self.world), (self.actor_opt, self.actor),
                                  (self.critic_opt, self.critic)):
            optimizer.build(module.trainable_variables)
        self.ac_updates = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.copy_target()

    def copy_target(self):
        for dest, src in zip(self.slow_critic.variables, self.critic.variables):
            dest.assign(src)

    def groups(self):
        return dict(world=self.world.variables, actor=self.actor.variables,
                    critic=self.critic.variables, slow_critic=self.slow_critic.variables,
                    model_optimizer=list(self.model_opt.variables),
                    actor_optimizer=list(self.actor_opt.variables),
                    critic_optimizer=list(self.critic_opt.variables), counter=[self.ac_updates])

    @tf.function(reduce_retracing=True)
    def policy(self, vector, previous_action, state, sample):
        embed = self.world.encoder(dict(vector=vector))
        post, _ = self.world.core.obs_step(state, previous_action, embed, sample=sample)
        dist = self.actor(self.world.core.get_feat(post))
        return (dist.sample() if sample else dist.mode()), post

    def world_loss(self, data):
        post, prior, vector, reward, discount = self.world.forward(data, True)
        valid = data['valid']
        transition = valid * (1. - data['is_first'])
        kl, raw_kl = self.world.core.kl_loss(post, prior, balance=.8, free=0., valid=valid)
        vector_nll = tools.masked_mean(-vector.log_prob(data['vector']), valid)
        reward_nll = tools.masked_mean(-reward.log_prob(data['reward']), transition)
        discount_bce = tools.masked_mean(-discount.log_prob(data['discount']), transition)
        loss = 10 * vector_nll + reward_nll + discount_bce + kl
        return post, loss, dict(model_loss=loss, vector_nll=vector_nll,
                               reward_nll=reward_nll, discount_bce=discount_bce, kl=raw_kl)

    def select_starts(self, post, data):
        # Terminal observations and padding must never generate imagined actions.
        eligible = tf.reshape((data['valid'] > 0) & (data['discount'] > 0), [-1])
        candidates = tf.cast(tf.where(eligible)[:, 0], tf.int32)
        tf.debugging.assert_positive(tf.size(candidates), 'No nonterminal replay states')
        draw = tf.random.uniform([self.c.starts], maxval=tf.size(candidates), dtype=tf.int32)
        indices = tf.gather(candidates, draw)
        state = {k: tf.stop_gradient(tf.gather(tf.reshape(v, [-1] + list(v.shape[2:])), indices))
                 for k, v in post.items()}
        return state, indices

    def imagine(self, state):
        sources, successors, actions = [], [], []
        for _ in range(self.c.horizon):
            feat = self.world.core.get_feat(state)
            # Same latent feature contract as online policy. Stop state->policy
            # derivatives; continuous action->dynamics derivatives remain live.
            action = self.actor(tf.stop_gradient(feat)).sample()
            state = self.world.core.img_step(state, action)
            sources.append(feat)
            successors.append(self.world.core.get_feat(state))
            actions.append(action)
        return tf.stack(sources), tf.stack(successors), tf.stack(actions)

    def actor_objective(self, starts):
        source, successor, action = self.imagine(starts)
        reward = self.world.reward(successor).mean()
        discount = self.world.discount(successor).mean()  # Already gamma * continuation.
        value = self.slow_critic(successor).mean()
        returns = tools.lambda_return_from_next_value(reward, value, discount, self.c.lambda_)
        weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(discount[:1]), discount[:-1]], axis=0), axis=0))
        dist = self.actor(tf.stop_gradient(source))
        baseline = self.critic(source).mean()
        advantage = tf.stop_gradient(returns - baseline)
        score = tools.masked_mean(dist.discrete_log_prob(action) * advantage, weights)
        dynamics = tools.masked_mean(returns, weights)
        entropy = tools.masked_mean(self.c.discrete_entropy * dist.discrete_entropy() +
                                   self.c.parameter_entropy * dist.parameter_entropy(), weights)
        loss = -self.c.imagination_scale * (score + self.c.parameter_grad_scale*dynamics) - entropy
        return loss, source, returns, weights, dict(actor_loss=loss, actor_score=score,
            imagined_return=dynamics, actor_entropy_bonus=entropy,
            imagined_discount=tf.reduce_mean(discount))

    def checked_gradients(self, tape, loss, variables, name):
        grads = tape.gradient(loss, variables)
        checks = [tf.debugging.assert_all_finite(loss, name + ' loss')]
        for var, grad in zip(variables, grads):
            if grad is None:
                raise ValueError(name + ' disconnected gradient: ' + var.name)
            checks.append(tf.debugging.assert_all_finite(grad, name + ' gradient'))
        norm = tf.linalg.global_norm(grads)
        checks.append(tf.debugging.assert_all_finite(norm, name + ' gradient norm'))
        with tf.control_dependencies(checks):
            clipped, _ = tf.clip_by_global_norm(grads, self.c.clip, norm)
            return [tf.identity(g) for g in clipped], tf.identity(norm)

    @tf.function(reduce_retracing=True)
    def update(self, data, world_only=False):
        with tf.GradientTape() as tape:
            post, loss, metrics = self.world_loss(data)
        wg, wn = self.checked_gradients(tape, loss, self.world.trainable_variables, 'world')
        if world_only:
            self.model_opt.apply_gradients(zip(wg, self.world.trainable_variables))
        else:
            starts, _ = self.select_starts(post, data)
            with tf.GradientTape() as actor_tape:
                actor_loss, source, returns, weights, actor_metrics = self.actor_objective(starts)
            ag, an = self.checked_gradients(actor_tape, actor_loss, self.actor.trainable_variables, 'actor')
            with tf.GradientTape() as critic_tape:
                critic_loss = -tools.masked_mean(self.critic(tf.stop_gradient(source)).log_prob(
                    tf.stop_gradient(returns)), weights)
            cg, cn = self.checked_gradients(critic_tape, critic_loss, self.critic.trainable_variables, 'critic')
            # Validate all three before any assignment. Gradients update only
            # their owning module, although actor derivatives pass through WM.
            with tf.control_dependencies(wg + ag + cg):
                self.model_opt.apply_gradients(zip(wg, self.world.trainable_variables))
                self.actor_opt.apply_gradients(zip(ag, self.actor.trainable_variables))
                self.critic_opt.apply_gradients(zip(cg, self.critic.trainable_variables))
                self.ac_updates.assign_add(1)
            if tf.equal(self.ac_updates % self.c.slow_update, 0):
                self.copy_target()
            metrics.update(actor_metrics, critic_loss=critic_loss,
                           actor_gradient_norm=an, critic_gradient_norm=cn)
        for group in (self.world.variables, self.actor.variables, self.critic.variables,
                      self.slow_critic.variables):
            for var in group:
                if tf.as_dtype(var.dtype).is_floating:
                    tf.debugging.assert_all_finite(var, 'Non-finite parameter after update')
        metrics['model_gradient_norm'] = wn
        return metrics

    def save(self, path, training_state):
        groups = {key: [v.numpy() for v in group] for key, group in self.groups().items()}
        if not all(np.isfinite(v).all() for values in groups.values() for v in values):
            raise ValueError('Refusing nonfinite checkpoint')
        require_free_space(Path(path).parent, 256 * 1024**2)
        payload = dict(contract=self.contract, core_contract=self.world.core.contract,
                       core_config=asdict(self.core_config), ac_config=asdict(self.c),
                       groups=groups, training_state=training_state,
                       resume_scope='Models, three optimizers, target, replay, counters and numpy RNG; '
                                    'restart at new episode; TF random stream not bit-exact')
        atomic_write(path, lambda stream: pickle.dump(payload, stream))

    def load(self, path):
        # Trusted artifacts generated by this project only.
        with Path(path).open('rb') as stream:
            payload = pickle.load(stream)
        if (payload.get('contract') != self.contract or
            payload.get('core_contract') != self.world.core.contract or
            payload.get('core_config') != asdict(self.core_config) or
            payload.get('ac_config') != asdict(self.c)):
            raise ValueError('Incompatible full-agent checkpoint')
        groups = self.groups()
        if payload['groups'].keys() != groups.keys():
            raise ValueError('Checkpoint group mismatch')
        for name, variables in groups.items():
            values = payload['groups'][name]
            if len(values) != len(variables):
                raise ValueError('Checkpoint variable count mismatch')
            for v, x in zip(variables, values):
                if (tuple(v.shape) != x.shape or np.dtype(tf.as_dtype(v.dtype).as_numpy_dtype) != x.dtype
                    or not np.isfinite(x).all()):
                    raise ValueError('Checkpoint variable mismatch')
                if v is self.world.core.ctm.left or v is self.world.core.ctm.right:
                    if not np.array_equal(v.numpy(), x):
                        raise ValueError('Fixed CTM indices mismatch')
        for name, variables in groups.items():
            for v, x in zip(variables, payload['groups'][name]):
                v.assign(x)
        return payload['training_state']
