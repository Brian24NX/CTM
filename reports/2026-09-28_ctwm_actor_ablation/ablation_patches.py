"""Runtime patches for CT-WM ablations. The team's code is never edited.

learned_reward_std (experiment 2): the world model's reward head predicts its own standard deviation
(std = min_std + softplus(raw)) instead of using a fixed unit-variance Gaussian. The loss is still the
reward negative log-likelihood; the actor still consumes reward.mean().

delta_weight (experiment 3): an auxiliary head predicts the one-step change of the motion coordinates
(x, y, speed, sin/cos heading) from the same posterior features as the other heads. Targets are divided
by fixed per-dimension scales (DELTA_SCALE, ~7.5 m for x/y), so a few metres of error matter as much as
a large error on the absolute vector does. The loss adds delta_weight * (unit-variance Gaussian NLL) to
the team's unchanged world-model loss. Absolute reconstruction, all other heads and the actor/critic are
unchanged.

reward_norm (experiment 4): the reward head keeps its network (same layers, same initialisation, so seed
pairing holds) but predicts symlog((r - REWARD_MU) / REWARD_SIGMA) with a unit-variance Gaussian. The
constants are the mean/std of non-terminal rewards in the baseline's random prefill, so ordinary steps
become O(1) targets, while symlog keeps the rare +/-1 terminal rewards representable (~+/-6.6) without
dominating. mean() returns the reward in original units, so the team's world loss (log_prob) and actor
objective (mean) run unchanged.

Training (run_ablation.py) and analysis scripts build agents through agent_for_run()/apply(), so a
checkpoint is always loaded into the architecture that produced it.
"""
import json
import sys
from pathlib import Path

DV2 = Path(__file__).resolve().parents[2] / 'Dreamer V2' / 'ctm_qiwei' / 'Dreamer-master'
if str(DV2) not in sys.path:
    sys.path.insert(0, str(DV2))

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers as tfkl  # noqa: E402
from tensorflow_probability import distributions as tfd  # noqa: E402

import models  # noqa: E402
import nrsm_online_agent  # noqa: E402
import tools  # noqa: E402
import validate_nrsm  # noqa: E402

REWARD_STD_KEY = 'world_model.reward_head_min_std'
REWARD_NORM_KEY = 'world_model.reward_head_normalised'
# Mean/std of the 1,018 non-terminal rewards in the same 12 random-prefill episodes as DELTA_SCALE.
REWARD_MU, REWARD_SIGMA = -0.007229, 0.001387
DELTA_KEY = 'world_model.delta_head_weight'
DELTA_DIMS = (0, 1, 2, 3, 4)   # x, y, speed, sin(heading), cos(heading) of the 13-vector
# Std of one-step changes in the 12 random-prefill episodes (1,030 transitions) of the 2026-09-28
# baseline run, seed 17. Fixed constants, so the target scale never drifts during training.
DELTA_SCALE = (0.00746, 0.00729, 0.07891, 0.28910, 0.27888)
DELTA_CLIP = 10.0
_ORIGINAL_WORLD_LOSS = nrsm_online_agent.OnlineAgent.world_loss


class LearnedStdHead(tools.Module):
    """Scalar Gaussian head that predicts a per-state standard deviation."""

    def __init__(self, layers=2, units=400, min_std=0.01, act=tf.nn.elu):
        super().__init__()
        self._layers, self._units, self._min_std, self._act = layers, units, float(min_std), act

    def __call__(self, features):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        mean, raw_std = tf.unstack(self.get('out', tfkl.Dense, 2)(x), axis=-1)
        return tfd.Independent(tfd.Normal(mean, self._min_std + tf.nn.softplus(raw_std)), 0)


class MotionHead(tools.Module):
    """models.DenseHead equivalent (ELU layers, linear output, unit-variance Gaussian) whose weights use
    fixed-seed initialisers. Building it draws nothing from the shared random stream, so every other
    initial weight (world model, actor, critic) stays identical to the paired run's."""

    def __init__(self, size, layers=2, units=400, act=tf.nn.elu, seed=20260928):
        super().__init__()
        self._size, self._layers, self._units, self._act, self._seed = size, layers, units, act, seed

    def __call__(self, features):
        glorot = lambda offset: tf.keras.initializers.GlorotUniform(seed=self._seed + offset)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, kernel_initializer=glorot(index))(x)
        x = self.get('out', tfkl.Dense, self._size, kernel_initializer=glorot(99))(x)
        return tfd.Independent(tfd.Normal(x, 1), 1)


def symlog(x):
    return tf.sign(x) * tf.math.log1p(tf.abs(x))


def symexp(x):
    return tf.sign(x) * tf.math.expm1(tf.abs(x))


class NormalisedReward:
    """Minimal distribution interface used by the team's code: log_prob() for the world loss, mean() for
    imagination and evaluation. The Gaussian lives in the transformed space."""

    def __init__(self, prediction):
        self.prediction = prediction

    def log_prob(self, reward):
        target = symlog((tf.cast(reward, tf.float32) - REWARD_MU) / REWARD_SIGMA)
        return tfd.Normal(self.prediction, 1.).log_prob(target)

    def mean(self):
        return REWARD_MU + REWARD_SIGMA * symexp(self.prediction)

    mode = mean


class NormalisedRewardHead(tools.Module):
    """The team's reward network (models.DenseHead, identical layers and initialisation) with a normalised,
    symlog-transformed target."""

    def __init__(self, units, layers=2):
        super().__init__()
        self.net = models.DenseHead((), layers=layers, units=units)

    def __call__(self, features):
        return NormalisedReward(self.net(features).mean())


def delta_targets(vector):
    """Normalised one-step changes [B, T-1, 5] of the motion coordinates."""
    dims = list(DELTA_DIMS)
    change = tf.gather(vector[:, 1:], dims, axis=-1) - tf.gather(vector[:, :-1], dims, axis=-1)
    return tf.clip_by_value(change / tf.constant(DELTA_SCALE, tf.float32), -DELTA_CLIP, DELTA_CLIP)


def delta_mask(data):
    """Rows t >= 1 whose row t is a valid, non-first transition and whose row t-1 is valid."""
    return data['valid'][:, 1:] * (1. - data['is_first'][:, 1:]) * data['valid'][:, :-1]


def world_model_class(learned_reward_std=None, delta_weight=None, reward_norm=False):
    base = validate_nrsm.DiagnosticWorldModel
    if learned_reward_std and reward_norm:
        raise ValueError('learned_reward_std and reward_norm both replace the reward head')
    if not learned_reward_std and not delta_weight and not reward_norm:
        return base

    class AblationWorldModel(base):
        def __init__(self, config):
            super().__init__(config)
            if learned_reward_std:
                # Replacing the attribute keeps its position, so variable order stays deterministic.
                self.reward = LearnedStdHead(layers=2, units=config.hidden, min_std=learned_reward_std)
            if reward_norm:
                self.reward = NormalisedRewardHead(units=config.hidden)
            if delta_weight:
                self.delta_weight = float(delta_weight)
                self.delta = MotionHead(len(DELTA_DIMS), layers=2, units=config.hidden)
                # Create its variables now, before OnlineAgent builds the optimizer (seeded: no RNG shift).
                self.delta(tf.zeros([1, config.stoch * config.classes + config.context]))

        def delta_nll(self, feat, data):
            return tools.masked_mean(-self.delta(feat[:, 1:]).log_prob(delta_targets(data['vector'])),
                                     delta_mask(data))

    return AblationWorldModel


def _world_loss_with_delta(self, data):
    """The team's world_loss, plus delta_weight * the delta-head NLL on the same posterior features."""
    post, loss, metrics = _ORIGINAL_WORLD_LOSS(self, data)
    delta_nll = self.world.delta_nll(self.world.core.get_feat(post), data)
    loss = loss + self.world.delta_weight * delta_nll
    return post, loss, dict(metrics, model_loss=loss, delta_nll=delta_nll)


def apply(learned_reward_std=None, delta_weight=None, reward_norm=False):
    """Every OnlineAgent built after this call uses the selected world model and loss."""
    nrsm_online_agent.DiagnosticWorldModel = world_model_class(learned_reward_std, delta_weight, reward_norm)
    nrsm_online_agent.OnlineAgent.world_loss = (
        _world_loss_with_delta if delta_weight else _ORIGINAL_WORLD_LOSS)


def run_changes(run_dir):
    sidecar = Path(run_dir).with_name(Path(run_dir).name + '.ablation.json')
    return json.loads(sidecar.read_text()).get('changed', {}) if sidecar.exists() else {}


def agent_for_run(run_dir):
    """Rebuild a run's exact agent (architecture, loss and ac_config) from its manifest and sidecar."""
    manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
    changes = run_changes(run_dir)
    apply(changes.get(REWARD_STD_KEY, [None, None])[1], changes.get(DELTA_KEY, [None, None])[1],
          bool(changes.get(REWARD_NORM_KEY, [False, False])[1]))
    return nrsm_online_agent.OnlineAgent(ac_config=nrsm_online_agent.ACConfig(**manifest['ac_config']))
