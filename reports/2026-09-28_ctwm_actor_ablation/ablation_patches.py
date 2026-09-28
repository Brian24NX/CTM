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


def delta_targets(vector):
    """Normalised one-step changes [B, T-1, 5] of the motion coordinates."""
    dims = list(DELTA_DIMS)
    change = tf.gather(vector[:, 1:], dims, axis=-1) - tf.gather(vector[:, :-1], dims, axis=-1)
    return tf.clip_by_value(change / tf.constant(DELTA_SCALE, tf.float32), -DELTA_CLIP, DELTA_CLIP)


def delta_mask(data):
    """Rows t >= 1 whose row t is a valid, non-first transition and whose row t-1 is valid."""
    return data['valid'][:, 1:] * (1. - data['is_first'][:, 1:]) * data['valid'][:, :-1]


def world_model_class(learned_reward_std=None, delta_weight=None):
    base = validate_nrsm.DiagnosticWorldModel
    if not learned_reward_std and not delta_weight:
        return base

    class AblationWorldModel(base):
        def __init__(self, config):
            super().__init__(config)
            if learned_reward_std:
                # Replacing the attribute keeps its position, so variable order stays deterministic.
                self.reward = LearnedStdHead(layers=2, units=config.hidden, min_std=learned_reward_std)
            if delta_weight:
                self.delta_weight = float(delta_weight)
                self.delta = models.DenseHead((len(DELTA_DIMS),), layers=2, units=config.hidden)
                # Create its variables now, before OnlineAgent builds the optimizer.
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


def apply(learned_reward_std=None, delta_weight=None):
    """Every OnlineAgent built after this call uses the selected world model and loss."""
    nrsm_online_agent.DiagnosticWorldModel = world_model_class(learned_reward_std, delta_weight)
    nrsm_online_agent.OnlineAgent.world_loss = (
        _world_loss_with_delta if delta_weight else _ORIGINAL_WORLD_LOSS)


def run_changes(run_dir):
    sidecar = Path(run_dir).with_name(Path(run_dir).name + '.ablation.json')
    return json.loads(sidecar.read_text()).get('changed', {}) if sidecar.exists() else {}


def agent_for_run(run_dir):
    """Rebuild a run's exact agent (architecture, loss and ac_config) from its manifest and sidecar."""
    manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
    changes = run_changes(run_dir)
    apply(changes.get(REWARD_STD_KEY, [None, None])[1], changes.get(DELTA_KEY, [None, None])[1])
    return nrsm_online_agent.OnlineAgent(ac_config=nrsm_online_agent.ACConfig(**manifest['ac_config']))
