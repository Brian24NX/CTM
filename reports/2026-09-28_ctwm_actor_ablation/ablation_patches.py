"""Runtime patches for CT-WM ablations. The team's code is never edited.

learned_reward_std: the world model's reward head predicts its own standard deviation
(std = min_std + softplus(raw)) instead of using a fixed unit-variance Gaussian. The loss is still
the reward negative log-likelihood; all other heads, losses and the actor/critic are unchanged, and
the actor still consumes reward.mean(). A per-state std lets the model be precise on ordinary steps
(tiny shaping rewards) while staying uncertain on rare terminal steps (+/-1).

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

import nrsm_online_agent  # noqa: E402
import tools  # noqa: E402
import validate_nrsm  # noqa: E402

REWARD_STD_KEY = 'world_model.reward_head_min_std'


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


def world_model_class(learned_reward_std=None):
    if not learned_reward_std:
        return validate_nrsm.DiagnosticWorldModel

    class LearnedRewardStdWorldModel(validate_nrsm.DiagnosticWorldModel):
        def __init__(self, config):
            super().__init__(config)
            # Replacing the attribute keeps its position, so variable order stays deterministic.
            self.reward = LearnedStdHead(layers=2, units=config.hidden, min_std=learned_reward_std)

    return LearnedRewardStdWorldModel


def apply(learned_reward_std=None):
    """Every OnlineAgent built after this call uses the selected world model."""
    nrsm_online_agent.DiagnosticWorldModel = world_model_class(learned_reward_std)


def run_changes(run_dir):
    sidecar = Path(run_dir).with_name(Path(run_dir).name + '.ablation.json')
    return json.loads(sidecar.read_text()).get('changed', {}) if sidecar.exists() else {}


def agent_for_run(run_dir):
    """Rebuild a run's exact agent (architecture and ac_config) from its manifest and sidecar."""
    manifest = json.loads((Path(run_dir) / 'manifest.json').read_text())
    apply(run_changes(run_dir).get(REWARD_STD_KEY, [None, None])[1])
    return nrsm_online_agent.OnlineAgent(ac_config=nrsm_online_agent.ACConfig(**manifest['ac_config']))
