"""One-variable CT-WM actor ablation: the unmodified trainer with a different imagination_scale.

train_nrsm_online.py builds OnlineAgent() with the default ACConfig (imagination_scale=0.1).
This wrapper replaces only that default and then runs the unmodified trainer with the remaining
arguments. The trainer records the resolved ac_config in the run's manifest.json; this wrapper also
writes <output>.ablation.json next to the run directory.

Usage (from "Dreamer V2/ctm_qiwei/Dreamer-master", with the TensorFlow venv):
    python -B -u <path>/run_ablation.py --imagination-scale 1.0 --output outputs/<new> --steps 137680 --seed 17
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

DV2 = Path(__file__).resolve().parents[2] / 'Dreamer V2' / 'ctm_qiwei' / 'Dreamer-master'
sys.path.insert(0, str(DV2))

import nrsm_online_agent  # noqa: E402
import train_nrsm_online  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--imagination-scale', type=float, required=True)
    known, trainer_args = parser.parse_known_args()
    default = nrsm_online_agent.ACConfig()
    config = nrsm_online_agent.ACConfig(imagination_scale=known.imagination_scale)
    base = nrsm_online_agent.OnlineAgent

    def agent_factory(core_config=None, ac_config=None):
        # Explicit configs (checkpoint-restore verification) pass through unchanged.
        return base(core_config, ac_config or config)

    train_nrsm_online.OnlineAgent = agent_factory
    output = Path(trainer_args[trainer_args.index('--output') + 1])
    sidecar = output.with_name(output.name + '.ablation.json')
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(dict(
        wrapper=Path(__file__).name,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changed={'ACConfig.imagination_scale': [default.imagination_scale, config.imagination_scale]},
        ac_config=asdict(config), trainer_args=trainer_args), indent=2), encoding='utf-8')
    sys.argv = [str(DV2 / 'train_nrsm_online.py')] + trainer_args
    train_nrsm_online.main()


if __name__ == '__main__':
    main()
