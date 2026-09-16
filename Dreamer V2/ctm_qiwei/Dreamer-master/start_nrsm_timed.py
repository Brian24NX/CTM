"""Launch the authorized one-hour validation detached, with an outer time limit.

Does not resume checkpoints or launch an Actor. Uses a new unique run directory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 3600:
        parser.error('Duration must be 1..3600 seconds')
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix('.launch.log')
    receipt = output.with_suffix('.launch.json')
    if receipt.exists():
        raise FileExistsError(receipt)
    env = dict(os.environ, CTM_REQUIRE_GPU='1', TF_FORCE_GPU_ALLOW_GROWTH='true', PYTHONUNBUFFERED='1')
    # Inner loop ends its training at the deadline. Outer timeout also bounds a
    # stuck update/startup: TERM after budget+120s, then KILL after another 90s.
    command = ['timeout', '--signal=TERM', '--kill-after=90', str(args.seconds+120),
               'bash', str(root/'run_wsl.sh'), str(root/'validate_nrsm.py'),
               '--output', str(output), '--duration-seconds', str(args.seconds),
               '--train-episodes', '16', '--eval-episodes', '8', '--batch', '2', '--seed', '17']
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(supervisor_pid=child.pid, started_at=datetime.now(timezone.utc).isoformat(),
                  output=str(output), log=str(log), command=command, duration_seconds=args.seconds)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
