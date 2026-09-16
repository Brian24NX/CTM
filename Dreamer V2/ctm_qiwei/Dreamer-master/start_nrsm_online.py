"""Freeze sources and launch one authorized bounded online NRSM+AC run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from run_support import atomic_write
from nrsm_run_budget import RunBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=None)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--final-eval-seed-start', type=int, default=71000)
    args = parser.parse_args()
    try:
        budget = RunBudget.from_args(args.seconds, args.steps)
    except ValueError as error:
        parser.error(str(error))
    if args.resume:
        args.resume = args.resume.resolve(strict=True)
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    source = output.with_name(output.name + '_source')
    receipt, log = output.with_suffix('.launch.json'), output.with_suffix('.launch.log')
    if any(p.exists() for p in (output, source, receipt, log)):
        raise FileExistsError('Unique output, source, receipt and log required')
    source.mkdir(parents=True)
    # Keep launch TF-free. Imports in the frozen trainer are all from this copy.
    names = ('train_nrsm_online.py', 'nrsm_online_agent.py', 'nrsm.py', 'ctm_rl_core.py',
             'validate_nrsm.py', 'models.py', 'tools.py', 'envs.py', 'uav_actions.py',
             'controller_baseline.py', 'run_support.py', 'run_wsl.sh', 'gpu_check.py', 'nrsm_run_budget.py')
    for name in names:
        content = (root/name).read_bytes()
        atomic_write(source/name, lambda f, data=content: f.write(data))
    env = dict(os.environ, CTM_REQUIRE_GPU='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
               PYTHONUNBUFFERED='1', PYTHONPATH=str(source))
    command = ['bash', str(source/'run_wsl.sh'), str(source/'train_nrsm_online.py'),
               '--output', str(output), '--final-eval-seed-start', str(args.final_eval_seed_start)]
    if budget.seconds:
        command = ['timeout', '--signal=TERM', '--kill-after=90', str(budget.seconds+180)] + command
        command += ['--duration-seconds', str(budget.seconds)]
    else:
        # No inherited one-hour timeout. Trainer stops at cumulative step target.
        command += ['--steps', str(budget.steps)]
    if args.resume:
        command += ['--resume', str(args.resume)]
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(supervisor_pid=child.pid, started_at=datetime.now(timezone.utc).isoformat(),
                  output=str(output), frozen_source=str(source), log=str(log), command=command,
                  target_env_steps=budget.steps or None, duration_seconds=budget.seconds or None)
    atomic_write(receipt, lambda f: f.write(json.dumps(record, indent=2).encode()))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
