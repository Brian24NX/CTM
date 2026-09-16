"""Read-only log snapshot plots for a resumed online NRSM+AC run; no TensorFlow."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np


def moving_average(values, window):
    values = np.asarray(values, float)
    return np.array([values[max(0, i-window+1):i+1].mean() for i in range(len(values))])


def read_snapshot(path):
    raw = path.read_bytes()
    # Ignore only an unfinished last line while the live writer is appending.
    if raw and not raw.endswith(b'\n'):
        raw = raw[:raw.rfind(b'\n')+1]
    return raw, [json.loads(line) for line in raw.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (args.initial, args.current)]
    for key in ('contract', 'core_config', 'ac_config'):
        assert manifests[0][key] == manifests[1][key], key
    for key in ('nrsm_online_agent.py', 'nrsm.py', 'ctm_rl_core.py', 'models.py', 'envs.py'):
        assert manifests[0]['source_sha256'][key] == manifests[1]['source_sha256'][key], key
    resume = Path(manifests[1]['args']['resume'])
    assert resume.parent.resolve() == args.initial.resolve()
    assert hashlib.sha256(resume.read_bytes()).hexdigest() == manifests[1]['resume_checkpoint_sha256']
    boundary = manifests[1]['initial_counts']['env_steps']
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    for index, run in enumerate((args.initial, args.current)):
        raw, records = read_snapshot(run/'metrics.jsonl')
        (args.output/f'metrics_snapshot_{index}.jsonl').write_bytes(raw)
        if index == 0:
            records = [r for r in records if r['env_steps'] <= boundary]
        else:
            assert all(r['env_steps'] >= boundary for r in records)
        rows.extend(records)
        sources.append(dict(run=str(run), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    train = {r['ac_updates']: r for r in rows if r['kind'] == 'train' and
             r.get('phase') == 'joint_ac' and 'actor_loss' in r['metrics']}
    episodes = {r['metrics']['scene']: r for r in rows if r['kind'] == 'episode' and r.get('phase') == 'policy'}
    evaluation = {r['env_steps']: r for r in rows if r['kind'] == 'evaluation' and
                  r.get('split') in ('fixed', 'fixed_final') and r['metrics'].get('count') == 20}
    train = sorted(train.values(), key=lambda r: r['env_steps'])
    episodes = sorted(episodes.values(), key=lambda r: r['env_steps'])
    evaluation = sorted(evaluation.values(), key=lambda r: r['env_steps'])
    for row in evaluation:
        assert [r['scene'] for r in row['metrics']['episodes']] == list(range(70000, 70020))
    last_step = max(r['env_steps'] for r in rows)
    captured = datetime.now().astimezone().isoformat(timespec='seconds')
    font = '/mnt/c/Windows/Fonts/msyh.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
        'axes.unicode_minus': False, 'font.size': 10, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    blue, orange, gray = '#2665ad', '#d36827', '#687581'

    def frame(title, shape, size):
        fig, axes = plt.subplots(*shape, figsize=size)
        fig.subplots_adjust(left=.072, right=.975, top=.85, bottom=.15, hspace=.46, wspace=.29)
        fig.suptitle(title, x=.072, y=.976, fontsize=19, ha='left')
        fig.text(.072, .922, f'Seed 17 · 截至 {last_step:,} / 300,000 环境步 · 已合并前一小时与续跑日志 · 非最终结果', color=gray, fontsize=10)
        for ax in axes.flat:
            ax.set_xlabel('累计环境步数（千步）')
            ax.set_xlim(0, last_step/1000*1.025)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.axvline(boundary/1000, color=gray, ls=':', lw=1, alpha=.55)
            ax.grid(alpha=.18, lw=.6)
        return fig, axes

    fig, axes = frame('NRSM + Actor-Critic｜当前训练指标', (3, 3), (15.4, 11.1))
    specs = [('model_loss', '世界模型总损失', '损失'),
             ('vector_nll', '状态重建损失', 'Gaussian NLL'),
             ('kl', '后验与先验 KL', 'KL'),
             ('actor_loss', 'Actor 总损失', '损失'),
             ('critic_loss', 'Critic 损失', 'Gaussian NLL'),
             ('imagined_return', '模型内想象回报', '估计回报（非真实配送回报）'),
             ('model_gradient_norm', '世界模型梯度范数', '裁剪前范数（对数轴）'),
             ('actor_gradient_norm', 'Actor 梯度范数', '裁剪前范数（对数轴）'),
             ('critic_gradient_norm', 'Critic 梯度范数', '裁剪前范数（对数轴）')]
    xs = np.array([r['env_steps'] for r in train])/1000
    for ax, (key, title, label) in zip(axes.flat, specs):
        ys = np.array([r['metrics'][key] for r in train])
        assert np.isfinite(ys).all()
        ax.set_title(title, loc='left')
        ax.set_ylabel(label)
        if 'gradient' in key:
            assert (ys > 0).all()
            ax.plot(xs, ys, color=blue, lw=.9, alpha=.85)
            ax.set_yscale('log')
            if key == 'model_gradient_norm':
                ax.axhline(100, color=orange, ls='--', lw=1.2, label='裁剪阈值 100')
                ax.legend(frameon=False, fontsize=9)
        else:
            ax.plot(xs, ys, color=blue, alpha=.2, lw=.7)
            ax.plot(xs, moving_average(ys, 15), color=blue, lw=2)
    fig.text(.072, .054, '上两行：淡线为原始日志值，粗线为最近15点单边滑动平均；下行梯度不平滑。竖虚线为8,400步checkpoint续跑位置。', fontsize=9, color=gray)
    fig.text(.072, .030, '只画联合AC训练记录，不混入预热、离线预测实验或短测；Loss下降不等于配送成功。梯度仅为日志采样值，不是每次更新峰值。', fontsize=9, color=gray)
    fig.savefig(args.output/'nrsm_online_learning_curves.png', dpi=170)

    fig2, axes = frame('NRSM + Actor-Critic｜当前任务表现', (2, 2), (12.6, 8.8))
    for ax, key, title in zip(axes.flat, ('success', 'pickup', 'timeout', 'oob'),
                              ('配送成功率 ↑', '取货率 ↑', '超时率 ↓', '越界率 ↓')):
        ax.set_title(title, loc='left')
        ax.set_ylabel('回合比例（%）')
        ax.set_ylim(-3, 103)
        tx = np.array([r['env_steps'] for r in episodes])/1000
        ty = np.array([r['metrics'][key] for r in episodes], float)*100
        ex = np.array([r['env_steps'] for r in evaluation])/1000
        ey = np.array([r['metrics'][key] for r in evaluation])*100
        ax.plot(tx, moving_average(ty, 100), color=blue, lw=2, label='训练：最近100局均值')
        ax.plot(ex, ey, color=orange, lw=.8, marker='o', ms=2.5, alpha=.25)
        ax.plot(ex, moving_average(ey, 5), color=orange, lw=2, ls='--', label='固定评估：最近5次均值')
        ax.legend(frameon=False, fontsize=9, loc='center right')
    fig2.text(.072, .057, '训练使用随机策略采样；评估使用确定性策略、相同20场景。橙色淡线/点为每次评估原值；不含随机预填回合。', fontsize=9, color=gray)
    fig2.text(.072, .030, '单边滑动平均，起始窗口不足按已有记录计算；平滑有滞后，不能把跨checkpoint均值视为单个checkpoint的测试成绩。', fontsize=9, color=gray)
    fig2.savefig(args.output/'nrsm_online_task_curves.png', dpi=170)
    with PdfPages(args.output/'nrsm_online_current_curves.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    plt.close('all')
    audit = dict(captured_at=captured, last_env_step=last_step, resume_step=boundary,
                 joint_ac_records=len(train), policy_episodes=len(episodes),
                 fixed_evaluations=len(evaluation), source_snapshots=sources,
                 smoothing=dict(losses_ma=15, train_episode_ma=100, eval_checkpoint_ma=5, gradients='raw'),
                 note='Static snapshot of running job. No TF/GPU, training changes, or extrapolation.')
    (args.output/'provenance.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
