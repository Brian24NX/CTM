"""Comparison figures (light + dark PNGs): baseline CT-WM run vs imagination_scale=1.0 runs.

Reads the baseline's small files from ../2026-09-28_first_run_mac/ and the ablation's from this folder.
Usage (any Python with numpy + matplotlib):  python make_figures.py
Colour follows the run: baseline = slot 1, seed 17 = slot 2, seed 18 = slot 3 (validated palette).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '2026-09-28_first_run_mac'
DPI, PX = 144, 72 / 144
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', axis='#c3c2b7', series=['#2a78d6', '#eb6834', '#1baf7a']),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781',
                 grid='#2c2c2a', axis='#383835', series=['#3987e5', '#d95926', '#199e70']),
}
RUNS = [('Baseline (scale 0.1, seed 17)', BASE, 'ctwm_online_1h'),
        ('Fix (scale 1.0, seed 17)', HERE, 'imagscale1_seed17'),
        ('Fix (scale 1.0, seed 18)', HERE, 'imagscale1_seed18')]
WINDOW = 100


def rows(folder, prefix, name):
    with (folder / f'{prefix}_{name}.csv').open() as stream:
        return list(csv.DictReader(stream))


def available():
    return [(label, folder, prefix, color) for (label, folder, prefix), color
            in zip(RUNS, range(3)) if (folder / f'{prefix}_episodes.csv').exists()]


def rolling(values, window=WINDOW):
    values = np.asarray(values, float)
    out = np.convolve(values, np.ones(window) / window, mode='valid')
    return out


def style(ax, t, grid_axis='y'):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
        ax.spines[side].set_linewidth(1 * PX)
    ax.tick_params(colors=t['muted'], labelcolor=t['muted'], width=1 * PX, length=3)
    ax.grid(axis=grid_axis, color=t['grid'], linewidth=1 * PX, linestyle='-')
    ax.set_axisbelow(True)


def chart(t, title, subtitle, xlabel, ylabel=None, log=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=DPI)
    fig.patch.set_facecolor(t['surface'])
    style(ax, t)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.72, bottom=0.14)
    fig.text(0.012, 0.965, title, color=t['ink'], fontsize=11.5, fontweight='bold', va='top')
    fig.text(0.012, 0.905, subtitle, color=t['ink2'], fontsize=8.5, va='top')
    ax.set_xlabel(xlabel, color=t['ink2'], fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=t['ink2'], fontsize=8.5)
    if log:
        ax.set_yscale('log')
    return fig, ax


def line(ax, t, x, y, slot):
    handle, = ax.plot(x, y, color=t['series'][slot], linewidth=2 * PX, solid_joinstyle='round',
                      solid_capstyle='round', zorder=3)
    return handle


def finish(fig, ax, t, handles, labels, name, mode):
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(labels),
              frameon=False, fontsize=8, labelcolor=t['ink2'], handlelength=1.6,
              borderaxespad=0.3, columnspacing=1.2)
    fig.savefig(HERE / f'{name}_{mode}.png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def policy_episodes(folder, prefix):
    episodes = rows(folder, prefix, 'episodes')
    stats = rows(folder, prefix, 'action_stats')
    assert len(episodes) == len(stats)
    return [(e, s) for e, s in zip(episodes, stats) if e['phase'] == 'policy']


def training_rate(mode, key, name, title, ylabel):
    t = THEMES[mode]
    fig, ax = chart(t, title, f'Rolling mean over {WINDOW} training episodes (sampled actions, '
                    'exploration on). Equal budget: 137,680 environment steps per run.',
                    'Environment steps (thousands)', ylabel)
    handles, labels = [], []
    for label, folder, prefix, slot in available():
        data = policy_episodes(folder, prefix)
        x = np.array([int(e['env_steps']) for e, _ in data])[WINDOW - 1:] / 1000
        y = 100 * rolling([e[key] == 'True' for e, _ in data])
        handles.append(line(ax, t, x, y, slot))
        labels.append(label)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 140)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}%'))
    finish(fig, ax, t, handles, labels, name, mode)


def final_distance(mode):
    t = THEMES[mode]
    fig, ax = chart(t, 'Distance to the active goal when a training episode ends',
                    f'Rolling mean over {WINDOW} training episodes. Lower is better; the random prefill '
                    'averages ~1,150 m.', 'Environment steps (thousands)', 'Metres')
    handles, labels = [], []
    for label, folder, prefix, slot in available():
        data = policy_episodes(folder, prefix)
        x = np.array([int(e['env_steps']) for e, _ in data])[WINDOW - 1:] / 1000
        y = rolling([float(s['final_distance_to_active_goal_m']) for _, s in data])
        handles.append(line(ax, t, x, y, slot))
        labels.append(label)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 140)
    finish(fig, ax, t, handles, labels, 'final_distance_to_goal', mode)


def actor_gradient(mode):
    t = THEMES[mode]
    fig, ax = chart(t, 'Actor gradient norm during training (log scale)',
                    'One logged update every ~30 s. The actor is never clipped (clip = 100).',
                    'Actor-critic updates (thousands)', 'Gradient norm')
    ax.set_yscale('log')
    handles, labels = [], []
    for label, folder, prefix, slot in available():
        log = [r for r in rows(folder, prefix, 'train_log') if r['actor_gradient_norm'] and int(r['ac_updates']) > 0]
        handles.append(line(ax, t, [int(r['ac_updates']) / 1000 for r in log],
                            [float(r['actor_gradient_norm']) for r in log], slot))
        labels.append(label)
    finish(fig, ax, t, handles, labels, 'actor_gradient_norm', mode)


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    for mode in THEMES:
        training_rate(mode, 'pickup', 'training_pickup_rate',
                      'Pickup rate in training episodes', 'Episodes with a pickup')
        training_rate(mode, 'success', 'training_delivery_rate',
                      'Delivery rate in training episodes', 'Episodes delivered')
        final_distance(mode)
        actor_gradient(mode)
    print('figures written to', HERE)
