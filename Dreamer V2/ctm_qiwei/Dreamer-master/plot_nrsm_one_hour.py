"""Plot saved NRSM diagnostics without loading TensorFlow or checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    history = json.loads((args.run / 'progress.json').read_text())
    result = json.loads((args.run / 'result.json').read_text())
    assert result['status'] == 'completed'
    # Final evaluation is recorded twice; keep exactly one per update.
    heldout = {row['update']: row['heldout'] for row in history if 'heldout' in row}
    train = {row['update']: row['train'] for row in history if 'train' in row}
    assert heldout[0] == result['before']
    assert heldout[result['updates']] == result['after']
    args.output.mkdir(parents=True, exist_ok=False)
    font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': font_manager.FontProperties(
        fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(), 'axes.unicode_minus': False,
        'font.size': 10, 'axes.titlesize': 12, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42})
    blue, orange, gray = '#2665ad', '#d36827', '#687581'

    def frame(title, subtitle, columns):
        fig, axes = plt.subplots(2, columns, figsize=(15.4 if columns == 3 else 12.6, 8.5))
        fig.subplots_adjust(left=.075, right=.97, bottom=.13, top=.82,
                            wspace=.29, hspace=.48)
        fig.suptitle(title, x=.075, y=.97, ha='left', fontsize=19)
        fig.text(.075, .914, subtitle, color=gray, fontsize=10)
        for ax in axes.flat:
            ax.set_xlabel('梯度更新次数（不是环境步数）')
            ax.set_xlim(0, result['updates'])
            ax.set_xticks(np.arange(0, result['updates'] + 1, 400))
            ax.grid(alpha=.18, linewidth=.6)
        return fig, axes

    def line(ax, records, key, label, color, style='-', scale=1):
        xs = sorted(records)
        ys = np.array([records[x][key] for x in xs]) * scale
        assert np.isfinite(ys).all(), key
        ax.plot(xs, ys, style, color=color, lw=2, marker='o', ms=3.5, label=label)
        return ys

    fig, axes = frame('NRSM｜一小时留出集验证',
        '官方 CTM-RL 对齐版 · Seed 17 · 16 条训练 / 8 条留出轨迹 · 2,400 次更新 · 仅世界模型，无 Actor-Critic', 3)
    ax = axes[0, 0]
    ax.set_title('状态重建与一步先验预测', loc='left')
    line(ax, heldout, 'posterior_vector_rmse', '后验重建（看当前观测）', blue)
    line(ax, heldout, 'prior_vector_rmse', '先验预测（不看当前观测）', orange, '--')
    ax.set_ylabel('归一化状态 RMSE ↓')
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]
    ax.set_title('奖励预测误差', loc='left')
    line(ax, heldout, 'posterior_reward_rmse', '后验', blue)
    line(ax, heldout, 'prior_reward_rmse', '先验', orange, '--')
    ax.set_ylabel('奖励 RMSE ↓')
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 2]
    ax.set_title('后验继续 / 终止预测损失', loc='left')
    line(ax, heldout, 'posterior_discount_bce', '后验 BCE', blue)
    ax.set_ylabel('二元交叉熵 ↓')
    for ax, horizon in zip(axes[1], (1, 5, 15)):
        ax.set_title(f'开环 {horizon} 步预测 · 对照状态不变基线', loc='left')
        values = line(ax, heldout, f'open_loop_{horizon}_vector_rmse', 'NRSM', blue)
        base = [r[f'persistence_{horizon}_vector_rmse'] for r in heldout.values()]
        np.testing.assert_allclose(base, base[0])
        ax.axhline(base[0], color=orange, linestyle='--', lw=1.8, label='状态不变基线')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('归一化状态 RMSE ↓')
        ax.legend(frameon=False, fontsize=9, loc='lower left', bbox_to_anchor=(0, .12))
        ax.text(.97, .91, f'最终：模型 {values[-1]:.3f} / 基线 {base[0]:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9, color=gray)
    fig.text(.075, .052, '点为真实评估记录，直线仅连接记录，不平滑、不插值造点；最终重复记录已去重。归一化 RMSE 不是米。', fontsize=9, color=gray)
    fig.text(.075, .024, '开环预测使用真实未来动作、不使用未来观测；同一组 8 条留出轨迹，单 seed，不能据此判断配送成功率或长记忆优势。', fontsize=9, color=gray)
    fig.savefig(args.output / 'nrsm_heldout_curves.png', dpi=170)

    fig2, axes = frame('NRSM｜一小时训练与数值稳定性',
        '固定离线数据 · Adam 3e-4 · batch=2 · 全局梯度裁剪阈值 100 · 程序记录耗时 60 分 9 秒', 2)
    for ax, key, title in ((axes[0, 0], 'loss', '世界模型总损失'),
                           (axes[0, 1], 'kl', '后验与先验 KL')):
        ax.set_title(title, loc='left')
        xs = sorted(train)
        ys = np.array([train[x][key] for x in xs])
        ax.plot(xs, ys, color=blue, alpha=.25, lw=.8, label='原始日志值')
        mean = np.array([ys[max(0, i-4):i+1].mean() for i in range(len(ys))])
        ax.plot(xs, mean, color=blue, lw=2, label='最近 5 个记录点均值')
        ax.set_ylabel('损失' if key == 'loss' else 'KL')
        ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]
    ax.set_title('裁剪前梯度范数（对数轴）', loc='left')
    line(ax, train, 'gradient_norm', '记录时的梯度范数', blue)
    line(ax, train, 'gradient_norm_window_max', '记录窗口内最大值', orange, '--')
    ax.axhline(100, color=gray, ls=':', lw=1.5, label='裁剪阈值 100')
    ax.set_yscale('log')
    ax.set_ylabel('全局梯度范数')
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]
    ax.set_title('各记录窗口内触发梯度裁剪的比例', loc='left')
    line(ax, train, 'clipped_fraction', '被裁剪更新占比', blue, scale=100)
    ax.set_ylim(-3, 103)
    ax.set_ylabel('更新占比（%）')
    fig2.text(.075, .052, '训练约每 30 秒记录；loss/KL 为记录时的 minibatch 值，不是窗口均值。MA5 为单边滑动平均，起始按已有记录计算。', fontsize=9, color=gray)
    fig2.text(.075, .024, '梯度和裁剪比例不平滑，以保留异常峰值；末次记录窗口仅 3 次更新，33.3% 表示其中 1 次被裁剪。Checkpoint 恢复校验通过。', fontsize=9, color=gray)
    fig2.savefig(args.output / 'nrsm_training_curves.png', dpi=170)
    with PdfPages(args.output / 'nrsm_one_hour_curves.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    plt.close('all')
    provenance = dict(run=str(args.run), unique_heldout_points=len(heldout),
                      training_records=len(train), smoothing='loss and KL trailing MA5; others unsmoothed',
                      sha256={f: hashlib.sha256((args.run / f).read_bytes()).hexdigest()
                              for f in ('progress.json', 'result.json', 'manifest.json')})
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
