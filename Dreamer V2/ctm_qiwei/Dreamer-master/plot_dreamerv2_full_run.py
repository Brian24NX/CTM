"""Render completed, matched DreamerV2 runs; no TF/GPU or checkpoint loading."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


def rolling(values, window):
    if window < 1:
        raise ValueError('Moving-average window must be positive')
    values = np.asarray(values, float)
    total = np.concatenate([[0.], np.cumsum(values)])
    end = np.arange(1, len(values)+1)
    start = np.maximum(0, end-window)
    return (total[end]-total[start])/(end-start)


def series(rows, key, mode):
    groups = defaultdict(list)
    for row in rows:
        if key in row and np.isfinite(row[key]):
            groups[int(row['step'])].append(float(row[key]))
    xs = sorted(groups)
    # Evaluation rows are individual episodes at one evaluation checkpoint.
    # Optimizer duplicate rows use the latest value, never average resumed logs.
    ys = [np.mean(groups[x]) if mode == 'test' else groups[x][-1] for x in xs]
    return np.asarray(xs)/1000, np.asarray(ys), [len(groups[x]) for x in xs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-window', type=int, default=100)
    parser.add_argument('--eval-window', type=int, default=5)
    parser.add_argument('--loss-window', type=int, default=15)
    args = parser.parse_args()
    if min(args.train_window, args.eval_window, args.loss_window) < 1:
        parser.error('All moving-average windows must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    font = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    batch = json.loads((args.root/'batch_status.json').read_text())
    if batch['status'] != 'completed':
        raise ValueError('Expected completed run, not current training')
    data, audit = {}, dict(source_root=str(args.root), runs=[],
        note='Existing customized DreamerV2, not an audited official-baseline reproduction. Seed1 excluded: old-KL first100k.',
        smoothing=dict(method='trailing arithmetic moving average, including current point',
            train_episodes=args.train_window, evaluation_checkpoints=args.eval_window,
            loss_records=args.loss_window, edge='available samples only; no padding',
            references='train MA100 when window differs; raw evaluation and losses'))
    for seed in (0, 2):
        path = args.root/f'seed_{seed}'/'metrics.jsonl'
        raw = path.read_bytes()
        rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
        metadata = [json.loads(x) for x in (path.parent/'run_metadata.jsonl').read_text().splitlines() if x.strip()]
        if metadata[0]['resumed_checkpoint']:
            raise ValueError('Expected from-zero provenance')
        data[seed] = rows
        train = [r for r in rows if 'train/success' in r and r['step'] >= 5000]
        loss = [r for r in rows if 'model_loss' in r]
        entry = next(r for r in batch['runs'] if r['seed'] == seed)
        test = [r for r in rows if 'test/success' in r]
        end = max(r['step'] for r in test)
        final = [r for r in test if r['step'] == end]
        audit['runs'].append(dict(seed=seed, final_step=entry['final_step'],
            metrics_sha256=hashlib.sha256(raw).hexdigest(), rows=len(rows),
            logged_train_episodes=len(train), loss_records=len(loss), test_episodes=len(test),
            test_checkpoints=len(set(r['step'] for r in test)),
            decreasing_steps=sum(b['step'] < a['step'] for a,b in zip(rows, rows[1:])),
            final_online_test={k: float(np.mean([r['test/'+k] for r in final]))
                               for k in ('success','pickup','truncated','return')},
            final_online_test_episodes=len(final)))
    colors, styles = {0:'#2665ad', 2:'#d36827'}, {0:'-', 2:'--'}
    handles = [Line2D([0],[0],color=colors[s],lw=2.3,ls=styles[s],label=f'Seed {s}') for s in (0,2)]

    def frame(title, subtitle):
        fig, axes = plt.subplots(2,3,figsize=(15.4,8.8))
        fig.subplots_adjust(left=.066,right=.985,top=.83,bottom=.12,wspace=.27,hspace=.38)
        fig.suptitle(title, x=.065,ha='left',fontsize=19,fontweight='normal',y=.975)
        fig.text(.066,.921,subtitle,fontsize=10,color='#52606a')
        fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.983,.91),ncol=2,frameon=False)
        for ax in axes.flat:
            ax.set_xlim(0,305)
            ax.set_xticks(np.arange(0,301,50))
            ax.set_xlabel('环境步数（千步）')
            ax.grid(alpha=.18, linewidth=.6)
        return fig, axes

    fig, axes = frame('Dreamer V2｜任务表现 · 滑动平均',
        'KL 修复后从零训练 · MOVE/TURN + 自动取货 · Seed 0: 300,025 步；Seed 2: 300,000 步')
    for row_index, phase in enumerate(('train','test')):
        for col, (key,title) in enumerate((('success','配送成功率'),('pickup','取货率'),('truncated','超时率'))):
            ax = axes[row_index,col]
            ax.set_title(('训练回合 · ' if phase == 'train' else '在线测试 · ')+title,loc='left')
            ax.set_ylabel('比例（%）')
            ax.set_ylim(-3,103)
            for seed, rows in data.items():
                selected = [r for r in rows if r['step'] >= 5000]
                xs, ys, _ = series(selected, phase+'/'+key, phase)
                if phase == 'test':
                    ax.plot(xs,ys*100,color=colors[seed],alpha=.12,lw=.7)
                    ax.scatter(xs,ys*100,color=colors[seed],alpha=.15,s=7)
                elif args.train_window != 100:
                    ax.plot(xs,rolling(ys,100)*100,color=colors[seed],alpha=.18,lw=.8)
                trend = rolling(ys,args.train_window if phase == 'train' else args.eval_window)
                ax.plot(xs,trend*100,color=colors[seed],ls=styles[seed],lw=2.1)
    train_ref = '训练淡线为原版 MA100；' if args.train_window != 100 else ''
    fig.text(.066,.046,f'粗线：训练最近 {args.train_window} 局、测试最近 {args.eval_window} 次评估的简单滑动平均。{train_ref}测试淡线为原始评估值。',fontsize=9,color='#52606a')
    fig.text(.066,.020,'单边窗口包含当前点，起始不足窗口按已有样本计算；平滑存在滞后。每次测试 5 局，非固定场景；Seed 1 未混入。',fontsize=9,color='#52606a')
    fig.savefig(args.output/'dreamerv2_full_task_curves.png',dpi=170)

    fig2, axes2 = frame('Dreamer V2｜世界模型与 Actor-Critic · 滑动平均',
        f'同一轮完整训练日志 · 淡线为原始记录，粗线为最近 {args.loss_window} 个记录点均值 · 不混入预热阶段或 NRSM 数据')
    specs = [('model_loss','世界模型总损失','损失'),('vector_loss','状态重建损失','Gaussian NLL'),
             ('kl','后验与先验 KL','KL'),('actor_loss','Actor 总损失','损失'),
             ('actor_bc_loss','Actor 行为克隆损失','损失'),('critic_loss','Critic 损失','损失')]
    for ax,(key,title,ylabel) in zip(axes2.flat,specs):
        ax.set_title(title,loc='left')
        ax.set_ylabel(ylabel)
        for seed, rows in data.items():
            xs, ys, _ = series([r for r in rows if r['step'] >= 5000],key,'loss')
            if not len(xs):
                raise ValueError('Missing metric: '+key)
            ax.plot(xs,ys,color=colors[seed],alpha=.17,lw=.7)
            ax.plot(xs,rolling(ys,args.loss_window),color=colors[seed],ls=styles[seed],lw=2)
        if key in ('kl','actor_loss','actor_bc_loss','critic_loss'):
            ax.axhline(0,color='#8c969f',lw=.7,alpha=.5)
    fig2.text(.066,.046,'单边简单滑动平均，起始不足窗口按已有样本计算；平滑存在滞后。Loss 或 KL 下降不等于策略变好。',fontsize=9,color='#52606a')
    fig2.text(.066,.020,'该历史实现包含控制器示范与 BC 等工程设置，不等同于已审计的官方 DreamerV2 baseline。',fontsize=9,color='#52606a')
    fig2.savefig(args.output/'dreamerv2_full_learning_curves.png',dpi=170)
    with PdfPages(args.output/'dreamerv2_full_curves.pdf') as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
    plt.close('all')
    (args.output/'provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
