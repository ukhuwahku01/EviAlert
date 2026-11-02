"""Summaries and Fig. 6 for the WESAD validation."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'results/wesad_results.json'
figdir = sys.argv[2] if len(sys.argv) > 2 else 'figs'
D = json.load(open(src))
meta = D['meta']; H = meta['hours']
MODELS = ['LR', 'RF', 'HGB']
NAMES = {'LR': 'Logistic regression', 'RF': 'Random forest', 'HGB': 'Gradient boosting'}
METH = ['Frame', 'MajVote', 'MovAvg', 'SelMovAvg', 'MovAvg-Cal', 'EviAlert-C', 'EviAlert-W', 'EviAlert-W-noClip', 'EviAlert-W-noGate']
out = {'meta': dict(n_subjects=len(meta['subjects']), total_hours=sum(H.values()),
                    hours_range=[min(H.values()), max(H.values())],
                    stress_min=[min((e[0][1] - e[0][0]) * 5 / 60 for e in meta['events'].values()),
                                max((e[0][1] - e[0][0]) * 5 / 60 for e in meta['events'].values())],
                    motion_unusable=[min(meta['motion_unusable'].values()), max(meta['motion_unusable'].values()),
                                     float(np.mean(list(meta['motion_unusable'].values())))])}
for m in MODELS:
    M = D['models'][m]; out[m] = {'frame': M['frame']}
    for sc in ['primary', 'defined', 'natural']:
        R = M[sc]; o = {}
        Tst = [r['T'] for r in R]
        o['T'] = [float(np.mean(Tst)), float(np.min(Tst)), float(np.max(Tst))]
        o['eval_hours'] = float(np.mean([r['EviAlert-W']['hours'] for r in R]))
        for k in METH:
            g = lambda f: np.array([f(r) for r in R], float)
            evh = g(lambda r: sum(H[s] for s in r['eval']))
            o[k] = dict(
                r05=float(g(lambda r: r[k]['recall@0.5']).mean()), r1=float(g(lambda r: r[k]['recall@1.0']).mean()),
                r2=float(g(lambda r: r[k]['recall@2.0']).mean()),
                delay=float(np.nanmean(g(lambda r: r[k]['oracle']['delay_med']))),
                **{f'{rule}_comp': float(g(lambda r: r[k][rule]['compliant']).mean()) for rule in ['naive', 'poisson', 'gboot']},
                feas=float(g(lambda r: r[k]['gboot']['feasible']).mean()),
                g_fa=float(g(lambda r: r[k]['gboot']['fa']).mean()), g_rec=float(g(lambda r: r[k]['gboot']['recall']).mean()),
                g_delay=float(np.nanmean(g(lambda r: r[k]['gboot']['delay_med']))),
                stale100=float((g(lambda r: r[k]['gboot']['stale']) / evh * 100).mean()),
                ie=float(g(lambda r: r[k]['gboot']['insuff']).mean()))
        for k in ['EviAlert-W', 'EviAlert-W-noClip']:
            for x in ['0.5', '1.0', '2.0']:
                d = np.array([r[k][f'recall@{x}'] - r['MovAvg-Cal'][f'recall@{x}'] for r in R])
                o[f'{k}_vs_MovAvgCal_{x}'] = dict(diff=float(d.mean()), ge=float((d >= 0).mean()), gt=float((d > 0).mean()))
        out[m][sc] = o
json.dump(out, open(os.path.join(os.path.dirname(src), 'wesad_summary.json'), 'w'), indent=1)

# ---- print compact tables
for sc in ['primary', 'defined', 'natural']:
    print('=== scenario', sc)
    for m in MODELS:
        o = out[m][sc]; print('--', m, 'T*', np.round(o['T'], 2), 'eval h', round(o['eval_hours'], 1))
        for k in METH:
            v = o[k]
            print(f"   {k:18s} R {v['r05']:.3f} {v['r1']:.3f} {v['r2']:.3f} | delay {v['delay']:.0f} | comp {v['naive_comp']:.2f}/{v['poisson_comp']:.2f}/{v['gboot_comp']:.2f} feas {v['feas']:.2f} | dep FA {v['g_fa']:.2f} rec {v['g_rec']:.3f} | stale/100h {v['stale100']:.1f} ie {v['ie']:.3f}")
        for k in ['EviAlert-W', 'EviAlert-W-noClip']:
            print('   ', k, 'vs MovAvg-Cal', {x: {a: round(b, 3) for a, b in o[f'{k}_vs_MovAvgCal_{x}'].items()} for x in ['0.5', '1.0', '2.0']})
print(json.dumps(out['meta'], indent=0))
for m in MODELS:
    print(m, {k: round(v, 3) for k, v in out[m]['frame'].items()})

# ---- Fig. 6: operating curves (primary scenario)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'savefig.dpi': 300, 'legend.frameon': False})
STY = {'EviAlert-W': ('#2a78d6', '-', 'o', 'EviAlert (pre-specified, c = 3)'),
       'EviAlert-W-noClip': ('#2a78d6', '--', 'v', 'EviAlert without clipping (post hoc)'),
       'MovAvg-Cal': ('#eb6834', '-', 's', 'Moving average (calibrated)'),
       'Frame': ('#1baf7a', '--', '^', 'Single-frame threshold')}
xs = np.linspace(0.1, 3.0, 30)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, m in zip(axes, MODELS):
    R = D['models'][m]['primary']
    for k, (c, ls, mk, lab) in STY.items():
        ys = []
        for r in R:
            fa = np.array(r[k]['curve']['fa']); rec = np.array(r[k]['curve']['recall'])
            ys.append([rec[fa <= x + 1e-12].max() if (fa <= x + 1e-12).any() else 0.0 for x in xs])
        ax.plot(xs, np.mean(ys, 0), color=c, ls=ls, marker=mk, markevery=6, ms=3.5, lw=1.6, label=lab)
    ax.set_title(NAMES[m], fontsize=9); ax.set_xlabel('False alerts per hour'); ax.set_xlim(0, 3); ax.set_ylim(0, 1.02)
axes[0].set_ylabel('Recall of stress episodes')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.13), fontsize=8)
fig.tight_layout(); os.makedirs(figdir, exist_ok=True)
fig.savefig(os.path.join(figdir, 'fig6_wesad_curves.png'), bbox_inches='tight')
print('figure saved')
