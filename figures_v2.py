"""Figures 2-5 for the final manuscript (EviAlert = tempered filter; EviAlert-W = first version)."""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evialert import core
from evialert.experiments import load_oof, method_scores, W_DEFAULT
from evialert.simulate import make_sessions, DISTRESS

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'legend.frameon': False})
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
BK = ['resnet50', 'vit_b16', 'convnext_t', 'swin_t', 'effnet_b0']
NAMES = {'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16', 'convnext_t': 'ConvNeXt-T', 'swin_t': 'Swin-T', 'effnet_b0': 'EfficientNet-B0'}
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))

def curves(rows, m, xs):
    ys = []
    for r in rows:
        fa = np.array(r['methods'][m]['curve']['fa']); rec = np.array(r['methods'][m]['curve']['recall'])
        ys.append([rec[fa <= x + 1e-12].max() if (fa <= x + 1e-12).any() else 0.0 for x in xs])
    return np.mean(ys, 0)

# ---- Fig. 2 operating curves
xs = np.linspace(0.1, 3.0, 30)
STY = [('F', 'EviAlert-F', BLUE, '-', 'o', 2.0, 'EviAlert'), ('1', 'EviAlert-W', BLUE, ':', 'v', 1.4, 'EviAlert-W (first version)'),
       ('1', 'MovAvg-Cal', ORANGE, '-', 's', 1.4, 'Moving average (calibrated)'), ('1', 'Frame', AQUA, '--', '^', 1.4, 'Single-frame threshold')]
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.4), sharey=True)
for ax, b in zip(axes, BK):
    R1 = json.load(open(f'results/main_{b}.json')); R2 = json.load(open(f'results/main_v2_{b}.json'))
    for src, m, c, ls, mk, lw, lab in STY:
        ax.plot(xs, curves(R2 if src == 'F' else R1, m, xs), color=c, ls=ls, lw=lw, marker=mk, markevery=6, ms=3.5, label=lab)
    ax.set_title(NAMES[b], fontsize=9); ax.set_xlabel('False alerts per hour'); ax.set_xlim(0, 3); ax.set_ylim(0, 0.75)
axes[0].set_ylabel('Event recall')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
fig.tight_layout(); fig.savefig('figs/fig3_operating_curves_v2.png', bbox_inches='tight'); plt.close(fig)

# ---- Fig. 5 robustness
m1 = json.load(open('results/robust_missing.json')); p1 = json.load(open('results/robust_persist.json'))
m2 = json.load(open('results/robust_v2_missing.json')); p2 = json.load(open('results/robust_v2_persist.json'))
RM = [('EviAlert-F', m2, p2, BLUE, '-', 'o', 'EviAlert'), ('EviAlert-W', m1, p1, BLUE, ':', 'v', 'EviAlert-W (first version)'),
      ('EviAlert-W-noGate', m1, p1, AQUA, '--', '^', 'EviAlert-W without gate'), ('MovAvg-Cal', m1, p1, ORANGE, '-', 's', 'Moving average (calibrated)')]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
rates = [0.0, 0.1, 0.2, 0.3]; pv = [1.0, 2.0, 4.0]
for m, MM, PP, c, ls, mk, lab in RM:
    st = [np.mean([np.mean([r[m]['gboot']['stale'] for r in MM[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in rates]
    rc = [np.mean([np.mean([r[m]['recall@1.0'] for r in MM[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in rates]
    rp = [np.mean([np.mean([r[m]['recall@1.0'] for r in PP[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in pv]
    axes[0].plot([100 * x for x in rates], st, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
    axes[1].plot([100 * x for x in rates], rc, color=c, ls=ls, marker=mk, ms=4, lw=1.6)
    axes[2].plot(pv, rp, color=c, ls=ls, marker=mk, ms=4, lw=1.6)
axes[0].set_xlabel('Unusable frames (%)'); axes[0].set_ylabel('Stale alerts per 150 h'); axes[0].set_title('(a) Alerts fired on stale evidence', fontsize=9, loc='left')
axes[1].set_xlabel('Unusable frames (%)'); axes[1].set_ylabel('Recall at 1 false alert/h'); axes[1].set_title('(b) Detection under missing frames', fontsize=9, loc='left')
axes[2].set_xlabel('Mean frame persistence (steps)'); axes[2].set_ylabel('Recall at 1 false alert/h'); axes[2].set_title('(c) Temporal correlation', fontsize=9, loc='left'); axes[2].set_xticks(pv)
axes[1].set_ylim(0.3, 0.6); axes[2].set_ylim(0.3, 0.6)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7.5)
fig.tight_layout(); fig.savefig('figs/fig4_robustness_v2.png', bbox_inches='tight'); plt.close(fig)

# ---- Fig. 4 compliance (pooled rules)
S = json.load(open('results/summary_tables.json'))
F = {b: json.load(open(f'results/main_v2_{b}.json')) for b in BK}
rules = [('naive', 'Empirical\nrate'), ('poisson', 'Poisson\nbound'), ('gboot', 'Identity\nbootstrap')]
vals_mc = [100 * np.mean([S[b]['methods']['MovAvg-Cal'][f'{r}_compliant'] for b in BK]) for r, _ in rules]
vals_f = [100 * np.mean([np.mean([x['methods']['EviAlert-F'][r]['compliant'] for x in F[b]]) for b in BK]) for r, _ in rules]
fig, ax = plt.subplots(figsize=(4.4, 2.6)); x = np.arange(3); w = 0.36
for k, (vals, c, lab) in enumerate([(vals_mc, ORANGE, 'Moving average (calibrated)'), (vals_f, BLUE, 'EviAlert')]):
    xx = x + (k - 0.5) * (w + 0.02); ax.bar(xx, vals, width=w, color=c, label=lab, zorder=3)
    for xi, v in zip(xx, vals):
        ax.text(xi, v - 3, f'{v:.0f}', ha='center', va='top', fontsize=7.5, color='#ffffff', fontweight='bold')
ax.axhline(95, color=MUTED, lw=0.8, ls='--', zorder=2); ax.text(-0.45, 96.5, 'target 95%', fontsize=7, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in rules]); ax.set_ylim(0, 112); ax.set_ylabel('Splits within budget (%)'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', fontsize=7.5, bbox_to_anchor=(0, 1.18), ncol=2)
fig.tight_layout(); fig.savefig('figs/fig5_compliance_v2.png', bbox_inches='tight'); plt.close(fig)

# ---- Fig. 3 example session with EviAlert (filter)
P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', 'resnet50')
rng = np.random.default_rng(101); ug = np.unique(g); rng.shuffle(ug); cal = np.isin(g, ug[:len(ug) // 2])
ci, ei = np.where(cal)[0], np.where(~cal)[0]
Ts = core.fit_temperature(P[ci], y[ci]); Pc = core.temperature_scale(P, Ts); pr = float(np.isin(y[ci], DISTRESS).mean())
thr1 = json.load(open('results/main_resnet50.json'))[0]['methods']; r2 = json.load(open('results/main_v2_resnet50.json'))[0]
hF = r2['methods']['EviAlert-F']['gboot']['h']; filt = r2['meta']['filt']; hM = thr1['MovAvg-Cal']['gboot']['h']
S1 = make_sessions(ei, y, g, 1, np.random.default_rng(5000 + 4482))   # illustrative session (see caption)
E, use_, _, _, _ = method_scores('EviAlert-F', P, Pc, S1, pr, 0.5, filt=filt)
onE, stE = core.run_policy(E, use_, [hF], h_watch=[hF - 2], window=W_DEFAULT, rho=0.5, refractory=24, gate=True, accumulate=False)
M, use2, _, _, _ = method_scores('MovAvg-Cal', P, Pc, S1, pr, 0.5)
onM, _ = core.run_policy(M, use2, [hM], window=W_DEFAULT, refractory=24, gate=False, accumulate=False)
T = S1['state'].shape[1]; t = np.arange(T) * 5 / 60.0
fig, axes = plt.subplots(4, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1.4, 1.4, 0.5]))
SC = {0: ('#eb6834', 'Angry'), 1: ('#6250d6', 'Cry'), 2: ('#1baf7a', 'Laugh'), 3: ('#d4d3cd', 'Normal')}
ax = axes[0]
for k in range(T):
    ax.axvspan(t[k], t[k] + 5 / 60, color=SC[S1['state'][0, k]][0], lw=0)
ax.set_yticks([]); ax.set_ylabel('True\nstate', rotation=0, ha='right', va='center'); ax.grid(False)
ax.legend(handles=[Patch(color=SC[k][0], label=SC[k][1]) for k in [3, 2, 0, 1]], loc='upper center', bbox_to_anchor=(0.5, 1.9), ncol=5, fontsize=7.5)
for a in axes[1:3]:
    for k in np.where(~S1['usable'][0])[0]:
        a.axvspan(t[k], t[k] + 5 / 60, color='#f1f0ec', lw=0, zorder=0)
ax = axes[1]; ax.plot(t, E[0], color=BLUE, lw=1.6); ax.axhline(hF, color=INK, lw=0.8, ls='--'); ax.axhline(hF - 2, color=MUTED, lw=0.8, ls=':')
ax.text(t[-1], hF, ' High Alert', va='center', fontsize=7, color=INK); ax.text(t[-1], hF - 2, ' Watchlist', va='center', fontsize=7, color=MUTED)
for k in np.where(onE[0, 0])[0]:
    ax.plot(t[k], E[0, k], marker='v', color=INK, ms=7)
ax.set_ylabel('EviAlert\nposterior log-odds')
ax = axes[2]; ax.plot(t, M[0], color=ORANGE, lw=1.6); ax.axhline(hM, color=INK, lw=0.8, ls='--')
for k in np.where(onM[0, 0])[0]:
    ax.plot(t[k], M[0, k], marker='v', color=INK, ms=7)
ax.set_ylabel('Moving avg.\nP(distress)')
ax = axes[3]; STC = {core.NO_ALERT: '#ffffff', core.WATCH: '#f5d58a', core.HIGH: '#e34948', core.INSUFF: '#b9b8b2'}
for k in range(T):
    ax.axvspan(t[k], t[k] + 5 / 60, color=STC[int(stE[0, 0, k])], lw=0)
ax.set_yticks([]); ax.set_ylabel('EviAlert\nstatus', rotation=0, ha='right', va='center'); ax.grid(False)
ax.legend(handles=[Patch(facecolor='#f5d58a', label='Watchlist'), Patch(facecolor='#e34948', label='High Alert'), Patch(facecolor='#b9b8b2', label='Insufficient Evidence')],
          loc='upper center', bbox_to_anchor=(0.5, -0.9), ncol=3, fontsize=7.5)
ax.set_xlabel('Minutes'); axes[3].set_xlim(0, t[-1] + 5 / 60)
fig.tight_layout(); fig.savefig('figs/fig2_example_session_v2.png', bbox_inches='tight'); plt.close(fig)
print('EviAlert onsets', np.where(onE[0, 0])[0], 'MovAvg onsets', np.where(onM[0, 0])[0], 'event', S1['events'][0], 'IE steps', np.where(stE[0,0]==core.INSUFF)[0])
