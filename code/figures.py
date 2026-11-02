import os
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, '.')
from evialert import core
from evialert.experiments import load_oof, method_scores, W_DEFAULT
from evialert.simulate import make_sessions, DISTRESS

os.makedirs('figs', exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.8,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'savefig.dpi': 300,
                     'legend.frameon': False})
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
BK = ['resnet50', 'vit_b16', 'convnext_t', 'swin_t', 'effnet_b0']
NAMES = {'resnet50': 'ResNet-50', 'vit_b16': 'ViT-B/16', 'convnext_t': 'ConvNeXt-T', 'swin_t': 'Swin-T',
         'effnet_b0': 'EfficientNet-B0'}
STYLE = {'EviAlert-W': (BLUE, '-', 'o', 2.0, 'EviAlert'), 'MovAvg-Cal': (ORANGE, '-', 's', 1.4, 'Moving average (calibrated)'),
         'Frame': (AQUA, '--', '^', 1.4, 'Single-frame threshold'), 'MovAvg': (GRAY, ':', 'D', 1.4, 'Moving average (raw)')}

# ------------------------------------------------------------------ Fig. 3 operating curves
def mean_curve(rows, m, xs):
    ys = []
    for r in rows:
        fa = np.array(r['methods'][m]['curve']['fa']); rec = np.array(r['methods'][m]['curve']['recall'])
        ys.append([rec[fa <= x + 1e-12].max() if (fa <= x + 1e-12).any() else 0.0 for x in xs])
    ys = np.array(ys)
    return ys.mean(0), ys.std(0) / np.sqrt(len(ys))

xs = np.linspace(0.1, 3.0, 30)
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
for ax, b in zip(axes, BK):
    rows = json.load(open(f'results/main_{b}.json'))
    for m in ['MovAvg', 'Frame', 'MovAvg-Cal', 'EviAlert-W']:
        c, ls, mk, lw, lab = STYLE[m]
        mu, se = mean_curve(rows, m, xs)
        ax.plot(xs, mu, color=c, ls=ls, lw=lw, label=lab, marker=mk, markevery=6, ms=3.5)
    ax.set_title(NAMES[b], fontsize=9, color=INK)
    ax.set_xlabel('False alerts per hour')
    ax.set_xlim(0, 3); ax.set_ylim(0, 0.75)
axes[0].set_ylabel('Event recall')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.07), fontsize=8)
fig.tight_layout()
fig.savefig('figs/fig3_operating_curves.png', bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ Fig. 4 robustness
miss = json.load(open('results/robust_missing.json')); pers = json.load(open('results/robust_persist.json'))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
RM = {'EviAlert-W': (BLUE, '-', 'o', 'EviAlert'), 'EviAlert-W-noGate': (AQUA, '--', '^', 'EviAlert without gate'),
      'MovAvg-Cal': (ORANGE, '-', 's', 'Moving average (calibrated)')}
rates = [0.0, 0.1, 0.2, 0.3]
for m, (c, ls, mk, lab) in RM.items():
    st = [np.mean([np.mean([r[m]['gboot']['stale'] for r in miss[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in rates]
    rc = [np.mean([np.mean([r[m]['recall@1.0'] for r in miss[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in rates]
    axes[0].plot([100 * x for x in rates], st, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
    axes[1].plot([100 * x for x in rates], rc, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
    pv = [1.0, 2.0, 4.0]
    rp = [np.mean([np.mean([r[m]['recall@1.0'] for r in pers[f'{b}|{x}']]) for b in ['resnet50', 'convnext_t']]) for x in pv]
    axes[2].plot(pv, rp, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
axes[0].set_xlabel('Unusable frames (%)'); axes[0].set_ylabel('Stale alerts per 150 h')
axes[0].set_title('(a) Alerts fired on stale evidence', fontsize=9, loc='left')
axes[1].set_xlabel('Unusable frames (%)'); axes[1].set_ylabel('Recall at 1 false alert/h')
axes[1].set_title('(b) Detection under missing frames', fontsize=9, loc='left')
axes[2].set_xlabel('Mean frame persistence (steps)'); axes[2].set_ylabel('Recall at 1 false alert/h')
axes[2].set_title('(c) Temporal correlation', fontsize=9, loc='left'); axes[2].set_xticks([1, 2, 4])
axes[1].set_ylim(0.25, 0.65); axes[2].set_ylim(0.25, 0.65)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=8)
fig.tight_layout(); fig.savefig('figs/fig4_robustness.png', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ Fig. 5 budget compliance
S = json.load(open('results/summary_tables.json'))
rules = [('naive', 'Empirical rate'), ('poisson', 'Poisson bound'), ('gboot', 'Identity bootstrap')]
mets = [('MovAvg-Cal', ORANGE, 'Moving average (calibrated)'), ('EviAlert-W', BLUE, 'EviAlert')]
fig, ax = plt.subplots(figsize=(4.4, 2.6))
x = np.arange(len(rules)); wdt = 0.36
for k, (m, c, lab) in enumerate(mets):
    vals = [100 * np.mean([S[b]['methods'][m][f'{r}_compliant'] for b in BK]) for r, _ in rules]
    bars = ax.bar(x + (k - 0.5) * (wdt + 0.02), vals, width=wdt, color=c, label=lab, zorder=3)
    for xi, v in zip(x + (k - 0.5) * (wdt + 0.02), vals):
        ax.text(xi, v - 3, f'{v:.0f}', ha='center', va='top', fontsize=7.5, color='#ffffff', fontweight='bold')
ax.axhline(95, color=MUTED, lw=0.8, ls='--', zorder=2); ax.text(-0.45, 96.5, 'target 95%', fontsize=7, color=MUTED, ha='left', va='bottom')
ax.set_xticks(x); ax.set_xticklabels(['Empirical\nrate', 'Poisson\nbound', 'Identity\nbootstrap']); ax.set_ylim(0, 112)
ax.set_ylabel('Splits within budget (%)'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', fontsize=7.5, bbox_to_anchor=(0, 1.18), ncol=2)
fig.tight_layout(); fig.savefig('figs/fig5_compliance.png', bbox_inches='tight'); plt.close(fig)

# ------------------------------------------------------------------ Fig. 2 example session
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', 'resnet50')
rng = np.random.default_rng(101)   # same identity split as evaluation split seed 101
ug = np.unique(g); rng.shuffle(ug); cal = np.isin(g, ug[:len(ug) // 2])
ci, ei = np.where(cal)[0], np.where(~cal)[0]
Ts = core.fit_temperature(P[ci], y[ci]); Pc = core.temperature_scale(P, Ts); pr = float(np.isin(y[ci], DISTRESS).mean())
# pick a session that shows the Insufficient Evidence state: one sustained event, an occlusion burst,
# and the calibrated moving average firing on stale (carried-forward) evidence
thr = json.load(open('results/main_resnet50.json'))[0]['methods']
hE = thr['EviAlert-W']['gboot']['h']; hM = thr['MovAvg-Cal']['gboot']['h']
# example chosen by scratchpad search (trial 9889): one sustained episode, a burst of unusable frames during
# which the calibrated moving average fires a stale false alert, no EviAlert false alert
trial = 9889
S1 = make_sessions(ei, y, g, 1, np.random.default_rng(5000 + trial))
E, use_, _, _, _ = method_scores('EviAlert-W', P, Pc, S1, pr, 0.5)
onE, stE = core.run_policy(E, use_, [hE], h_watch=[hE / 2], window=W_DEFAULT, rho=0.5, refractory=24, gate=True, accumulate=False)
M, use2, _, _, _ = method_scores('MovAvg-Cal', P, Pc, S1, pr, 0.5)
onM, _ = core.run_policy(M, use2, [hM], window=W_DEFAULT, refractory=24, gate=False, accumulate=False)
print('example trial', trial)
T = S1['state'].shape[1]; t = np.arange(T) * 5 / 60.0
fig, axes = plt.subplots(4, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1.4, 1.4, 0.5]))
SC = {0: ('#eb6834', 'Angry'), 1: ('#6250d6', 'Cry'), 2: ('#1baf7a', 'Laugh'), 3: ('#d4d3cd', 'Normal')}
ax = axes[0]
for k in range(T):
    ax.axvspan(t[k], t[k] + 5 / 60, color=SC[S1['state'][0, k]][0], lw=0)
ax.set_yticks([]); ax.set_ylabel('True\nstate', rotation=0, ha='right', va='center'); ax.grid(False)
ax.legend(handles=[Patch(color=SC[k][0], label=SC[k][1]) for k in [3, 2, 0, 1]], loc='upper center',
          bbox_to_anchor=(0.5, 1.9), ncol=5, fontsize=7.5)
for a in axes[1:3]:
    for k in np.where(~S1['usable'][0])[0]:
        a.axvspan(t[k], t[k] + 5 / 60, color='#f1f0ec', lw=0, zorder=0)
ax = axes[1]
ax.plot(t, E[0], color=BLUE, lw=1.6); ax.axhline(hE, color=INK, lw=0.8, ls='--'); ax.axhline(hE / 2, color=MUTED, lw=0.8, ls=':')
ax.text(t[-1], hE, ' High Alert', va='center', fontsize=7, color=INK); ax.text(t[-1], hE / 2, ' Watchlist', va='center', fontsize=7, color=MUTED)
for k in np.where(onE[0, 0])[0]:
    ax.plot(t[k], E[0, k], marker='v', color=INK, ms=7)
ax.set_ylabel('EviAlert\nevidence (30 s)')
ax = axes[2]
ax.plot(t, M[0], color=ORANGE, lw=1.6); ax.axhline(hM, color=INK, lw=0.8, ls='--')
for k in np.where(onM[0, 0])[0]:
    ax.plot(t[k], M[0, k], marker='v', color=INK, ms=7)
ax.set_ylabel('Moving avg.\nP(distress)')
ax = axes[3]
STC = {core.NO_ALERT: '#ffffff', core.WATCH: '#f5d58a', core.HIGH: '#e34948', core.INSUFF: '#b9b8b2'}
for k in range(T):
    ax.axvspan(t[k], t[k] + 5 / 60, color=STC[int(stE[0, 0, k])], lw=0)
ax.set_yticks([]); ax.set_ylabel('EviAlert\nstatus', rotation=0, ha='right', va='center'); ax.grid(False)
ax.legend(handles=[Patch(facecolor='#f5d58a', label='Watchlist'), Patch(facecolor='#e34948', label='High Alert'),
                   Patch(facecolor='#b9b8b2', label='Insufficient Evidence')], loc='upper center',
          bbox_to_anchor=(0.5, -0.9), ncol=3, fontsize=7.5)
ax.set_xlabel('Minutes'); axes[3].set_xlim(0, t[-1] + 5 / 60)
fig.tight_layout(); fig.savefig('figs/fig2_example_session.png', bbox_inches='tight'); plt.close(fig)
print('event', S1['events'][0], 'EviAlert onsets', np.where(onE[0, 0])[0], 'MovAvg onsets', np.where(onM[0, 0])[0], 'missing steps', int((~S1['usable'][0]).sum()))
