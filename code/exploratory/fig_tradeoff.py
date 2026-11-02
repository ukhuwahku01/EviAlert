"""Fig. 4: what a person-level budget guarantee costs. Share of held-out persons within the budget and recall at the
selected threshold, as a function of the number of calibration persons, for certification (Learn then Test),
person-level conformal risk control and the empirical pooled rate (EviAlert, mean over three classifiers).
Usage: python exploratory/fig_tradeoff.py results/explore_wesad.json [results/explore_nurse.json] out.png"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.8, 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'legend.frameon': False})
BLUE, ORANGE, GRAY, MUTED = '#2a78d6', '#eb6834', '#8a8984', '#52514e'
RULES = [('n_ltt', 'Certification (Learn then Test)', BLUE, '-', 'o'),
         ('n_crc', 'Conformal risk control (persons)', ORANGE, '--', 's'),
         ('n_naive', 'Empirical pooled rate', GRAY, ':', '^')]
files = sys.argv[1:-1]; out = sys.argv[-1]
names = ['WESAD (development)', 'Nurses (confirmatory data)'][:len(files)]
fig, axes = plt.subplots(len(files), 2, figsize=(7.2, 2.6 * len(files)), squeeze=False)
for row, (f, nm) in enumerate(zip(files, names)):
    D = json.load(open(f)); mods = ['LR', 'RF', 'HGB']
    sizes = sorted(int(k) for k in D['LR']['EviAlert-F']['n_ltt'])
    for key, lab, c, ls, mk in RULES:
        w = np.mean([[D[m]['EviAlert-F'][key][str(n)]['within'] for n in sizes] for m in mods], 0)
        r = np.mean([[D[m]['EviAlert-F'][key][str(n)]['recall'] for n in sizes] for m in mods], 0)
        axes[row, 0].plot(sizes, w, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
        axes[row, 1].plot(sizes, r, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=lab)
    for ax in axes[row]:
        ax.axvspan(min(sizes) - 0.5, 10.5, color='#f2f1ec', zorder=0, lw=0)
        ax.set_xticks(sizes); ax.set_xlim(min(sizes) - 0.5, max(sizes) + 0.5); ax.set_xlabel('Calibration persons')
    axes[row, 0].axhline(0.8, color=MUTED, lw=0.8, ls='-.'); axes[row, 0].text(min(sizes) - 0.3, 0.79, 'target 0.8', ha='left', va='top', fontsize=7.5, color=MUTED)
    axes[row, 0].set_ylim(0.4, 1.02); axes[row, 0].set_ylabel('Held-out persons\nwithin budget')
    axes[row, 1].set_ylim(0, 1.0); axes[row, 1].set_ylabel('Event recall at\nselected threshold')
    axes[row, 0].set_title(f'{nm}: budget compliance', fontsize=9, loc='left')
    axes[row, 1].set_title(f'{nm}: detection', fontsize=9, loc='left')
    axes[row, 0].text(min(sizes) - 0.3, 0.42, 'certification\nnot possible', fontsize=7.5, color=MUTED, va='bottom')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(out, bbox_inches='tight')
print('saved', out)
