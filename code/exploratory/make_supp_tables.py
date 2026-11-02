"""Markdown tables for the Supplementary Material: exploratory analyses (Tables S4-S5) and per-nurse results (Table S6).
Usage: python exploratory/make_supp_tables.py results/explore_wesad.json results/explore_nurse.json results/nurse_results.json"""
import json, sys
import numpy as np

MODS = ['LR', 'RF', 'HGB']; MN = {'LR': 'LR', 'RF': 'RF', 'HGB': 'GB'}


def f3(x):
    return f'{x:.2f}'


def grid_table(D, name):
    rows = [f'| Rule | α | δ | ' + ' | '.join(f'{MN[m]}: within / certified / recall' for m in MODS) + ' |',
            '|:--|--:|--:|' + ':--:|' * len(MODS)]
    for a in ('0.1', '0.2', '0.3'):
        for d in ('0.05', '0.1', '0.2'):
            cells = []
            for m in MODS:
                v = D[m]['EviAlert-F']['grid_ltt'][f'{a}_{d}']
                cells.append(f"{f3(v['within'])} / {f3(v['certified'])} / {f3(v['recall'])}")
            rows.append(f'| Certification | {a} | {d} | ' + ' | '.join(cells) + ' |')
    for a in ('0.1', '0.2', '0.3'):
        cells = []
        for m in MODS:
            v = D[m]['EviAlert-F']['grid_crc'][a]
            cells.append(f"{f3(v['within'])} / {f3(v['certified'])} / {f3(v['recall'])}")
        rows.append(f'| Conformal risk control | {a} | – | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def n_table(D):
    sizes = sorted(int(k) for k in D['LR']['EviAlert-F']['n_ltt'])
    rows = ['| Rule | ' + ' | '.join(f'n = {n}' for n in sizes) + ' |', '|:--|' + ':--:|' * len(sizes)]
    for key, lab in [('n_ltt', 'Certification'), ('n_crc', 'Conformal risk control'), ('n_naive', 'Empirical pooled rate')]:
        cells = []
        for n in sizes:
            w = np.mean([D[m]['EviAlert-F'][key][str(n)]['within'] for m in MODS])
            r = np.mean([D[m]['EviAlert-F'][key][str(n)]['recall'] for m in MODS])
            c = np.mean([D[m]['EviAlert-F'][key][str(n)]['certified'] for m in MODS])
            cells.append(f'{f3(w)} / {f3(r)}' + ('' if c > 0.999 else f' ({c:.0%} cert.)'))
        rows.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def tempering_table(D):
    rows = ['| Classifier | Method | R@0.5 | R@1 | R@2 |', '|:--|:--|--:|--:|--:|']
    for m in MODS:
        for meth, lab in [('MovAvg-Cal', 'MovAvg-Cal'), ('MovAvg-Cal-wide', 'MovAvg-Cal, wide grid'), ('EviAlert-F', 'EviAlert'), ('EviAlert-F-g1', 'EviAlert without tempering')]:
            pr = D[m][meth]['person']
            vals = [np.mean([v[f'recall@{x}'] for v in pr.values()]) for x in (0.5, 1.0, 2.0)]
            rows.append(f'| {MN[m]} | {lab} | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    return '\n'.join(rows)


def per_nurse(R, key, lab, fmt):
    meta = R['meta']; ids = sorted(meta)
    rows = ['| Nurse | Hours | Sessions | Events | ' + ' | '.join(f'{MN[m]}: MovAvg-Cal / EviAlert' for m in MODS) + ' |',
            '|:--|--:|--:|--:|' + ':--:|' * len(MODS)]
    for p in ids:
        cells = []
        for m in MODS:
            S = R['models'][m]['summary']; a = S['MovAvg-Cal']['person'].get(p); b = S['EviAlert-F']['person'].get(p)
            cells.append('–' if a is None else f"{fmt(a[key])} / {fmt(b[key])}")
        rows.append(f"| {p} | {meta[p]['hours']:.1f} | {meta[p]['sessions']} | {meta[p]['events']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def tests_table(T):
    rows = ['| Hypothesis | Classifier | EviAlert | MovAvg-Cal | Nurses higher / lower | W | p | p (Holm) |', '|:--|:--|--:|--:|:--:|--:|--:|--:|']
    for h, lab, f in [('H1', 'H1: per-nurse recall at 1 false alert per hour', '{:.3f}'), ('H2', 'H2: stale alerts per 100 h', '{:.2f}')]:
        for m in MODS:
            r = T[h][m]
            rows.append(f"| {lab} | {MN[m]} | {f.format(r['mean_x'])} | {f.format(r['mean_y'])} | {r.get('n_pos', 0)} / {r.get('n_neg', 0)} | {r['stat'] if r['stat'] is None else round(r['stat'], 1)} | {r['p']:.4f} | {r['p_holm']:.4f} |")
    rows += ['', '| H3 | Classifier | Nurses within budget | Share | Exact 95% CI | Folds certified | Recall at certified threshold |', '|:--|:--|--:|--:|:--:|--:|--:|']
    for m in MODS:
        r = T['H3'][m]
        rows.append(f"| H3 | {MN[m]} | {r['within']} of {r['n']} | {r['share']:.3f} | {r['ci'][0]:.3f}–{r['ci'][1]:.3f} | {r['certification_rate']:.0%} | {r['recall']:.3f} |")
    return '\n'.join(rows)


if __name__ == '__main__':
    W = json.load(open(sys.argv[1]))
    N = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    R = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    print('### WESAD grid\n' + grid_table(W, 'WESAD') + '\n')
    print('### WESAD n\n' + n_table(W) + '\n')
    print('### WESAD tempering\n' + tempering_table(W) + '\n')
    if N:
        print('### Nurse grid\n' + grid_table(N, 'Nurses') + '\n')
        print('### Nurse n\n' + n_table(N) + '\n')
        print('### Nurse tempering\n' + tempering_table(N) + '\n')
    if R:
        print('### Per nurse recall\n' + per_nurse(R, 'recall@1.0', 'R@1', lambda v: f'{v:.2f}') + '\n')
        print('### Per nurse stale\n' + per_nurse(R, 'gboot_stale100', 'stale', lambda v: f'{v:.1f}') + '\n')
    if len(sys.argv) > 4:
        print('### Tests\n' + tests_table(json.load(open(sys.argv[4]))) + '\n')
