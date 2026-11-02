"""Pre-registered hypothesis tests H1-H3 for the nurse confirmatory evaluation (see PREREGISTRATION.md).
Written and dry-run on WESAD results before any access to the nurse data.
Usage: python tests_nurse.py <nurse_results.json> <out_json>"""
import sys, json
import numpy as np
from scipy import stats

NEW, REF = 'EviAlert-F', 'MovAvg-Cal'
ALPHA = 0.2


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj.tolist()


def wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = int((np.abs(d) > 1e-12).sum())
    if nz == 0:
        return dict(n=len(d), nonzero=0, stat=None, p=1.0, median_diff=0.0, mean_x=float(np.mean(x)), mean_y=float(np.mean(y)))
    r = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    return dict(n=len(d), nonzero=nz, stat=float(r.statistic), p=float(r.pvalue), median_diff=float(np.median(d)),
                mean_x=float(np.mean(x)), mean_y=float(np.mean(y)), n_pos=int((d > 1e-12).sum()), n_neg=int((d < -1e-12).sum()))


def analyse(models):
    out = dict(H1={}, H2={}, H3={}, secondary={})
    for mod, res in models.items():
        S = res['summary']; pn, pr = S[NEW]['person'], S[REF]['person']
        ids = sorted(set(pn) & set(pr))
        out['H1'][mod] = wilcoxon([pn[i]['recall@1.0'] for i in ids], [pr[i]['recall@1.0'] for i in ids])
        out['H2'][mod] = wilcoxon([pn[i]['gboot_stale100'] for i in ids], [pr[i]['gboot_stale100'] for i in ids])
        n = len(res['filt']) if 'filt' in res else None
        c = S[NEW]['certified']; k = int(round(c['within'] * n)) if n else None
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method='exact') if n else None
        out['H3'][mod] = dict(n=n, within=k, share=c['within'], ci=[ci.low, ci.high] if ci else None,
                              meets=bool(c['within'] >= 1 - ALPHA), certification_rate=c['feasible'],
                              recall=c['recall'], rate=c['rate'],
                              other_rules={r: dict(share=S[NEW][r]['within'], recall=S[NEW][r]['recall']) for r in ('naive', 'poisson', 'gboot')},
                              movavgcal={r: dict(share=S[REF][r]['within'], recall=S[REF][r]['recall'], feasible=S[REF][r]['feasible']) for r in ('naive', 'poisson', 'gboot', 'certified')})
        out['secondary'][mod] = {m: {k2: v for k2, v in S[m].items() if k2 != 'person'} for m in S}
        if 'auroc' in res:
            out['secondary'][mod]['auroc'] = res['auroc']
    for h in ('H1', 'H2'):
        mods = list(out[h]); adj = holm([out[h][m]['p'] for m in mods])
        for m, a in zip(mods, adj):
            out[h][m]['p_holm'] = a
    return out


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    res = analyse(d['models'])
    if 'meta' in d:
        res['meta'] = d['meta']
    json.dump(res, open(sys.argv[2], 'w'), indent=1)
    for h in ('H1', 'H2'):
        for m, r in res[h].items():
            print(h, m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for m, r in res['H3'].items():
        print('H3', m, 'share', round(r['share'], 3), 'CI', [round(x, 3) for x in r['ci']] if r['ci'] else None,
              'cert rate', r['certification_rate'], 'recall', round(r['recall'], 3))
