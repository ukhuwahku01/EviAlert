"""Exploratory analyses on per-person real streams (added after the design was frozen; not part of H1-H3).

Re-runs the leave-one-person-out protocol of evialert/realstream.py with the frozen functions, but keeps the
raw per-threshold quantities so that further threshold rules can be applied offline:
  * calibration persons: false-alert rate of every person at every threshold (k x n)
  * held-out person: false alerts, hours, detections, events and stale alerts at every threshold
Additional method: 'EviAlert-F-g1' = the final filter without tempering (gamma = 1).
Additional rules: person-level conformal risk control (CRC; marginal guarantee) and Learn-then-Test
certification for a grid of (alpha, delta) and for random subsets of n calibration persons.
"""
import numpy as np
from scipy import stats
from evialert import core
from evialert import realstream as RS
from evialert.experiments import recall_at, GRID_LOGODDS, GAMMAS, W_DEFAULT, method_scores

# Post hoc (added after the first nurse results had been inspected): the probability grid of the moving-average
# baselines (0.30-0.995, fixed on the infant data) does not reach low enough for a rare target, so a version
# whose grid is extended downwards (log-spaced from 0.005; a superset of the original grid) is evaluated.
GRID_WIDE = np.unique(np.concatenate([np.round(np.geomspace(0.005, 0.30, 100), 6), np.round(np.linspace(0.30, 0.995, 50), 4)]))  # superset of GRID01
OVERRIDE = {'MovAvg-Cal-wide': ('MovAvg-Cal', GRID_WIDE)}


def fast_run_policy(E, usable, h_high, window=6, rho=0.5, refractory=24, gate=True, accumulate=False):
    """Same High-Alert onsets as core.run_policy for accumulate=False (checked in test_fast_policy),
    with a time-major layout and without the status array. Returns onset (k, n, T) and insuff (n, T)."""
    assert not accumulate
    h = np.atleast_1d(np.asarray(h_high, float)); k = len(h); n, T = E.shape
    refr = np.zeros((k, n), int); onset = np.zeros((T, k, n), bool); insuff = np.zeros((n, T), bool)
    cum_use = np.concatenate([np.zeros((n, 1)), np.cumsum(usable, 1)], 1)
    for t in range(T):
        lo = max(0, t - window + 1)
        frac = (cum_use[:, t + 1] - cum_use[:, lo]) / (t + 1 - lo)
        ins = (frac < rho) if gate else np.zeros(n, bool)
        insuff[:, t] = ins
        fire = (E[:, t][None, :] >= h[:, None]) & (refr == 0) & ~ins[None, :]
        onset[t] = fire
        refr = np.where(fire, refractory, np.maximum(refr - 1, 0))
    return np.ascontiguousarray(onset.transpose(1, 2, 0)), insuff


def _run_fast(m, Praw, Pcal, sess, prior, tau, target_idx, filt, rho, refractory):
    base, grid = OVERRIDE.get(m, (m, None))
    E, use, g0, acc, gate = method_scores(base, Praw, Pcal, sess, prior, tau, target=target_idx, filt=filt)
    if grid is None:
        grid = g0
    if base == 'EviAlert-W-noClip':
        grid = np.round(np.linspace(0.3, 90, 150), 4)
    on, ins = fast_run_policy(E, use, grid, window=W_DEFAULT, rho=rho, refractory=refractory, gate=gate, accumulate=acc)
    return grid, on, ins


def lopo_raw(data, methods, target_idx, grace=60, rho=0.5, refractory=24):
    pids = sorted(data); folds = []
    for held in pids:
        cal = [p for p in pids if p != held]
        cal_streams = [s for p in cal for s in data[p]]; owner = np.array([cal.index(p) for p in cal for _ in data[p]])
        ev_streams = data[held]
        Pc = np.vstack([s['P'][s['y'] >= 0] for s in cal_streams]); yc = np.concatenate([s['y'][s['y'] >= 0] for s in cal_streams])
        Tst = core.fit_temperature(Pc, yc); prior = float(np.isin(yc, target_idx).mean()); tau = float(np.quantile(Pc.max(1), 0.30))
        Praw_c, sess_c, Lc, fam_c, tgt_c = RS._batch(cal_streams); Pcal_c = core.temperature_scale(Praw_c, Tst)
        Praw_e, sess_e, Le, fam_e, tgt_e = RS._batch(ev_streams); Pcal_e = core.temperature_scale(Praw_e, Tst)
        ev_c = [s['events'] for s in cal_streams]; ev_e = [s['events'] for s in ev_streams]
        a, b = core.fit_transitions([t[:l] for t, l in zip(tgt_c, Lc)])
        ell_c = core.evidence(Pcal_c[sess_c['img']], target_idx, prior, None)
        best = None
        for g in GAMMAS:                               # identical to realstream.lopo
            Lg = core.filter_logodds(ell_c, sess_c['usable'], a, b, g)
            on_, _ = fast_run_policy(Lg, sess_c['usable'], GRID_LOGODDS, window=W_DEFAULT, rho=rho, refractory=refractory, gate=True, accumulate=False)
            fa_, hr_, dt_, tt_, _ = RS._score(on_, Lc, ev_c, fam_c, grace)
            rc = fa_.sum(1) / hr_.sum(); rr = dt_ / max(tt_, 1)
            sc = np.mean([recall_at(rc, rr, x) for x in (0.5, 1.0, 2.0)])
            if best is None or sc > best[1] + 1e-12:
                best = (g, sc)
        filts = {'EviAlert-F': dict(a=a, b=b, gamma=best[0]), 'EviAlert-F-g1': dict(a=a, b=b, gamma=1.0)}
        fold = dict(held=held, gamma=best[0], n_cal=len(cal))
        valid = np.zeros_like(sess_e['usable'])
        for i, l in enumerate(Le):
            valid[i, :l] = True
        for m in methods:
            base = 'EviAlert-F' if m == 'EviAlert-F-g1' else m
            filt = filts.get(m, filts['EviAlert-F'])
            grid, on_c, _ = _run_fast(base, Praw_c, Pcal_c, sess_c, prior, tau, target_idx, filt, rho, refractory)
            fa_c, hr_c, _, _, _ = RS._score(on_c, Lc, ev_c, fam_c, grace)
            fa_p = np.stack([fa_c[:, owner == i].sum(1) for i in range(len(cal))], 1)
            hr_p = np.array([hr_c[owner == i].sum() for i in range(len(cal))])
            grid, on_e, ins_e = _run_fast(base, Praw_e, Pcal_e, sess_e, prior, tau, target_idx, filt, rho, refractory)
            fa_e, hr_e, det_e, tot_e, _ = RS._score(on_e, Le, ev_e, fam_e, grace)
            stale = (on_e & ~sess_e['usable'][None] & valid[None]).reshape(len(grid), -1).sum(1)
            person = dict(fa=fa_e.sum(1), hours=float(hr_e.sum()), det=np.asarray(det_e, float), tot=int(tot_e))
            fold[m] = dict(grid=np.asarray(grid), cal_fa=fa_p, cal_hours=hr_p, stale=stale, insuff=float(ins_e[valid].mean()), **person)
        folds.append(fold)
    return folds


def _fixed_sequence(grid, ok):
    """Walk from the most conservative threshold while ok holds; return (index, certified)."""
    order = np.argsort(grid)[::-1]
    if not ok[order[0]]:
        return int(order[0]), False
    j_sel = order[0]
    for j in order:
        if ok[j]:
            j_sel = j
        else:
            break
    return int(j_sel), True


def rule_index(r, rule, budget=1.0, alpha=0.2, delta=0.1, subset=None):
    R = r['cal_fa'] / np.maximum(r['cal_hours'], 1e-9)          # (k, n)
    if subset is not None:
        R = R[:, subset]
    n = R.shape[1]; X = (R > budget + 1e-12).sum(1)
    if rule == 'ltt':
        ok = stats.binom.cdf(X, n, alpha) <= delta
    elif rule == 'crc':                                          # E[1{new person above budget}] <= alpha
        ok = (X + 1.0) / (n + 1.0) <= alpha + 1e-12
    elif rule == 'naive':
        fa = r['cal_fa'] if subset is None else r['cal_fa'][:, subset]
        hr = r['cal_hours'] if subset is None else r['cal_hours'][subset]
        ok = fa.sum(1) / hr.sum() <= budget
    else:
        raise ValueError(rule)
    return _fixed_sequence(r['grid'], ok)


def evaluate_rule(folds, method, rule, budget=1.0, **kw):
    within, cert, det, tot, fa, hrs, stale = [], [], 0.0, 0, 0.0, 0.0, 0.0
    for f in folds:
        r = f[method]; j, c = rule_index(r, rule, budget, **kw)
        rate = r['fa'][j] / max(r['hours'], 1e-9)
        within.append(rate <= budget + 1e-12); cert.append(c)
        det += r['det'][j]; tot += r['tot']; fa += r['fa'][j]; hrs += r['hours']; stale += r['stale'][j]
    return dict(within=float(np.mean(within)), certified=float(np.mean(cert)), recall=det / max(tot, 1),
                rate=fa / hrs, stale_per100h=stale / hrs * 100)


def subset_analysis(folds, method, rule, sizes, n_draw=30, seed=0, budget=1.0, **kw):
    rng = np.random.default_rng(seed); out = {}
    for n in sizes:
        W, C, D, T = [], [], 0.0, 0
        for f in folds:
            r = f[method]; N = r['cal_fa'].shape[1]
            draws = [np.arange(N)] if n >= N else [rng.choice(N, n, replace=False) for _ in range(n_draw)]
            for sub in draws:
                j, c = rule_index(r, rule, budget, subset=sub, **kw)
                W.append(r['fa'][j] / max(r['hours'], 1e-9) <= budget + 1e-12); C.append(c)
                D += r['det'][j] / len(draws); T += r['tot'] / len(draws)
        out[int(n)] = dict(within=float(np.mean(W)), certified=float(np.mean(C)), recall=D / max(T, 1e-9))
    return out


def analyse(folds, methods, alphas=(0.1, 0.2, 0.3), deltas=(0.05, 0.1, 0.2), sizes=(5, 7, 9, 11, 14)):
    out = {}
    for m in methods:
        o = dict(ltt=evaluate_rule(folds, m, 'ltt'), crc=evaluate_rule(folds, m, 'crc'), naive=evaluate_rule(folds, m, 'naive'))
        o['grid_ltt'] = {f'{a}_{d}': evaluate_rule(folds, m, 'ltt', alpha=a, delta=d) for a in alphas for d in deltas}
        o['grid_crc'] = {f'{a}': evaluate_rule(folds, m, 'crc', alpha=a) for a in alphas}
        o['n_ltt'] = subset_analysis(folds, m, 'ltt', sizes)
        o['n_crc'] = subset_analysis(folds, m, 'crc', sizes)
        o['n_naive'] = subset_analysis(folds, m, 'naive', sizes)
        # operating-curve recall (pooled over held-out persons) for reference
        fa = np.sum([f[m]['fa'] for f in folds], 0); hrs = sum(f[m]['hours'] for f in folds)
        det = np.sum([f[m]['det'] for f in folds], 0); tot = sum(f[m]['tot'] for f in folds)
        o.update({f'recall@{x}': recall_at(fa / hrs, det / max(tot, 1), x) for x in (0.5, 1.0, 2.0)})
        o['person'] = {f['held']: {f'recall@{x}': recall_at(f[m]['fa'] / f[m]['hours'], f[m]['det'] / f[m]['tot'], x) for x in (0.5, 1.0, 2.0)}
                       for f in folds if f[m]['tot'] > 0 and f[m]['hours'] > 0}
        o['insuff'] = float(np.mean([f[m]['insuff'] for f in folds]))
        out[m] = o
    out['gamma'] = [f['gamma'] for f in folds]
    return out
