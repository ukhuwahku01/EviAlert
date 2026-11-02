"""Leave-one-person-out evaluation of alerting policies on real, per-person monitoring streams.

A dataset is a dict  pid -> list of streams; each stream is a dict with
  P       (T, K) out-of-person class probabilities from the base classifier
  y       (T,)   class index for frames with a known class, -1 otherwise (used for T* and the prior)
  use     (T,)   bool, usable frames (quality check, optionally with simulated dropouts)
  events  list of (start, end) alert-worthy episodes (end exclusive)
  famask  (T,)   bool, steps on which false alerts are counted and monitored time accrues
  target  (T,)   bool, target state (for the filter's transition hazards)
"""
import numpy as np
from . import core
from .experiments import method_scores, recall_at, GRID_LOGODDS, GAMMAS, W_DEFAULT

STEP_S = 5.0


def _batch(streams):
    L = np.array([len(s['use']) for s in streams]); Tm = int(L.max())
    P = np.vstack([s['P'] for s in streams]); off = np.concatenate([[0], np.cumsum(L)[:-1]])
    img = np.zeros((len(streams), Tm), int); use = np.zeros((len(streams), Tm), bool)
    fam = np.zeros((len(streams), Tm), bool); tgt = np.zeros((len(streams), Tm), bool)
    for i, s in enumerate(streams):
        img[i, :L[i]] = off[i] + np.arange(L[i]); img[i, L[i]:] = off[i] + L[i] - 1
        use[i, :L[i]] = s['use']; fam[i, :L[i]] = s['famask']; tgt[i, :L[i]] = s['target']
    return P, dict(img=img, usable=use), L, fam, tgt


def _score(on, L, events, fam, grace):
    """Per-stream false alerts and hours; detections and delays pooled."""
    k, n, T = on.shape
    fa = np.zeros((k, n)); hrs = np.zeros(n); det = np.zeros(k); tot = 0; delays = [[] for _ in range(k)]
    for i in range(n):
        ok = np.zeros((k, T), bool)
        for (a, b) in events[i]:
            tot += 1
            ok[:, a:min(L[i], b + grace)] = True
            hit = on[:, i, a:min(L[i], b + grace)]; anyh = hit.any(1); det += anyh; first = hit.argmax(1)
            for j in np.where(anyh)[0]:
                delays[j].append(first[j] * STEP_S)
        v = fam[i].copy(); v[L[i]:] = False
        fa[:, i] = (on[:, i, :] & ~ok & v[None, :]).sum(1); hrs[i] = v.sum() * STEP_S / 3600.0
    return fa, hrs, det, tot, delays


def _run(m, Praw, Pcal, sess, prior, tau, target_idx, filt, rho, refractory):
    E, use, grid, acc, gate = method_scores(m, Praw, Pcal, sess, prior, tau, target=target_idx, filt=filt)
    if m == 'EviAlert-W-noClip':
        grid = np.round(np.linspace(0.3, 90, 150), 4)
    on, st = core.run_policy(E, use, grid, window=W_DEFAULT, rho=rho, refractory=refractory, gate=gate, accumulate=acc)
    return grid, on, st


def lopo(data, methods, target_idx, budget=1.0, delta=0.05, n_boot=200, alpha=0.2, delta_cert=0.1,
         grace=60, rho=0.5, refractory=24, seed=0):
    pids = sorted(data); rng = np.random.default_rng(seed)
    folds = []
    for held in pids:
        cal = [p for p in pids if p != held]
        cal_streams = [s for p in cal for s in data[p]]; owner = np.array([cal.index(p) for p in cal for _ in data[p]])
        ev_streams = data[held]
        Pc = np.vstack([s['P'][s['y'] >= 0] for s in cal_streams]); yc = np.concatenate([s['y'][s['y'] >= 0] for s in cal_streams])
        Tst = core.fit_temperature(Pc, yc); prior = float(np.isin(yc, target_idx).mean()); tau = float(np.quantile(Pc.max(1), 0.30))
        Praw_c, sess_c, Lc, fam_c, tgt_c = _batch(cal_streams); Pcal_c = core.temperature_scale(Praw_c, Tst)
        Praw_e, sess_e, Le, fam_e, tgt_e = _batch(ev_streams); Pcal_e = core.temperature_scale(Praw_e, Tst)
        ev_c = [s['events'] for s in cal_streams]; ev_e = [s['events'] for s in ev_streams]
        # filter parameters (calibration persons only)
        a, b = core.fit_transitions([t[:l] for t, l in zip(tgt_c, Lc)])
        ell_c = core.evidence(Pcal_c[sess_c['img']], target_idx, prior, None)
        best = None
        for g in GAMMAS:
            Lg = core.filter_logodds(ell_c, sess_c['usable'], a, b, g)
            on_, _ = core.run_policy(Lg, sess_c['usable'], GRID_LOGODDS, window=W_DEFAULT, rho=rho, refractory=refractory, gate=True, accumulate=False)
            fa_, hr_, dt_, tt_, _ = _score(on_, Lc, ev_c, fam_c, grace)
            rc = fa_.sum(1) / hr_.sum(); rr = dt_ / max(tt_, 1)
            sc = np.mean([recall_at(rc, rr, x) for x in (0.5, 1.0, 2.0)])
            if best is None or sc > best[1] + 1e-12:
                best = (g, sc)
        filt = dict(a=a, b=b, gamma=best[0])
        fold = dict(held=held, T=Tst, prior=prior, filt=filt)
        boots = [rng.choice(len(cal), len(cal), replace=True) for _ in range(n_boot)]
        for m in methods:
            grid, on_c, _ = _run(m, Praw_c, Pcal_c, sess_c, prior, tau, target_idx, filt, rho, refractory)
            fa_c, hr_c, _, _, _ = _score(on_c, Lc, ev_c, fam_c, grace)
            fa_p = np.stack([fa_c[:, owner == i].sum(1) for i in range(len(cal))], 1)      # (k, persons)
            hr_p = np.array([hr_c[owner == i].sum() for i in range(len(cal))])
            rate_p = fa_p / np.maximum(hr_p, 1e-9)
            bounds = dict(naive=fa_p.sum(1) / hr_p.sum(),
                          poisson=np.array([core.poisson_ucb(c, hr_p.sum(), delta) for c in fa_p.sum(1)]),
                          gboot=np.quantile(np.array([fa_p[:, bb].sum(1) / hr_p[bb].sum() for bb in boots]), 1 - delta, axis=0))
            grid, on_e, st_e = _run(m, Praw_e, Pcal_e, sess_e, prior, tau, target_idx, filt, rho, refractory)
            fa_e, hr_e, det_e, tot_e, del_e = _score(on_e, Le, ev_e, fam_e, grace)
            valid = np.zeros_like(sess_e['usable'])
            for i, l in enumerate(Le):
                valid[i, :l] = True
            r = dict(curve_fa=fa_e.sum(1).tolist(), curve_det=det_e.tolist(), hours=float(hr_e.sum()), n_events=int(tot_e),
                     delays=[float(np.median(d)) if d else None for d in del_e])
            sel = {}
            for rule, bd in bounds.items():
                h, feas = core.select_threshold(grid, bd, budget); sel[rule] = (h, feas, None)
            h, cert, nmin = core.certify_threshold(grid, rate_p, budget, alpha, delta_cert); sel['certified'] = (h, cert, nmin)
            for rule, (h, ok, extra) in sel.items():
                j = int(np.where(grid == h)[0][0])
                r[rule] = dict(h=h, feasible=bool(ok), fa=float(fa_e[j].sum()), hours=float(hr_e.sum()),
                               rate=float(fa_e[j].sum() / max(hr_e.sum(), 1e-9)), det=float(det_e[j]), tot=int(tot_e),
                               delay=float(np.median(del_e[j])) if del_e[j] else None,
                               stale=int((on_e[j] & ~sess_e['usable'] & valid).sum()),
                               insuff=float((st_e[j][valid] == core.INSUFF).mean()), n_min=extra)
            fold[m] = r
        folds.append(fold)
    return folds


def summarize(folds, methods, budget=1.0):
    out = {}
    for m in methods:
        fa = np.sum([f[m]['curve_fa'] for f in folds], 0); det = np.sum([f[m]['curve_det'] for f in folds], 0)
        hrs = sum(f[m]['hours'] for f in folds); tot = sum(f[m]['n_events'] for f in folds)
        rate = fa / hrs; rec = det / max(tot, 1)
        o = {f'recall@{x}': recall_at(rate, rec, x) for x in (0.5, 1.0, 2.0)}
        # person-level operating curves: each held-out person's own recall at their own false-alert rate
        o['person'] = {}
        for f in folds:
            r = f[m]
            if r['n_events'] == 0 or r['hours'] <= 0:
                continue
            pr = np.array(r['curve_fa']) / r['hours']; pc = np.array(r['curve_det']) / r['n_events']
            g = r['gboot']
            o['person'][f['held']] = {**{f'recall@{x}': recall_at(pr, pc, x) for x in (0.5, 1.0, 2.0)},
                                      'gboot_recall': g['det'] / max(g['tot'], 1), 'gboot_rate': g['rate'],
                                      'gboot_stale100': g['stale'] / max(g['hours'], 1e-9) * 100,
                                      'cert_rate': r['certified']['rate'], 'cert_recall': r['certified']['det'] / max(r['certified']['tot'], 1),
                                      'certified': r['certified']['feasible'], 'events': r['n_events'], 'hours': r['hours']}
        for x in (0.5, 1.0, 2.0):
            o[f'person_recall@{x}'] = float(np.mean([v[f'recall@{x}'] for v in o['person'].values()])) if o['person'] else float('nan')
        o.update(hours=hrs, events=tot)
        for rule in ['naive', 'poisson', 'gboot', 'certified']:
            R = [f[m][rule] for f in folds]
            o[rule] = dict(within=float(np.mean([r['rate'] <= budget for r in R])),          # share of persons within budget
                           feasible=float(np.mean([r['feasible'] for r in R])),
                           rate=float(sum(r['fa'] for r in R) / sum(r['hours'] for r in R)),
                           recall=float(sum(r['det'] for r in R) / max(sum(r['tot'] for r in R), 1)),
                           stale_per100h=float(sum(r['stale'] for r in R) / sum(r['hours'] for r in R) * 100),
                           insuff=float(np.mean([r['insuff'] for r in R])),
                           delay=float(np.nanmedian([r['delay'] if r['delay'] is not None else np.nan for r in R])))
        out[m] = o
    return out
