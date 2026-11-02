"""Experiment runner: EviAlert vs baselines and ablations on semi-synthetic sessions built from
out-of-fold InfantFER predictions (identity-grouped 5-fold CV, five backbones).

Two questions are evaluated separately:
  (A) Detection quality of each alert statistic: operating curves on evaluation sessions
      -> recall at a fixed false-alert rate (FA/h), detection delay, stale alerts.
  (B) Threshold calibration: thresholds chosen on calibration sessions only (disjoint child
      identities) with three rules -> realized FA/h on evaluation sessions and budget compliance.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evialert import core
from evialert.simulate import make_sessions, DISTRESS
DISTRESS_DEFAULT = DISTRESS

BACKBONES = ['resnet50', 'vit_b16', 'convnext_t', 'swin_t', 'effnet_b0']
W_DEFAULT, CLIP_DEFAULT, KAPPA_DEFAULT = 6, 3.0, 0.5     # fixed on the development split (seed 0)


def load_oof(runs_dir, index_csv, model):
    ci = pd.read_csv(index_csv)
    n = len(ci); P = np.zeros((n, 4)); y = np.full(n, -1)
    for f in range(5):
        d = json.load(open(os.path.join(runs_dir, f'grouped_{model}_fold{f}.json')))
        P[d['test_idx']] = d['test_prob']; y[d['test_idx']] = d['test_true']
    assert (y >= 0).all() and (y == ci.y.values).all()
    return P, y, ci.group.values


def locf(X, usable, fill=0.0):
    """Carry the last usable value forward over unusable steps (common practice for baselines)."""
    X = X.copy(); n, T = usable.shape
    prev = np.full(n, fill, float)
    for t in range(T):
        prev = np.where(usable[:, t], X[:, t], prev)
        X[:, t] = prev
    return X


def window_mean(V, W, valid=None):
    n, T = V.shape
    valid = np.ones_like(V, bool) if valid is None else valid
    Vz = np.where(valid, V, 0.0)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(Vz, 1)], 1)
    cn = np.concatenate([np.zeros((n, 1)), np.cumsum(valid, 1)], 1)
    out = np.zeros((n, T))
    for t in range(T):
        lo = max(0, t - W + 1)
        s = cs[:, t + 1] - cs[:, lo]; c = cn[:, t + 1] - cn[:, lo]
        out[:, t] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    return out


def window_sum(V, W, valid):
    n, T = V.shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(np.where(valid, V, 0.0), 1)], 1)
    out = np.zeros((n, T))
    for t in range(T):
        out[:, t] = cs[:, t + 1] - cs[:, max(0, t - W + 1)]
    return out


GRID01 = np.round(np.linspace(0.30, 0.995, 50), 4)
GRID_MV = np.array([2, 3, 4, 5, 6]) / 6.0 - 1e-9
GRID_CUSUM = np.round(np.geomspace(0.3, 40, 60), 4)
GRID_SUM = np.round(np.linspace(0.3, 18, 80), 4)
GRID_LOGODDS = np.round(np.linspace(-6, 18, 150), 4)    # EviAlert v2 (posterior log-odds)
GRID_Z = np.round(np.linspace(-1, 40, 165), 4)             # EviAlert v2 multiscale scan (standardized)
SCALES = (3, 6, 12, 24)
GAMMAS = (1.0, 0.5, 0.25, 0.125)                         # candidate tempering exponents (largest first)

MAIN = ['Frame', 'MajVote', 'MovAvg', 'SelMovAvg', 'MovAvg-Cal', 'EviAlert-W', 'EviAlert-C']
ABL = ['EviAlert-W-noCal', 'EviAlert-W-noClip', 'EviAlert-W-noGate']


def method_scores(name, P_raw, P_cal, sess, prior_cal, tau, W=W_DEFAULT, clip=CLIP_DEFAULT, kappa=KAPPA_DEFAULT,
                  target=None, filt=None):
    """Returns (E, usable, grid, accumulate, gate). Missing frames: LOCF for baselines, abstention for EviAlert."""
    DISTRESS = target if target is not None else DISTRESS_DEFAULT
    img = sess['img']; use = sess['usable']
    Praw = P_raw[img]; Pcal = P_cal[img]
    if name == 'Frame':
        return locf(core.target_prob(Praw, DISTRESS), use), use, GRID01, False, False
    if name == 'MajVote':
        lab = locf(np.isin(Praw.argmax(-1), DISTRESS).astype(float), use)
        return window_mean(lab, W), use, GRID_MV, False, False
    if name == 'MovAvg':
        return window_mean(locf(core.target_prob(Praw, DISTRESS), use), W), use, GRID01, False, False
    if name == 'MovAvg-Cal':
        return window_mean(locf(core.target_prob(Pcal, DISTRESS), use), W), use, GRID01, False, False
    if name == 'SelMovAvg':
        valid = use & (Praw.max(-1) >= tau)
        return window_mean(core.target_prob(Praw, DISTRESS), W, valid), use, GRID01, False, False
    # EviAlert family: evidence = clipped, prior-corrected log-likelihood ratio from calibrated posteriors
    if name == 'EviAlert-W':
        E = core.evidence(Pcal, DISTRESS, prior_cal, clip)
        return window_sum(E, W, use), use, GRID_SUM, False, True
    if name == 'EviAlert-C':
        return core.evidence(Pcal, DISTRESS, prior_cal, clip) - kappa, use, GRID_CUSUM, True, True
    if name == 'EviAlert-W-noCal':
        E = core.evidence(Praw, DISTRESS, prior_cal, clip)
        return window_sum(E, W, use), use, GRID_SUM, False, True
    if name == 'EviAlert-MS':     # v2: multiscale scan of abstaining evidence sums, standardized on calibration null
        E = core.evidence(Pcal, DISTRESS, prior_cal, None)
        Z = np.full(E.shape, -np.inf)
        for w_, (mu_, sd_) in zip(SCALES, filt['null']):
            Z = np.maximum(Z, (window_sum(E, w_, use) - mu_) / sd_)
        return Z, use, GRID_Z, False, True
    if name == 'EviAlert-F':      # v2: tempered two-state filter on unclipped evidence (filt = dict(a, b, gamma))
        L = core.filter_logodds(core.evidence(Pcal, DISTRESS, prior_cal, None), use, filt['a'], filt['b'], filt['gamma'])
        return L, use, GRID_LOGODDS, False, True
    if name == 'EviAlert-W-noClip':
        E = core.evidence(Pcal, DISTRESS, prior_cal, None)
        return window_sum(E, W, use), use, GRID_SUM, False, True
    if name == 'EviAlert-W-noGate':
        E = locf(core.evidence(Pcal, DISTRESS, prior_cal, clip), use)
        return window_sum(E, W, np.ones_like(use)), np.ones_like(use), GRID_SUM, False, False
    raise ValueError(name)


def _curves(name, P, P_cal, S, prior_cal, tau, rho, refractory, kw):
    E, use, grid, acc, gate = method_scores(name, P, P_cal, S, prior_cal, tau, **kw)
    on, st = core.run_policy(E, use, grid, window=kw.get('W', W_DEFAULT), rho=rho, refractory=refractory,
                             gate=gate, accumulate=acc)
    return grid, on, st, acc


def recall_at(fa_rate, recall, x):
    ok = fa_rate <= x + 1e-12
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate_split(P, y, groups, seed, methods, budget=1.0, delta=0.05, n_cal=300, n_eval=600,
                   n_boot=20, n_per_boot=100, cfg_cal=None, cfg_eval=None, refractory=24, rho=0.5,
                   kw=None, fa_points=(0.5, 1.0, 2.0), keep_curves=False):
    kw = kw or {}
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); rng.shuffle(ug)
    cal_g = ug[: len(ug) // 2]
    is_cal = np.isin(groups, cal_g)
    cal_idx = np.where(is_cal)[0]; ev_idx = np.where(~is_cal)[0]
    T_star = core.fit_temperature(P[cal_idx], y[cal_idx])
    P_cal = core.temperature_scale(P, T_star)
    prior_cal = float(np.isin(y[cal_idx], DISTRESS).mean())
    tau = np.quantile(P[cal_idx].max(1), 0.30)
    frame = dict(T=T_star, ece_raw=core.ece(P[ev_idx], y[ev_idx]), ece_cal=core.ece(P_cal[ev_idx], y[ev_idx]),
                 nll_raw=core.nll(P[ev_idx], y[ev_idx]), nll_cal=core.nll(P_cal[ev_idx], y[ev_idx]),
                 brier_raw=core.brier(P[ev_idx], y[ev_idx]), brier_cal=core.brier(P_cal[ev_idx], y[ev_idx]))
    S_cal = make_sessions(cal_idx, y, groups, n_cal, rng, cfg_cal)
    # identity-level bootstrap of the calibration pool
    gid_of = {g: np.where(groups == g)[0] for g in cal_g}
    S_boot = []
    for b in range(n_boot):
        gb = rng.choice(cal_g, size=len(cal_g), replace=True)
        pool = np.concatenate([gid_of[g] for g in gb])
        S_boot.append(make_sessions(pool, y, groups, n_per_boot, rng, cfg_cal))
    S_ev = make_sessions(ev_idx, y, groups, n_eval, rng, cfg_eval)
    if any(m in ('EviAlert-F', 'EviAlert-MS') for m in methods):
        if kw.get('filt_target', 'sustained') == 'sustained':      # alert-worthy (sustained) episodes as the target state
            st_cal = np.zeros(S_cal['state'].shape, bool)
            for i_, ev_ in enumerate(S_cal['events']):
                for (s_, e_) in ev_:
                    st_cal[i_, s_:e_] = True
        else:
            st_cal = np.isin(S_cal['state'], DISTRESS)
        fa_, fb_ = core.fit_transitions(list(st_cal))
        ell_cal = core.evidence(P_cal[S_cal['img']], DISTRESS, prior_cal, None)
        # decision-focused tempering: the gamma whose calibration operating curve has the highest mean recall
        # at 0.5, 1 and 2 false alerts per hour (calibration sessions only)
        best_ = None
        for g_ in GAMMAS:
            Lc = core.filter_logodds(ell_cal, S_cal['usable'], fa_, fb_, g_)
            onc, _ = core.run_policy(Lc, S_cal['usable'], GRID_LOGODDS, window=kw.get('W', W_DEFAULT), rho=rho,
                                     refractory=refractory, gate=True, accumulate=False)
            scc = core.score_alerts(onc, S_cal['events'], S_cal['cfg']['T'], S_cal['cfg']['step_s'], 6)
            rc = scc['fa'] / scc['hours']; rr = scc['det'] / max(scc['tot'], 1)
            sc_ = np.mean([recall_at(rc, rr, x_) for x_ in (0.5, 1.0, 2.0)])
            if best_ is None or sc_ > best_[1] + 1e-12:
                best_ = (g_, sc_)
        fg_ = best_[0]
        kw = {k_: v_ for k_, v_ in kw.items() if k_ != 'filt_target'}
        null_ = []
        for w_ in SCALES:                       # null distribution: windows entirely outside sustained episodes
            Sw = window_sum(ell_cal, w_, S_cal['usable'])
            inside = np.zeros_like(st_cal)
            for i_, ev_ in enumerate(S_cal['events']):
                for (s_, e_) in ev_:
                    inside[i_, s_:min(st_cal.shape[1], e_ + w_)] = True
            vals = Sw[:, w_ - 1:][~inside[:, w_ - 1:]]
            null_.append((float(vals.mean()), float(vals.std() + 1e-9)))
        kw = {**kw, 'filt': dict(a=fa_, b=fb_, gamma=fg_, null=null_)}
    step_s = S_ev['cfg']['step_s']; T = S_ev['cfg']['T']; grace = 6
    out = {}
    for m in methods:
        grid, on_c, _, acc = _curves(m, P, P_cal, S_cal, prior_cal, tau, rho, refractory, kw)
        sc_c = core.score_alerts(on_c, S_cal['events'], T, step_s, grace)
        rate_c = sc_c['fa'] / sc_c['hours']
        boot_rates = []
        for Sb in S_boot:
            _, on_b, _, _ = _curves(m, P, P_cal, Sb, prior_cal, tau, rho, refractory, kw)
            sb = core.score_alerts(on_b, Sb['events'], T, step_s, grace)
            boot_rates.append(sb['fa'] / sb['hours'])
        boot_rates = np.array(boot_rates)
        bounds = dict(naive=rate_c,
                      poisson=np.array([core.poisson_ucb(c, sc_c['hours'], delta) for c in sc_c['fa']]),
                      gboot=np.quantile(boot_rates, 1 - delta, axis=0))
        _, on_e, st_e, _ = _curves(m, P, P_cal, S_ev, prior_cal, tau, rho, refractory, kw)
        sc_e = core.score_alerts(on_e, S_ev['events'], T, step_s, grace)
        rate_e = sc_e['fa'] / sc_e['hours']; rec_e = sc_e['det'] / max(sc_e['tot'], 1)
        res = dict(n_events=int(sc_e['tot']), hours=float(sc_e['hours']))
        for x in fa_points:
            res[f'recall@{x}'] = recall_at(rate_e, rec_e, x)
        # operating point at the evaluation curve's best recall with FA/h <= budget (oracle point)
        ok = np.where(rate_e <= budget + 1e-12)[0]
        j = ok[np.argmax(rec_e[ok])] if len(ok) else int(np.argmin(rate_e))
        d = sc_e['delays'][j]
        res['oracle'] = dict(h=float(grid[j]), fa=float(rate_e[j]), recall=float(rec_e[j]),
                             delay_med=float(np.median(d)) if d else float('nan'),
                             stale=int((on_e[j] & ~S_ev['usable']).sum()),
                             insuff_share=float((st_e[j] == core.INSUFF).mean()))
        for rule, bd in bounds.items():
            h, feas = core.select_threshold(grid, bd, budget)
            j = int(np.where(grid == h)[0][0])
            d = sc_e['delays'][j]
            res[rule] = dict(h=h, feasible=feas, fa=float(rate_e[j]), recall=float(rec_e[j]),
                             delay_med=float(np.median(d)) if d else float('nan'),
                             compliant=bool(rate_e[j] <= budget))
        jg = int(np.where(grid == res['gboot']['h'])[0][0])
        res['gboot']['stale'] = int((on_e[jg] & ~S_ev['usable']).sum())
        res['gboot']['insuff_share'] = float((st_e[jg] == core.INSUFF).mean())
        if m.startswith('EviAlert'):   # watchlist lead time at the gboot operating point (h_W = h_H / 2)
            h = res['gboot']['h']
            E, use, _, acc_, gate = method_scores(m, P, P_cal, S_ev, prior_cal, tau, **kw)
            hw = h - 2.0 if m == 'EviAlert-F' else h / 2
            on1, st1 = core.run_policy(E, use, [h], h_watch=[hw], window=kw.get('W', W_DEFAULT), rho=rho,
                                       refractory=refractory, gate=gate, accumulate=acc_)
            leads = []
            for i, ev in enumerate(S_ev['events']):
                for (s, e) in ev:
                    seg = on1[0, i, s:min(T, e + grace)]
                    if seg.any():
                        ta = s + int(seg.argmax())
                        w = np.where(st1[0, i, s:ta + 1] == core.WATCH)[0]
                        leads.append((ta - (s + w[0])) * step_s if len(w) else 0)
            res['watch_lead_med'] = float(np.median(leads)) if leads else float('nan')
            res['watch_share'] = float((st1[0] == core.WATCH).mean())
        if keep_curves:
            res['curve'] = dict(fa=rate_e.tolist(), recall=rec_e.tolist(), grid=grid.tolist())
        out[m] = res
    meta = dict(seed=seed, n_cal_img=int(len(cal_idx)), n_eval_img=int(len(ev_idx)), prior_cal=prior_cal,
                filt=kw.get('filt'),
                same_identity_share=S_ev['same_identity_share'],
                distress_time_share=float(np.isin(S_ev['state'], DISTRESS).mean()),
                missing_share=float(1 - S_ev['usable'].mean()),
                events_per_hour=float(sum(len(e) for e in S_ev['events']) / (n_eval * T * step_s / 3600)))
    return frame, out, meta
