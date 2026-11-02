"""EviAlert on real wearable streams (WESAD, wrist E4). Same methods, parameters and grids as the
InfantFER experiments; nothing is tuned on WESAD.

Usage: python run_wesad.py <features_dir> <out_json> [n_splits]
"""
import json, os, sys, time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evialert import core
from evialert.experiments import method_scores, GRID01, GRID_MV, GRID_CUSUM, GRID_SUM, W_DEFAULT, GRID_LOGODDS, GAMMAS, recall_at

SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
CLASSES = [1, 2, 3, 4]                  # baseline, stress, amusement, meditation
D = [1]                                 # index of stress in CLASSES
STEP = 5.0
MOTION_THR = 16.0                       # acc magnitude SD over the last 5 s > 0.25 g (1/64 g units) -> unusable
MIN_EVENT, MERGE_GAP, GRACE = 9, 12, 60 # 45 s minimum episode, merge gaps <= 60 s, 5-min grace after stress
METHODS = ['Frame', 'MajVote', 'MovAvg', 'SelMovAvg', 'MovAvg-Cal', 'EviAlert-C', 'EviAlert-W', 'EviAlert-W-noClip',
           'EviAlert-W-noGate', 'EviAlert-F']
GRID_SUM_WIDE = np.round(np.linspace(0.3, 90, 150), 4)   # unclipped evidence can exceed W * c
GRIDS = {'Frame': GRID01, 'MajVote': GRID_MV, 'MovAvg': GRID01, 'SelMovAvg': GRID01, 'MovAvg-Cal': GRID01,
         'EviAlert-C': GRID_CUSUM, 'EviAlert-W': GRID_SUM, 'EviAlert-W-noGate': GRID_SUM}


def load(feat_dir):
    S = {}
    for s in SUBJECTS:
        p = os.path.join(feat_dir, f'{s}_frames.npz')
        if not os.path.exists(p):
            continue
        d = dict(np.load(p, allow_pickle=False))
        lab = d['label_end']
        known = np.where(np.isin(lab, CLASSES))[0]
        a, b = known.min(), known.max() + 1           # stream = protocol span
        S[s] = {k: (v[a:b] if hasattr(v, '__len__') and len(v) == len(lab) else v) for k, v in d.items()}
    return S


def events_of(lab):
    on = (lab == 2).astype(int)
    idx = np.where(on)[0]
    if len(idx) == 0:
        return []
    runs = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > MERGE_GAP + 1:
            runs.append((s, prev + 1)); s = i
        prev = i
    runs.append((s, prev + 1))
    return [(a, b) for a, b in runs if b - a >= MIN_EVENT]


def loso_probs(S, model):
    subs = list(S)
    P = {}
    for s in subs:
        tr = [t for t in subs if t != s]
        X = np.vstack([S[t]['X'][np.isin(S[t]['label_end'], CLASSES)] for t in tr])
        y = np.concatenate([S[t]['label_end'][np.isin(S[t]['label_end'], CLASSES)] for t in tr])
        if model == 'LR':
            clf = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        elif model == 'RF':
            clf = make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=0))
        else:
            clf = make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(random_state=0))
        clf.fit(X, y)
        assert list(clf.classes_) == CLASSES
        P[s] = clf.predict_proba(S[s]['X'])
    return P


def markov_missing(T, rng, rate=0.10, burst=4):
    if rate <= 0:
        return np.ones(T, bool)
    p_exit = 1.0 / burst; p_enter = rate * p_exit / (1 - rate)
    use = np.ones(T, bool); st = rng.random() < rate
    for t in range(T):
        use[t] = not st
        st = (rng.random() >= p_exit) if st else (rng.random() < p_enter)
    return use


def build(S, subs, P, usable):
    """Pad subjects into one (n, Tmax) batch; img indexes the concatenated probability array."""
    L = np.array([len(S[s]['label_end']) for s in subs]); Tm = L.max()
    Pall = np.vstack([P[s] for s in subs]); off = np.concatenate([[0], np.cumsum(L)[:-1]])
    img = np.zeros((len(subs), Tm), int); use = np.zeros((len(subs), Tm), bool)
    for i, s in enumerate(subs):
        img[i, :L[i]] = off[i] + np.arange(L[i]); img[i, L[i]:] = off[i] + L[i] - 1
        use[i, :L[i]] = usable[s]
    return Pall, dict(img=img, usable=use), L


def score(onset, L, events, step_s=STEP, grace=GRACE, defined=None):
    """defined: optional list of (T_i,) bool arrays; if given, false alerts and monitored time are counted
    only on steps with a defined non-target label (WESAD labels 1, 3, 4)."""
    k, n, T = onset.shape
    fa = np.zeros((k, n)); det = np.zeros(k); tot = sum(len(e) for e in events); delays = [[] for _ in range(k)]
    hours = np.zeros(n)
    for i in range(n):
        ok = np.zeros((k, T), bool); valid = np.zeros(T, bool); valid[:L[i]] = True
        if defined is not None:
            valid[:L[i]] &= defined[i]
        hours[i] = valid.sum() * step_s / 3600.0
        for (a, b) in events[i]:
            ok[:, a:min(L[i], b + grace)] = True
            hit = onset[:, i, a:min(L[i], b + grace)]
            anyh = hit.any(1); det += anyh; first = hit.argmax(1)
            for j in np.where(anyh)[0]:
                delays[j].append(first[j] * step_s)
        fa[:, i] = (onset[:, i, :] & ~ok & valid[None, :]).sum(1)
    return fa, hours, det, tot, delays


def run_method(m, Pall_raw, Pall_cal, sess, prior, tau, filt=None):
    E, use, grid, acc, gate = method_scores(m, Pall_raw, Pall_cal, sess, prior, tau, target=D, filt=filt)
    if m == 'EviAlert-W-noClip':
        grid = GRID_SUM_WIDE
    on, st = core.run_policy(E, use, grid, window=W_DEFAULT, rho=0.5, refractory=24, gate=gate, accumulate=acc)
    return grid, on, st


def evaluate(S, P, n_splits=200, budget=1.0, delta=0.05, n_boot=200, miss_rate=0.10, seed0=0, scoring='all',
             keep_curves=True):
    subs_all = list(S)
    out = []
    for sp in range(n_splits):
        rng = np.random.default_rng(seed0 + sp)
        perm = rng.permutation(subs_all); cal, ev = list(perm[:7]), list(perm[7:])
        usable = {s: (S[s]['acc_std5'] <= MOTION_THR) & markov_missing(len(S[s]['label_end']), rng, miss_rate) for s in subs_all}
        Xc = np.vstack([P[s][np.isin(S[s]['label_end'], CLASSES)] for s in cal])
        yc = np.concatenate([np.searchsorted(CLASSES, S[s]['label_end'][np.isin(S[s]['label_end'], CLASSES)]) for s in cal])
        Tst = core.fit_temperature(Xc, yc); prior = float(np.isin(yc, D).mean()); tau = float(np.quantile(Xc.max(1), 0.30))
        res = dict(split=sp, cal=cal, eval=ev, T=Tst, prior=prior)
        filt = None
        if 'EviAlert-F' in METHODS:                         # v2 filter parameters from calibration persons only
            Praw_c, sess_c, L_c = build(S, cal, P, usable)
            Pcal_c = core.temperature_scale(Praw_c, Tst)
            tgt = [S[s_]['label_end'] == 2 for s_ in cal]
            fa_, fb_ = core.fit_transitions(tgt)
            ell_c = core.evidence(Pcal_c[sess_c['img']], D, prior, None)
            evs_c = [events_of(S[s_]['label_end']) for s_ in cal]
            best = None
            for g_ in GAMMAS:
                Lg = core.filter_logodds(ell_c, sess_c['usable'], fa_, fb_, g_)
                on_, _ = core.run_policy(Lg, sess_c['usable'], GRID_LOGODDS, window=W_DEFAULT, rho=0.5, refractory=24, gate=True, accumulate=False)
                fa0, hr0, dt0, tt0, _ = score(on_, L_c, evs_c)
                rc = fa0.sum(1) / hr0.sum(); rr = dt0 / max(tt0, 1)
                sc_ = np.mean([recall_at(rc, rr, x_) for x_ in (0.5, 1.0, 2.0)])
                if best is None or sc_ > best[1] + 1e-12:
                    best = (g_, sc_)
            filt = dict(a=fa_, b=fb_, gamma=best[0]); res['filt'] = filt
        boots = [rng.choice(len(cal), len(cal), replace=True) for _ in range(n_boot)]
        for grp, subs in (('cal', cal), ('eval', ev)):
            Praw, sess, L = build(S, subs, P, usable)
            Pcal = core.temperature_scale(Praw, Tst)
            evs = [events_of(S[s]['label_end']) for s in subs]
            dfn = None
            if scoring == 'defined':
                dfn = [np.isin(S[s]['label_end'], [1, 3, 4]) for s in subs]
                for d_, e_ in zip(dfn, evs):          # the stress block itself is evaluated as an event
                    for (a_, b_) in e_:
                        d_[a_:b_] = True
            for m in METHODS:
                grid, on, st = run_method(m, Praw, Pcal, sess, prior, tau, filt)
                fa, hrs, det, tot, delays = score(on, L, evs, defined=dfn)
                r = res.setdefault(m, {})
                if grp == 'cal':
                    r['_bounds'] = dict(naive=fa.sum(1) / hrs.sum(),
                                        poisson=np.array([core.poisson_ucb(c, hrs.sum(), delta) for c in fa.sum(1)]),
                                        gboot=np.quantile(np.array([fa[:, b].sum(1) / hrs[b].sum() for b in boots]), 1 - delta, axis=0))
                else:
                    rate = fa.sum(1) / hrs.sum(); rec = det / max(tot, 1)
                    r['n_events'] = int(tot); r['hours'] = float(hrs.sum())
                    for x in (0.5, 1.0, 2.0):
                        okx = rate <= x + 1e-12
                        r[f'recall@{x}'] = float(rec[okx].max()) if okx.any() else 0.0
                    okb = np.where(rate <= budget + 1e-12)[0]
                    j = okb[np.argmax(rec[okb])] if len(okb) else int(np.argmin(rate))
                    r['oracle'] = dict(fa=float(rate[j]), recall=float(rec[j]), delay_med=float(np.median(delays[j])) if delays[j] else float('nan'))
                    valid = np.zeros_like(sess['usable'])
                    for i, l in enumerate(L):
                        valid[i, :l] = True
                    for rule, bd in r.pop('_bounds').items():
                        h, feas = core.select_threshold(grid, bd, budget)
                        jj = int(np.where(grid == h)[0][0])
                        r[rule] = dict(h=h, feasible=feas, fa=float(rate[jj]), recall=float(rec[jj]), compliant=bool(rate[jj] <= budget),
                                       delay_med=float(np.median(delays[jj])) if delays[jj] else float('nan'),
                                       stale=int((on[jj] & ~sess['usable'] & valid).sum()),
                                       insuff=float((st[jj][valid] == core.INSUFF).mean()))
                    if keep_curves:
                        r['curve'] = dict(fa=rate.tolist(), recall=rec.tolist())
        out.append(res)
    return out


def frame_metrics(S, P):
    y = np.concatenate([S[s]['label_end'] for s in S]); Pa = np.vstack([P[s] for s in S])
    m = np.isin(y, CLASSES); yi = np.searchsorted(CLASSES, y[m]); Pm = Pa[m]
    return dict(acc=float(accuracy_score(yi, Pm.argmax(1))), mf1=float(f1_score(yi, Pm.argmax(1), average='macro')),
                auroc_stress=float(roc_auc_score(yi == 1, Pm[:, 1])), ece_raw=core.ece(Pm, yi), nll_raw=core.nll(Pm, yi),
                n_frames=int(m.sum()))


SCENARIOS = {
    'primary': dict(miss_rate=0.10, scoring='all', keep_curves=True),     # pre-specified
    'defined': dict(miss_rate=0.10, scoring='defined', keep_curves=False),  # undefined segments not scored
    'natural': dict(miss_rate=0.0, scoring='all', keep_curves=False),      # only natural motion artefacts
}


def run_model(args):
    feat_dir, model, n_splits = args
    S = load(feat_dir)
    t0 = time.time()
    P = loso_probs(S, model)
    out = dict(frame=frame_metrics(S, P))
    for name, kw in SCENARIOS.items():
        out[name] = evaluate(S, P, n_splits=n_splits, **kw)
    print(model, out['frame'], 'done in', round(time.time() - t0), 's', flush=True)
    return model, out


if __name__ == '__main__':
    from multiprocessing import Pool
    feat_dir, out_json = sys.argv[1], sys.argv[2]
    n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    S = load(feat_dir)
    meta = dict(subjects=list(S), frames={s: int(len(S[s]['label_end'])) for s in S},
                hours={s: float(len(S[s]['label_end']) * STEP / 3600) for s in S},
                events={s: [list(map(int, e)) for e in events_of(S[s]['label_end'])] for s in S},
                motion_unusable={s: float((S[s]['acc_std5'] > MOTION_THR).mean()) for s in S},
                labels={s: {int(k): int(v) for k, v in zip(*np.unique(S[s]['label_end'], return_counts=True))} for s in S})
    with Pool(3) as pool:
        res = dict(pool.map(run_model, [(feat_dir, m, n_splits) for m in ['LR', 'RF', 'HGB']]))
    json.dump(dict(meta=meta, models=res), open(out_json, 'w'))
