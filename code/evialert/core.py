"""EviAlert: evidence-aware, abstaining, tiered alerting from streams of uncertain classifier outputs.

Components
  1. calibrate()      temperature scaling of per-frame class probabilities (NLL on a calibration set)
  2. evidence()       prior-corrected, clipped log-likelihood ratio for the target set D (e.g. distress)
  3. run_policy()     CUSUM-type accumulation with frame abstention (missing/unusable frames add no
                      evidence), a window-level evidence-sufficiency gate (Insufficient Evidence state),
                      two tiers (Watchlist, High Alert) and a refractory period after each High Alert
  4. ltt_threshold()  risk-controlling threshold selection: the smallest threshold whose one-sided
                      (1 - delta) upper confidence bound on the false-alert rate stays below a budget,
                      scanned from conservative to liberal (fixed-sequence testing, Learn-then-Test)

Baseline scorers used in the experiments share the same alert mechanics and threshold selection,
so differences come from how evidence is represented and accumulated.
"""
import numpy as np
from scipy import optimize, stats

EPS = 1e-12

# ----------------------------------------------------------------------------------------- calibration
def temperature_scale(P, T):
    L = np.log(np.clip(P, EPS, 1.0)) / T
    L -= L.max(axis=-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=-1, keepdims=True)


def fit_temperature(P, y):
    """Temperature minimizing the negative log-likelihood of labels y under softmax(log P / T)."""
    def nll(logT):
        Q = temperature_scale(P, np.exp(logT))
        return -np.mean(np.log(np.clip(Q[np.arange(len(y)), y], EPS, 1)))
    res = optimize.minimize_scalar(nll, bounds=(np.log(0.05), np.log(20)), method='bounded')
    return float(np.exp(res.x))


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def nll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], EPS, 1))))


def brier(P, y):
    Y = np.eye(P.shape[1])[y]
    return float(np.mean(((P - Y) ** 2).sum(1)))


# ----------------------------------------------------------------------------------------- evidence
def target_prob(P, target):
    return P[..., target].sum(-1)


def evidence(P, target, prior_cal, clip=3.0):
    """Clipped log-likelihood ratio log p(x | D) / p(x | not D), recovered from calibrated posteriors
    by removing the calibration-set prior odds (Saerens et al., 2002)."""
    p = np.clip(target_prob(P, target), 1e-6, 1 - 1e-6)
    llr = np.log(p / (1 - p)) - np.log(prior_cal / (1 - prior_cal))
    return np.clip(llr, -clip, clip) if clip is not None else llr


# ----------------------------------------------------------------------------------------- policy
NO_ALERT, WATCH, HIGH, INSUFF = 0, 1, 2, 3


def run_policy(E, usable, h_high, h_watch=None, window=6, rho=0.5, refractory=24, drift=0.0,
               gate=True, accumulate=True):
    """Vectorized over sessions.

    E       (n, T) per-step evidence (any score; for baselines a window statistic)
    usable  (n, T) bool, False where the frame is missing/unusable
    h_high  scalar or (k,) array of High Alert thresholds -> returns results for every threshold
    If accumulate: S_t = max(0, S_{t-1} + E_t - drift) on usable steps, unchanged otherwise (CUSUM).
    Else the score itself is compared with the thresholds (baselines).
    Returns onset (k, n, T) bool High-Alert onsets, status (k, n, T) int8.
    """
    h = np.atleast_1d(np.asarray(h_high, float))
    k = len(h); n, T = E.shape
    hw = None if h_watch is None else np.atleast_1d(np.asarray(h_watch, float))
    S = np.zeros((k, n)); refr = np.zeros((k, n), int)
    onset = np.zeros((k, n, T), bool); status = np.zeros((k, n, T), np.int8)
    cum_use = np.concatenate([np.zeros((n, 1)), np.cumsum(usable, 1)], 1)
    for t in range(T):
        lo = max(0, t - window + 1)
        frac = (cum_use[:, t + 1] - cum_use[:, lo]) / (t + 1 - lo)
        insuff = (frac < rho) if gate else np.zeros(n, bool)
        u = usable[:, t]
        if accumulate:
            upd = np.maximum(0.0, S + (E[:, t] - drift)[None, :])
            S = np.where(u[None, :], upd, S)
            score = S
        else:
            score = np.broadcast_to(E[:, t][None, :], (k, n))
        active = refr == 0
        fire = (score >= h[:, None]) & active & ~insuff[None, :]
        onset[:, :, t] = fire
        st = np.full((k, n), NO_ALERT, np.int8)
        if hw is not None:
            st[(score >= hw[:, None]) if len(hw) == k else (score >= hw[0])] = WATCH
        st[(score >= h[:, None])] = HIGH
        st[~active] = HIGH                        # alert remains displayed until acknowledged
        st[:, insuff] = INSUFF
        status[:, :, t] = st
        if accumulate:
            S = np.where(fire, 0.0, S)            # reset after an alert ("Mark as checked")
        refr = np.where(fire, refractory, np.maximum(refr - 1, 0))
    return onset, status


# ----------------------------------------------------------------------------------------- evaluation
def score_alerts(onset, events, T, step_s, grace_steps):
    """onset (k, n, T); events: list per session of (start, end) sustained episodes (end exclusive).
    Returns per threshold: false alerts, hours, detected events, total events, delays (s)."""
    k, n, _ = onset.shape
    hours = n * T * step_s / 3600.0
    fa = np.zeros(k); det = np.zeros(k); tot = sum(len(e) for e in events)
    delays = [[] for _ in range(k)]
    for i in range(n):
        ev = events[i]
        on = onset[:, i, :]
        ok = np.zeros((k, T), bool)
        for (s, e) in ev:
            ok[:, s:min(T, e + grace_steps)] = True
            hit = on[:, s:min(T, e + grace_steps)]
            any_hit = hit.any(1)
            det += any_hit
            first = hit.argmax(1)
            for j in np.where(any_hit)[0]:
                delays[j].append(first[j] * step_s)
        fa += (on & ~ok).sum(1)
    return dict(fa=fa, hours=hours, det=det, tot=tot, delays=delays)


def poisson_ucb(count, hours, delta=0.05):
    return stats.chi2.ppf(1 - delta, 2 * count + 2) / 2.0 / hours


def select_threshold(thresholds, rate_bound, budget):
    """Fixed-sequence testing from the most conservative (largest) threshold downwards: keep lowering
    the threshold while the bound on the false-alert rate stays <= budget. Returns (threshold, feasible)."""
    thresholds = np.asarray(thresholds); rate_bound = np.asarray(rate_bound)
    order = np.argsort(thresholds)[::-1]
    if rate_bound[order[0]] > budget:
        return float(thresholds[order[0]]), False
    chosen = thresholds[order[0]]
    for j in order:
        if rate_bound[j] <= budget:
            chosen = thresholds[j]
        else:
            break
    return float(chosen), True


def ltt_threshold(thresholds, fa_counts, hours, budget, delta=0.05):
    """Learn-then-Test style selection with a one-sided Poisson upper confidence bound."""
    ub = np.array([poisson_ucb(c, hours, delta) for c in fa_counts])
    return select_threshold(thresholds, ub, budget)


# ========================================================================================== EviAlert v2
# (1) a two-state hidden Markov filter driven by calibrated, prior-corrected frame evidence, with a
#     sequence-level tempering exponent gamma that discounts redundant (temporally dependent) frames;
# (2) person-level certification of the false-alert budget by Learn-then-Test with binomial p-values.

def fit_transitions(state_seqs, usable_target=None):
    """Per-step onset hazard a = P(D_t | N_{t-1}) and exit hazard b = P(N_t | D_{t-1}) from boolean
    target-state sequences (list of 1-D arrays). Add-one smoothing keeps both strictly positive."""
    nn = nd = dn = dd = 0
    for s in state_seqs:
        s = np.asarray(s, bool)
        if len(s) < 2:
            continue
        p, c = s[:-1], s[1:]
        nn += int((~p & ~c).sum()); nd += int((~p & c).sum()); dn += int((p & ~c).sum()); dd += int((p & c).sum())
    a = (nd + 1.0) / (nn + nd + 2.0); b = (dn + 1.0) / (dd + dn + 2.0)
    return float(a), float(b)


def filter_logodds(ell, usable, a, b, gamma, l0=None):
    """Forward filter in log-odds. ell, usable: (n, T). Returns posterior log-odds of the target state.
    Unusable frames receive only the prediction step (no evidence, no imputation)."""
    ell = np.asarray(ell, float); usable = np.asarray(usable, bool)
    n, T = ell.shape
    p = np.full(n, a / (a + b)) if l0 is None else 1 / (1 + np.exp(-l0))
    out = np.empty((n, T))
    for t in range(T):
        pp = p * (1 - b) + (1 - p) * a                       # prediction
        L = np.log(pp) - np.log1p(-pp)
        L = np.where(usable[:, t], L + gamma * ell[:, t], L)  # tempered evidence update
        L = np.clip(L, -50, 50)
        out[:, t] = L
        p = 1 / (1 + np.exp(-L))
    return out


def fit_gamma(ell, usable, states, a, b, lo=0.01, hi=1.0):
    """Sequence-level calibration: the tempering exponent that minimizes the negative log-likelihood of
    the true target states under the filtered posterior on calibration streams."""
    st = np.asarray(states, bool)
    def nll(lg):
        L = filter_logodds(ell, usable, a, b, np.exp(lg))
        logp = -np.logaddexp(0, -L); log1mp = -np.logaddexp(0, L)
        return -float(np.where(st, logp, log1mp).mean())
    res = optimize.minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method='bounded', options=dict(xatol=1e-3))
    return float(np.exp(res.x))


def certify_threshold(thresholds, person_rates, budget, alpha=0.2, delta=0.1):
    """Person-level budget certification. person_rates: (k, n) false-alert rate of each calibration person
    at each threshold. H_j: more than a fraction alpha of new persons exceed the budget at threshold j.
    p_j = P(Binomial(n, alpha) <= X_j), X_j = #calibration persons above budget. Fixed-sequence testing from
    the most conservative threshold at level delta (Learn-then-Test). Returns (threshold, certified, n_min),
    where n_min is the smallest calibration population that could certify anything at (alpha, delta)."""
    thresholds = np.asarray(thresholds); R = np.asarray(person_rates)
    n = R.shape[1]
    order = np.argsort(thresholds)[::-1]
    pvals = stats.binom.cdf((R > budget + 1e-12).sum(1), n, alpha)
    n_min = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    if pvals[order[0]] > delta:
        return float(thresholds[order[0]]), False, n_min
    chosen = thresholds[order[0]]
    for j in order:
        if pvals[j] <= delta:
            chosen = thresholds[j]
        else:
            break
    return float(chosen), True, n_min
