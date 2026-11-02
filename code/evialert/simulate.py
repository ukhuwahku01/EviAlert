"""Semi-synthetic play sessions built from real out-of-fold classifier outputs.

Each session has a latent state sequence (Normal/Laugh baseline, with transient and sustained distress
episodes of Angry or Cry). At every step an image whose true label equals the latent state is drawn
from the evaluation pool, preferring the session's own child identity, and its real out-of-fold
probability vector becomes the observation. An image persists for a geometric number of steps to
induce temporal correlation, and bursts of unusable frames (face not visible) are added.
Only the temporal structure is synthetic; images, labels and model outputs are real.
"""
import numpy as np

CLASSES = ['Angry', 'Cry', 'Laugh', 'Normal']
ANGRY, CRY, LAUGH, NORMAL = 0, 1, 2, 3
DISTRESS = [ANGRY, CRY]

DEFAULT = dict(
    T=180, step_s=5,                       # 15-minute sessions observed every 5 s
    base_seg_mean=12, p_laugh=0.35,        # baseline segments (mean 60 s), 35% Laugh
    episodes_mean=0.8, p_sustained=0.6,    # Poisson number of distress episodes per session
    sustained=(9, 36), transient=(2, 5),   # durations in steps: 45-180 s vs 10-25 s
    persist_mean=2.0,                      # mean steps an image persists (temporal correlation)
    miss_rate=0.10, miss_burst=4,          # long-run share of unusable steps, mean burst length
    p_same_identity=0.8,                   # prefer the session child's own images when available
)


def _latent(rng, cfg):
    T = cfg['T']; s = np.empty(T, int); t = 0
    while t < T:
        L = rng.geometric(1.0 / cfg['base_seg_mean'])
        s[t:t + L] = LAUGH if rng.random() < cfg['p_laugh'] else NORMAL
        t += L
    events = []
    k = rng.poisson(cfg['episodes_mean'])
    for _ in range(k):
        sus = rng.random() < cfg['p_sustained']
        lo, hi = cfg['sustained'] if sus else cfg['transient']
        d = int(rng.integers(lo, hi + 1))
        if d >= T - 2:
            continue
        start = int(rng.integers(1, T - d))
        if any(not (start + d + 2 <= a or b + 2 <= start) for a, b, _ in events):
            continue                                            # keep episodes separated
        s[start:start + d] = ANGRY if rng.random() < 0.5 else CRY
        events.append((start, start + d, sus))
    return s, sorted(events)


def _missing(rng, cfg):
    T = cfg['T']; r = cfg['miss_rate']
    if r <= 0:
        return np.ones(T, bool)
    b = cfg['miss_burst']; p_exit = 1.0 / b; p_enter = r * p_exit / (1 - r)
    use = np.ones(T, bool); state = rng.random() < r
    for t in range(T):
        use[t] = not state
        state = (rng.random() >= p_exit) if state else (rng.random() < p_enter)
    return use


def make_sessions(pool_idx, labels, groups, n, rng, cfg=None):
    """pool_idx: indices (into the OOF arrays) of images available for these sessions."""
    cfg = {**DEFAULT, **(cfg or {})}
    by_class = {c: pool_idx[labels[pool_idx] == c] for c in range(4)}
    by_gc = {}
    for i in pool_idx:
        by_gc.setdefault((groups[i], labels[i]), []).append(i)
    pool_groups = np.unique(groups[pool_idx])
    T = cfg['T']
    IMG = np.empty((n, T), int); STATE = np.empty((n, T), int); USE = np.empty((n, T), bool)
    EVENTS, TRANS = [], []
    same = 0; draws = 0
    for j in range(n):
        s, ev = _latent(rng, cfg)
        g = rng.choice(pool_groups)
        t = 0
        while t < T:
            c = s[t]
            cand = by_gc.get((g, c))
            if cand and rng.random() < cfg['p_same_identity']:
                img = cand[rng.integers(len(cand))]; same += 1
            else:
                arr = by_class[c]; img = arr[rng.integers(len(arr))]
            draws += 1
            L = rng.geometric(1.0 / cfg['persist_mean'])
            end = t + L
            while t < min(end, T) and s[t] == c:              # an image never outlives its state
                IMG[j, t] = img; t += 1
        STATE[j] = s; USE[j] = _missing(rng, cfg)
        EVENTS.append([(a, b) for a, b, sus in ev if sus])
        TRANS.append([(a, b) for a, b, sus in ev if not sus])
    return dict(img=IMG, state=STATE, usable=USE, events=EVENTS, transient=TRANS, cfg=cfg,
                same_identity_share=same / max(draws, 1))
