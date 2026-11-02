"""Frame-level wrist (Empatica E4) features from WESAD, computed causally every STEP seconds over the
preceding WIN seconds. NumPy only, so it can run next to the raw data.

Usage: python3 extract_features.py <WESAD.zip or WESAD folder> <out_dir> [S2 S3 ...]
Writes <out_dir>/<S>_frames.npz with X (frames x features), names, label_end, label_mode5,
t_end (s), and quality signals (acc_std5, eda_mean5, ibi_valid).
WESAD labels: 0 transient/undefined, 1 baseline, 2 stress, 3 amusement, 4 meditation, 5-7 ignore.
"""
import io, os, sys, pickle, zipfile
import numpy as np

FS = dict(ACC=32, BVP=64, EDA=4, TEMP=4, label=700)
WIN, STEP = 60.0, 5.0
SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']


def _slope(x, fs):
    n = len(x)
    if n < 2:
        return np.nan
    t = np.arange(n) / fs
    t = t - t.mean()
    return float((t * (x - x.mean())).sum() / (t * t).sum())


def _movavg(x, w):
    w = max(1, int(w))
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = (c[w:] - c[:-w]) / w
    return np.concatenate([np.full(w - 1, out[0] if len(out) else 0.0), out])


def _bvp_peaks(x, fs):
    """Greedy systolic-peak detector: local maxima above the 1-s moving mean, >= 0.33 s apart."""
    if len(x) < 3:
        return np.array([], int)
    y = x - _movavg(x, fs)
    lm = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > 0))[0] + 1
    keep = []
    mind = int(0.33 * fs)
    for i in lm:
        if keep and i - keep[-1] < mind:
            if y[i] > y[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    return np.asarray(keep, int)


def frame_features(sig, t_end):
    out = {}
    def seg(name, w):
        fs = FS[name]; a = sig[name]
        hi = int(round(t_end * fs)); lo = max(0, hi - int(round(w * fs)))
        return a[lo:hi]
    eda = seg('EDA', WIN).ravel(); temp = seg('TEMP', WIN).ravel()
    acc = seg('ACC', WIN).reshape(-1, 3); bvp = seg('BVP', WIN).ravel()
    # EDA (microsiemens)
    out['eda_mean'] = eda.mean(); out['eda_std'] = eda.std(); out['eda_min'] = eda.min(); out['eda_max'] = eda.max()
    out['eda_slope'] = _slope(eda, 4)
    d = np.diff(_movavg(eda, 4)) * 4
    out['eda_rise'] = d[d > 0].sum() / max(len(d), 1) if len(d) else 0.0
    sm = _movavg(eda, 4); dd = np.diff(sm)
    pk = np.where((dd[:-1] > 0) & (dd[1:] <= 0))[0] + 1
    out['eda_scr_n'] = float(sum(1 for i in pk if sm[i] - sm[max(0, i - 8):i + 1].min() > 0.01))
    # skin temperature
    out['temp_mean'] = temp.mean(); out['temp_std'] = temp.std(); out['temp_slope'] = _slope(temp, 4)
    # acceleration (1/64 g units)
    mag = np.sqrt((acc.astype(float) ** 2).sum(1))
    out['acc_mag_mean'] = mag.mean(); out['acc_mag_std'] = mag.std()
    for j, ax in enumerate('xyz'):
        out[f'acc_{ax}_std'] = acc[:, j].std()
    # blood volume pulse -> inter-beat intervals
    pks = _bvp_peaks(bvp.astype(float), 64)
    ibi = np.diff(pks) / 64.0
    ok = (ibi > 0.33) & (ibi < 1.5)
    ibi_v = ibi[ok]
    if len(ibi_v) >= 5:
        out['hr_mean'] = 60.0 / ibi_v.mean(); out['hr_std'] = (60.0 / ibi_v).std()
        out['ibi_sdnn'] = ibi_v.std(); out['ibi_rmssd'] = np.sqrt(np.mean(np.diff(ibi_v) ** 2)) if len(ibi_v) > 2 else np.nan
    else:
        out['hr_mean'] = out['hr_std'] = out['ibi_sdnn'] = out['ibi_rmssd'] = np.nan
    out['bvp_std'] = bvp.std()
    # quality over the current frame period (last STEP seconds)
    acc5 = seg('ACC', STEP).reshape(-1, 3); mag5 = np.sqrt((acc5.astype(float) ** 2).sum(1))
    q = dict(acc_std5=mag5.std() if len(mag5) else np.nan,
             eda_mean5=seg('EDA', STEP).mean() if len(seg('EDA', STEP)) else np.nan,
             ibi_valid=float(ok.mean()) if len(ok) else 0.0)
    return out, q


def process(data):
    sig = {k: np.asarray(v) for k, v in data['signal']['wrist'].items()}
    lab = np.asarray(data['label']).ravel()
    T = len(lab) / FS['label']
    T = min(T, min(len(sig[k]) / FS[k] for k in ('ACC', 'BVP', 'EDA', 'TEMP')))
    t_ends = np.arange(WIN, T + 1e-9, STEP)
    rows, qs, le, lm = [], [], [], []
    for te in t_ends:
        f, q = frame_features(sig, te); rows.append(f); qs.append(q)
        i = min(len(lab) - 1, int(round(te * FS['label'])) - 1)
        le.append(int(lab[i]))
        lo = max(0, int(round((te - STEP) * FS['label'])))
        vals, cnt = np.unique(lab[lo:i + 1], return_counts=True)
        lm.append(int(vals[np.argmax(cnt)]))
    names = list(rows[0].keys())
    X = np.array([[r[n] for n in names] for r in rows], float)
    return dict(X=X, names=np.array(names), label_end=np.array(le), label_mode5=np.array(lm), t_end=t_ends,
                acc_std5=np.array([q['acc_std5'] for q in qs]), eda_mean5=np.array([q['eda_mean5'] for q in qs]),
                ibi_valid=np.array([q['ibi_valid'] for q in qs]))


def load_subject(src, s):
    if os.path.isdir(src):
        with open(os.path.join(src, s, f'{s}.pkl'), 'rb') as fh:
            return pickle.load(fh, encoding='latin1')
    z = zipfile.ZipFile(src)
    name = [n for n in z.namelist() if n.endswith(f'/{s}/{s}.pkl') or n == f'{s}/{s}.pkl'][0]
    with z.open(name) as fh:
        return pickle.load(io.BufferedReader(fh), encoding='latin1')


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    subs = sys.argv[3:] or SUBJECTS
    os.makedirs(out, exist_ok=True)
    for s in subs:
        dst = os.path.join(out, f'{s}_frames.npz')
        if os.path.exists(dst):
            print(s, 'exists, skipped'); continue
        try:
            data = load_subject(src, s)
        except (FileNotFoundError, IndexError):
            print(s, 'not found, skipped'); continue
        res = process(data)
        np.savez_compressed(dst, **res)
        print(s, 'frames', len(res['t_end']), 'labels', np.unique(res['label_end'], return_counts=True))
