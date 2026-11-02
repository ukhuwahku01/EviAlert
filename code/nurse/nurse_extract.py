"""Frame-level wrist features from the nurse stress dataset (Hosseini et al., 2022; Empatica E4).

Usage: python3 nurse_extract.py <extracted_root> <SurveyResults.xlsx> <out_dir>

* Finds every recording session (a folder or zip that contains EDA.csv, TEMP.csv, ACC.csv, HR.csv in
  the E4 export format: row 1 = start epoch, row 2 = sampling rate, then samples).
* Participant ID = name of the top-level folder under <extracted_root>.
* Survey events are aligned to the recordings with the time-zone offset that places the most events
  inside recorded time (an objective rule that does not look at the physiological signals).
* Every STEP s, features are computed over the preceding WIN s (causal), vectorized per session.
Writes <out_dir>/<ID>_frames.npz and <out_dir>/alignment.json.
"""
import io, json, os, sys, zipfile, glob, re
import numpy as np
import pandas as pd

WIN, STEP = 60.0, 5.0
SIGS = ('EDA', 'TEMP', 'ACC', 'HR')


def read_e4(text):
    lines = text.strip().splitlines()
    start = float(lines[0].split(',')[0]); fs = float(lines[1].split(',')[0])
    data = np.loadtxt(io.StringIO('\n'.join(lines[2:])), delimiter=',', ndmin=2) if len(lines) > 2 else np.zeros((0, 1))
    return start, fs, data


def find_sessions(root):
    """Yield (participant, session_name, {sig: (start, fs, data)})."""
    sessions = []
    for dirpath, dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root); pid = rel.split(os.sep)[0] if rel != '.' else None
        if set(f'{s}.csv' for s in SIGS) <= set(files) and pid:
            sessions.append((pid, rel, {s: read_e4(open(os.path.join(dirpath, f'{s}.csv')).read()) for s in SIGS}))
        for f in files:
            if f.lower().endswith('.zip') and pid:
                try:
                    z = zipfile.ZipFile(os.path.join(dirpath, f))
                except zipfile.BadZipFile:
                    continue
                names = {os.path.basename(n): n for n in z.namelist()}
                if all(f'{s}.csv' in names for s in SIGS):
                    sessions.append((pid, os.path.join(rel, f),
                                     {s: read_e4(z.read(names[f'{s}.csv']).decode('utf8', 'ignore')) for s in SIGS}))
    return sessions


def read_survey(path):
    xl = pd.read_excel(path, sheet_name=None)
    rows = []
    for sheet, df in xl.items():
        cols = {c: str(c).strip().lower() for c in df.columns}
        def pick(*keys):
            for c, lc in cols.items():
                if all(k in lc for k in keys):
                    return c
            return None
        cid, cs, ce, cd, cl = pick('id'), pick('start'), pick('end'), pick('date'), pick('stress', 'level')
        for _, r in df.iterrows():
            pid = str(r[cid]).strip() if cid is not None and pd.notna(r[cid]) else str(sheet).strip()
            if cs is None or ce is None or cd is None or pd.isna(r[cs]) or pd.isna(r[ce]) or pd.isna(r[cd]):
                continue
            lvl = str(r[cl]).strip().lower() if cl is not None else 'na'
            rows.append(dict(pid=pid, date=r[cd], start=r[cs], end=r[ce], level=lvl))
    return pd.DataFrame(rows)


def to_naive_seconds(date, tm):
    d = pd.to_datetime(date).date() if not isinstance(date, pd.Timestamp) else date.date()
    if isinstance(tm, str):
        tm = pd.to_datetime(tm).time()
    elif hasattr(tm, 'hour') and not hasattr(tm, 'date'):
        pass
    elif isinstance(tm, pd.Timestamp):
        tm = tm.time()
    ts = pd.Timestamp.combine(d, tm)
    return (ts - pd.Timestamp('1970-01-01')).total_seconds()


def norm_pid(p):
    return re.sub(r'[^0-9a-z]', '', str(p).lower()).lstrip('0') or '0'


def windows_idx(n_frames, t_ends, start, fs, w):
    hi = np.round((t_ends - start) * fs).astype(int); lo = hi - int(round(w * fs))
    return lo, hi


def roll_stats(x, lo, hi):
    """Mean, SD, min, max, OLS slope of x[lo:hi] for many windows (vectorized with cumulative sums)."""
    n = len(x); lo = np.clip(lo, 0, n); hi = np.clip(hi, 0, n); L = np.maximum(hi - lo, 1)
    c1 = np.concatenate([[0], np.cumsum(x)]); c2 = np.concatenate([[0], np.cumsum(x * x)])
    i = np.arange(n, dtype=float); ci = np.concatenate([[0], np.cumsum(i * x)])
    s1 = c1[hi] - c1[lo]; s2 = c2[hi] - c2[lo]; si = ci[hi] - ci[lo]
    mean = s1 / L; var = np.maximum(s2 / L - mean ** 2, 0)
    # slope of x on index (per sample), then per second outside
    k = np.arange(len(lo))
    tbar = (lo + hi - 1) / 2.0
    stt = (L ** 3 - L) / 12.0
    slope = np.where(stt > 0, (si - tbar * s1) / np.where(stt > 0, stt, 1), 0.0)
    W = int(np.max(hi - lo)) if len(lo) else 1
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(W, x[0] if n else 0.0), x])
    sw = sliding_window_view(pad, W)            # window ending at position j (exclusive) = sw[j]
    mn = sw[hi].min(1); mx = sw[hi].max(1)
    return mean, np.sqrt(var), mn, mx, slope


def session_frames(sig):
    st = {s: sig[s][0] for s in SIGS}; fs = {s: sig[s][1] for s in SIGS}
    eda = sig['EDA'][2][:, 0].astype(float); tmp = sig['TEMP'][2][:, 0].astype(float)
    acc = sig['ACC'][2][:, :3].astype(float); hr = sig['HR'][2][:, 0].astype(float)
    t0 = max(st['EDA'], st['TEMP'], st['ACC'], st['HR'])
    t1 = min(st['EDA'] + len(eda) / fs['EDA'], st['TEMP'] + len(tmp) / fs['TEMP'],
             st['ACC'] + len(acc) / fs['ACC'], st['HR'] + len(hr) / fs['HR'])
    t_ends = np.arange(t0 + WIN, t1 + 1e-9, STEP)
    if len(t_ends) == 0:
        return None
    F = {}
    lo, hi = windows_idx(len(t_ends), t_ends, st['EDA'], fs['EDA'], WIN)
    m, sd, mn, mx, sl = roll_stats(eda, lo, hi)
    F.update(eda_mean=m, eda_std=sd, eda_min=mn, eda_max=mx, eda_slope=sl * fs['EDA'])
    de = np.diff(eda, prepend=eda[0]); dpos = np.where(de > 0, de, 0.0)
    rm = roll_stats(dpos, lo, hi)[0]; F['eda_rise'] = rm * fs['EDA']
    lo, hi = windows_idx(len(t_ends), t_ends, st['TEMP'], fs['TEMP'], WIN)
    m, sd, mn, mx, sl = roll_stats(tmp, lo, hi); F.update(temp_mean=m, temp_std=sd, temp_slope=sl * fs['TEMP'])
    mag = np.sqrt((acc ** 2).sum(1))
    lo, hi = windows_idx(len(t_ends), t_ends, st['ACC'], fs['ACC'], WIN)
    m, sd, _, _, _ = roll_stats(mag, lo, hi); F.update(acc_mag_mean=m, acc_mag_std=sd)
    for j, ax in enumerate('xyz'):
        F[f'acc_{ax}_std'] = roll_stats(acc[:, j], lo, hi)[1]
    lo5, hi5 = windows_idx(len(t_ends), t_ends, st['ACC'], fs['ACC'], STEP)
    acc_std5 = roll_stats(mag, lo5, hi5)[1]
    lo, hi = windows_idx(len(t_ends), t_ends, st['HR'], fs['HR'], WIN)
    m, sd, _, _, sl = roll_stats(hr, lo, hi); F.update(hr_mean=m, hr_std=sd, hr_slope=sl * fs['HR'])
    lo5, hi5 = windows_idx(len(t_ends), t_ends, st['EDA'], fs['EDA'], STEP)
    eda_mean5 = roll_stats(eda, lo5, hi5)[0]
    names = list(F); X = np.column_stack([F[k] for k in names])
    return dict(X=X, names=names, t_end=t_ends, acc_std5=acc_std5, eda_mean5=eda_mean5, t0=t0, t1=t1)


if __name__ == '__main__':
    root, survey, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    sess = find_sessions(root)
    print('sessions found:', len(sess), 'participants:', sorted(set(p for p, _, _ in sess)))
    frames = {}
    for pid, name, sig in sess:
        r = session_frames(sig)
        if r is None:
            continue
        frames.setdefault(pid, []).append((name, r))
    cover = {pid: [(r['t0'], r['t1']) for _, r in L] for pid, L in frames.items()}
    ev = read_survey(survey)
    print('survey events:', len(ev), 'levels:', ev['level'].value_counts().to_dict() if len(ev) else {})
    pid_map = {norm_pid(p): p for p in frames}
    ev['pkey'] = ev['pid'].map(norm_pid)
    ev = ev[ev['pkey'].isin(pid_map)].copy(); ev['pid'] = ev['pkey'].map(pid_map)
    ev['s_naive'] = [to_naive_seconds(d, t) for d, t in zip(ev['date'], ev['start'])]
    ev['e_naive'] = [to_naive_seconds(d, t) for d, t in zip(ev['date'], ev['end'])]
    ev.loc[ev['e_naive'] < ev['s_naive'], 'e_naive'] += 86400
    # alignment: survey times are local wall-clock times at the hospital (US Central, Hosseini et al., 2022).
    # Primary rule: America/Chicago with daylight saving. Alternatives are only reported as a check of how many
    # event midpoints fall inside recorded time (a rule that does not look at the physiological signals).
    def local_to_utc(naive_sec, tz):
        ts = pd.to_datetime(pd.Series(naive_sec), unit='s')
        if tz.startswith('UTC'):
            return (naive_sec - float(tz[3:]) * 3600) if tz != 'UTC' else naive_sec
        loc = ts.dt.tz_localize(tz, ambiguous='NaT', nonexistent='shift_forward')
        return ((loc.dt.tz_convert('UTC').dt.tz_localize(None) - pd.Timestamp('1970-01-01')).dt.total_seconds()).to_numpy()
    def inside(s_, e_):
        mid = (np.asarray(s_) + np.asarray(e_)) / 2
        return int(sum(any(a <= m_ <= b for a, b in cover[p]) for p, m_ in zip(ev['pid'], mid)))
    checks = {}
    for tz in ['America/Chicago', 'UTC-6', 'UTC-5', 'UTC']:
        checks[tz] = inside(local_to_utc(ev['s_naive'].to_numpy(), tz), local_to_utc(ev['e_naive'].to_numpy(), tz))
    print('event midpoints inside recordings by time-zone rule:', checks, 'of', len(ev))
    ev['s'] = local_to_utc(ev['s_naive'].to_numpy(), 'America/Chicago')
    ev['e'] = local_to_utc(ev['e_naive'].to_numpy(), 'America/Chicago')
    ev = ev[np.isfinite(ev['s']) & np.isfinite(ev['e'])]
    best = ('America/Chicago', checks['America/Chicago'])
    lvl_code = lambda v: 1 if v in ('1', '2', '1.0', '2.0') else (2 if v in ('0', '0.0') else 3)
    ev['code'] = ev['level'].map(lvl_code)
    json.dump(dict(timezone=best[0], events_inside=best[1], checks=checks, events_total=int(len(ev)),
                   n_sessions=len(sess), participants=sorted(frames)), open(os.path.join(out, 'alignment.json'), 'w'))
    for pid, L in frames.items():
        L = sorted(L, key=lambda x: x[1]['t0'])
        X = np.vstack([r['X'] for _, r in L]); t = np.concatenate([r['t_end'] for _, r in L])
        sid = np.concatenate([np.full(len(r['t_end']), i) for i, (_, r) in enumerate(L)])
        lab = np.zeros(len(t), int)
        E = ev[ev['pid'] == pid]
        for code in (3, 2, 1):                       # stress (1) overrides other codes
            for _, r in E[E['code'] == code].iterrows():
                lab[(t >= r['s']) & (t < r['e'])] = code
        np.savez_compressed(os.path.join(out, f'{pid}_frames.npz'), X=X, names=np.array(L[0][1]['names']), t_end=t,
                            session=sid, label=lab, acc_std5=np.concatenate([r['acc_std5'] for _, r in L]),
                            eda_mean5=np.concatenate([r['eda_mean5'] for _, r in L]),
                            events=E[['s', 'e', 'code']].to_numpy(float))
        u, c = np.unique(lab, return_counts=True)
        print(pid, 'sessions', len(L), 'hours', round(len(t) * STEP / 3600, 1), 'labels', dict(zip(u.tolist(), c.tolist())),
              'events (stress/no/na)', [int((E['code'] == k).sum()) for k in (1, 2, 3)])
