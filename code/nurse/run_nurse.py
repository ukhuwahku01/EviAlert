"""Confirmatory evaluation of EviAlert v2 on the nurse stress dataset (Hosseini et al., 2022).
Pre-registered: see PREREGISTRATION.md. Usage: python run_nurse.py <features_dir> <out_json> [models]"""
import os, sys, json, time, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from evialert import core
from evialert import realstream as RS

METHODS = ['Frame', 'MajVote', 'MovAvg', 'SelMovAvg', 'MovAvg-Cal', 'EviAlert-W', 'EviAlert-W-noClip', 'EviAlert-F']
TARGET = [1]                      # class index of stress in the binary classifier
MIN_EVENT, MERGE_GAP, GRACE, SUB = 9, 12, 60, 6
MOTION_G = 0.25                   # unusable if SD of acceleration magnitude over 5 s > 0.25 g (g = participant median magnitude)


def load(feat_dir):
    S = {}
    for f in sorted(glob.glob(os.path.join(feat_dir, '*_frames.npz'))):
        d = dict(np.load(f, allow_pickle=False)); pid = os.path.basename(f).replace('_frames.npz', '')
        S[pid] = d
    return S


def runs(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out = []; s = idx[0]; prev = idx[0]
    for i in idx[1:]:
        if i - prev > MERGE_GAP + 1:
            out.append((int(s), int(prev + 1))); s = i
        prev = i
    out.append((int(s), int(prev + 1)))
    return [(a, b) for a, b in out if b - a >= MIN_EVENT]


def make_clf(model):
    if model == 'LR':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if model == 'RF':
        return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0))
    return make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(random_state=0))


def lopo_probs(S, model):
    P = {}
    for p in S:
        Xs, ys = [], []
        for q in S:
            if q == p:
                continue
            lab = S[q]['label']; keep = np.isin(lab, [0, 1, 2]); idx = np.where(keep)[0][::SUB]
            Xs.append(S[q]['X'][idx]); ys.append((lab[idx] == 1).astype(int))
        clf = make_clf(model).fit(np.vstack(Xs), np.concatenate(ys))
        P[p] = clf.predict_proba(S[p]['X'])
    return P


def build(S, P):
    data = {}
    for p, d in S.items():
        lab = d['label']; ses = d['session']
        g = np.nanmedian(d['X'][:, list(d['names']).index('acc_mag_mean')])
        usable_all = d['acc_std5'] <= MOTION_G * g
        streams = []
        for k in np.unique(ses):
            ix = np.where(ses == k)[0]
            l = lab[ix]
            streams.append(dict(P=P[p][ix], y=np.where(np.isin(l, [0, 1, 2]), (l == 1).astype(int), -1),
                                use=usable_all[ix], events=runs(l == 1), famask=(l != 3), target=(l == 1)))
        data[p] = streams
    return data


if __name__ == '__main__':
    S = load(sys.argv[1]); models = sys.argv[3].split(',') if len(sys.argv) > 3 else ['LR', 'RF', 'HGB']
    meta = {p: dict(hours=float(len(d['t_end']) * 5 / 3600), sessions=int(len(np.unique(d['session']))),
                    events=int(sum(len(runs(d['label'][d['session'] == k] == 1)) for k in np.unique(d['session']))),
                    label_counts={int(a): int(b) for a, b in zip(*np.unique(d['label'], return_counts=True))}) for p, d in S.items()}
    out = dict(meta=meta, models={})
    for model in models:
        t0 = time.time(); P = lopo_probs(S, model)
        y = np.concatenate([S[p]['label'] for p in S]); pr = np.concatenate([P[p][:, 1] for p in S]); k = np.isin(y, [0, 1, 2])
        auc = float(roc_auc_score(y[k] == 1, pr[k]))
        folds = RS.lopo(build(S, P), METHODS, TARGET, grace=GRACE)
        out['models'][model] = dict(auroc=auc, summary=RS.summarize(folds, METHODS), filt=[f['filt'] | {'held': f['held'], 'T': f['T'], 'prior': f['prior']} for f in folds])
        print(model, 'AUROC', round(auc, 3), 'done', round(time.time() - t0), 's', flush=True)
        json.dump(out, open(sys.argv[2], 'w'))
