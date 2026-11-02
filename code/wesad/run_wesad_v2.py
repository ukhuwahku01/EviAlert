"""WESAD with leave-one-person-out evaluation (development data for EviAlert v2).
Usage: python run_wesad_v2.py <features_dir> <out_json>"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_wesad as RW
from evialert import realstream as RS

METHODS = ['Frame', 'MajVote', 'MovAvg', 'SelMovAvg', 'MovAvg-Cal', 'EviAlert-W', 'EviAlert-W-noClip', 'EviAlert-F']


def build(S, P, miss_rate=0.10, seed=0):
    data = {}
    for k, s in enumerate(sorted(S)):
        lab = S[s]['label_end']; T = len(lab)
        rng = np.random.default_rng(seed + 1000 + k)
        use = (S[s]['acc_std5'] <= RW.MOTION_THR) & RW.markov_missing(T, rng, miss_rate)
        y = np.where(np.isin(lab, RW.CLASSES), np.searchsorted(RW.CLASSES, lab), -1)
        data[s] = [dict(P=P[s], y=y, use=use, events=RW.events_of(lab), famask=np.ones(T, bool), target=(lab == 2))]
    return data


if __name__ == '__main__':
    S = RW.load(sys.argv[1]); out = {}
    for model in ['LR', 'RF', 'HGB']:
        t0 = time.time(); P = RW.loso_probs(S, model)
        res = {}
        for name, mr in [('primary', 0.10), ('natural', 0.0)]:
            folds = RS.lopo(build(S, P, mr), METHODS, RW.D)
            res[name] = dict(summary=RS.summarize(folds, METHODS), filt=[f['filt'] for f in folds])
        out[model] = res
        print(model, 'done', round(time.time() - t0), 's', flush=True)
    json.dump(out, open(sys.argv[2], 'w'))
