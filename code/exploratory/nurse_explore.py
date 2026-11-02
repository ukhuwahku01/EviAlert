"""Exploratory analyses on the nurse dataset (declared in PREREGISTRATION.md, addendum 2; not part of H1-H3):
untempered filter, person-level CRC, (alpha, delta) grid and calibration-population size.
Usage: python exploratory/nurse_explore.py <features_dir> <out_json>"""
import os, sys, json, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'nurse')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_nurse as RN
import lopo_curves as LC

METHODS = ['MovAvg-Cal', 'MovAvg-Cal-wide', 'EviAlert-F', 'EviAlert-F-g1']

if __name__ == '__main__':
    S = RN.load(sys.argv[1]); out = {}
    for model in ['LR', 'RF', 'HGB']:
        t0 = time.time(); P = RN.lopo_probs(S, model)
        folds = LC.lopo_raw(RN.build(S, P), METHODS, RN.TARGET, grace=RN.GRACE)
        out[model] = LC.analyse(folds, METHODS)
        print(model, 'done', round(time.time() - t0), 's', flush=True)
        json.dump(out, open(sys.argv[2], 'w'), default=float)
