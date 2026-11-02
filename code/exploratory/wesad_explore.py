"""Exploratory analyses on WESAD (leave-one-participant-out, primary scenario): untempered filter, person-level
CRC, (alpha, delta) grid and calibration-population size. Usage: python exploratory/wesad_explore.py <features_dir> <out_json>"""
import os, sys, json, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'wesad')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_wesad as RW
import run_wesad_v2 as RW2
import lopo_curves as LC

METHODS = ['MovAvg-Cal', 'MovAvg-Cal-wide', 'EviAlert-W', 'EviAlert-F', 'EviAlert-F-g1']

if __name__ == '__main__':
    S = RW.load(sys.argv[1]); out = {}
    for model in ['LR', 'RF', 'HGB']:
        t0 = time.time(); P = RW.loso_probs(S, model)
        folds = LC.lopo_raw(RW2.build(S, P, 0.10), METHODS, RW.D)
        out[model] = LC.analyse(folds, METHODS)
        print(model, 'done', round(time.time() - t0), 's', flush=True)
    json.dump(out, open(sys.argv[2], 'w'), default=float)
