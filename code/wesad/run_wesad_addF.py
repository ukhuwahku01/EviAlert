"""Adds EviAlert v2 (EviAlert-F) to the 200-split WESAD analysis (primary scenario); comparators are unchanged
(identical random streams), MovAvg-Cal is re-run as a consistency check."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import run_wesad as R
R.METHODS = ['MovAvg-Cal', 'EviAlert-F']
def job(model):
    S = R.load(sys.argv[1]); P = R.loso_probs(S, model); t0 = time.time()
    res = R.evaluate(S, P, n_splits=200, miss_rate=0.10, scoring='all', keep_curves=True)
    print(model, 'done', round(time.time() - t0), flush=True)
    return model, res
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(3) as p:
        out = dict(p.map(job, ['LR', 'RF', 'HGB']))
    json.dump(out, open(sys.argv[2], 'w'))
