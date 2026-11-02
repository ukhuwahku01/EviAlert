"""Exploratory (added after the design was frozen): the final filter without tempering (gamma fixed at 1)
on the same infant test splits (seeds 101-120). Frozen modules are not modified; the candidate set of
tempering values is replaced at run time. Usage: python exploratory/infant_gamma1.py <model>"""
import json, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from evialert import experiments as EX
EX.GAMMAS = (1.0,)
U = os.environ.get('INFANTFER_RESULTS', os.path.join(ROOT, '..', 'results'))
model = sys.argv[1]
P, y, g = EX.load_oof(U + '/runs', U + '/clean_index.csv', model)
res = []; t0 = time.time()
for s in range(101, 121):
    fr, out, meta = EX.evaluate_split(P, y, g, seed=s, methods=['EviAlert-F'], budget=1.0, keep_curves=False)
    res.append(dict(methods=out, meta=meta))
json.dump(res, open(os.path.join(ROOT, 'results', f'explore_gamma1_{model}.json'), 'w'))
print(model, 'done in', round(time.time() - t0), 's')
