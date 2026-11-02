import os
"""Main evaluation: 5 backbones x 20 fresh identity splits (seeds 101-120; development used seeds 0-10)."""
import json, sys, time
import numpy as np
sys.path.insert(0, '.')
from evialert.experiments import load_oof, evaluate_split, MAIN, ABL
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
model = sys.argv[1]
P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', model)
res = []
t0 = time.time()
for s in range(101, 121):
    fr, out, meta = evaluate_split(P, y, g, seed=s, methods=MAIN + ABL, budget=1.0, keep_curves=True)
    res.append(dict(frame=fr, methods=out, meta=meta))
json.dump(res, open(f'results/main_{model}.json', 'w'))
print(model, 'done in', round(time.time() - t0), 's')
