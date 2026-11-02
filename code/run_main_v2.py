"""EviAlert v2 (EviAlert-F) on the frozen test splits (seeds 101-120); comparators are in results/main_<model>.json."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evialert.experiments import load_oof, evaluate_split
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
model = sys.argv[1]
P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', model)
res = []; t0 = time.time()
for s in range(101, 121):
    fr, out, meta = evaluate_split(P, y, g, seed=s, methods=['EviAlert-F'], budget=1.0, keep_curves=True)
    res.append(dict(methods=out, meta=meta))
json.dump(res, open(f'results/main_v2_{model}.json', 'w'))
print(model, 'done in', round(time.time() - t0), 's')
