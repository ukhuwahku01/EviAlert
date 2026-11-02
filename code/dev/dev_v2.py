"""Development of EviAlert v2 on development seeds 0-10 only (no test seeds)."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evialert.experiments import load_oof, evaluate_split
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results'))
M = ['MovAvg-Cal', 'EviAlert-W', 'EviAlert-W-noClip', 'EviAlert-F']
cfg = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
out = {}
for model in sys.argv[1].split(','):
    P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', model)
    rows = []
    for seed in range(0, 11):
        fr, res, meta = evaluate_split(P, y, g, seed=seed, methods=M, budget=1.0, cfg_cal=cfg, cfg_eval=cfg)
        rows.append((res, meta))
    print('==', model, 'cfg', cfg, 'gamma', [m['filt']['gamma'] for _, m in rows], 'a,b', np.round([rows[0][1]['filt']['a'], rows[0][1]['filt']['b']], 4))
    for m in M:
        R = [r for r, _ in rows]
        print(f"  {m:18s}", [round(np.mean([r[m][f'recall@{x}'] for r in R]), 3) for x in (0.5, 1.0, 2.0)],
              'gboot comp', round(np.mean([r[m]['gboot']['compliant'] for r in R]), 2), 'stale', round(np.mean([r[m]['gboot']['stale'] for r in R]), 1),
              'delay', round(np.nanmean([r[m]['oracle']['delay_med'] for r in R]), 1))
