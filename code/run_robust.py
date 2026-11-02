import os
import json, sys, time, numpy as np
sys.path.insert(0, '.')
from evialert.experiments import load_oof, evaluate_split
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
which = sys.argv[1]
ms = ['MovAvg', 'MovAvg-Cal', 'EviAlert-W', 'EviAlert-W-noGate']
out = {}
t0 = time.time()
for model in ['resnet50', 'convnext_t']:
    P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', model)
    if which == 'missing':
        for r in [0.0, 0.1, 0.2, 0.3]:
            cfg = dict(miss_rate=r)
            rows = [evaluate_split(P, y, g, seed=s, methods=ms, n_boot=5, n_per_boot=60, cfg_cal=cfg, cfg_eval=cfg)[1] for s in range(201, 211)]
            out[f'{model}|{r}'] = rows
    elif which == 'persist':
        for pm in [1.0, 2.0, 4.0]:
            cfg = dict(persist_mean=pm)
            rows = [evaluate_split(P, y, g, seed=s, methods=ms, n_boot=5, n_per_boot=60, cfg_cal=cfg, cfg_eval=cfg)[1] for s in range(201, 211)]
            out[f'{model}|{pm}'] = rows
    elif which == 'budget':
        for b in [0.5, 2.0]:
            rows = [evaluate_split(P, y, g, seed=s, methods=['MovAvg', 'MovAvg-Cal', 'EviAlert-W'], budget=b)[1] for s in range(101, 121)]
            out[f'{model}|{b}'] = rows
json.dump(out, open(f'results/robust_{which}.json', 'w'))
print(which, 'done', round(time.time() - t0), 's')
