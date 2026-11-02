import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evialert.experiments import load_oof, evaluate_split
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
which = sys.argv[1]; out = {}
for model in ['resnet50', 'convnext_t']:
    P, y, g = load_oof(U + '/runs', U + '/clean_index.csv', model)
    vals = [0.0, 0.1, 0.2, 0.3] if which == 'missing' else [1.0, 2.0, 4.0]
    for v in vals:
        cfg = {'miss_rate': v} if which == 'missing' else {'persist_mean': v}
        out[f'{model}|{v}'] = [evaluate_split(P, y, g, seed=s, methods=['EviAlert-F'], n_boot=5, n_per_boot=60, cfg_cal=cfg, cfg_eval=cfg)[1] for s in range(201, 211)]
json.dump(out, open(f'results/robust_v2_{which}.json', 'w'))
print(which, 'done')
