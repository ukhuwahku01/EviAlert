import os
import json, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from evialert.experiments import load_oof
from evialert.core import ece
BK=['resnet50','vit_b16','convnext_t','swin_t','effnet_b0']
NM={'resnet50':'ResNet-50','vit_b16':'ViT-B/16','convnext_t':'ConvNeXt-T','swin_t':'Swin-T','effnet_b0':'EfficientNet-B0'}
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')); RUNS = U + '/runs'; IDX = U + '/clean_index.csv'
S=json.load(open('results/summary_tables.json'))
f=lambda x,d=3: f'{x:.{d}f}'.lstrip('0') if abs(x)<1 else f'{x:.{d}f}'
out=[]
# Table: frame-level + calibration
out.append('TABLE frame')
out.append('| Backbone | Accuracy | Macro-F1 | AUROC (distress) | T* | ECE raw | ECE cal | NLL raw | NLL cal |')
for b in BK:
    P,y,g=load_oof(RUNS,IDX,b)
    pd_=P[:,0]+P[:,1]; yd=np.isin(y,[0,1])
    fr=S[b]['frame']
    out.append(f"| {NM[b]} | {f(accuracy_score(y,P.argmax(1)))} | {f(f1_score(y,P.argmax(1),average='macro'))} | {f(roc_auc_score(yd,pd_))} | {fr['T'][0]:.2f} | {f(fr['ece_raw'][0])} | {f(fr['ece_cal'][0])} | {fr['nll_raw'][0]:.3f} | {fr['nll_cal'][0]:.3f} |")
# Table main recall
M=['Frame','MajVote','MovAvg','SelMovAvg','MovAvg-Cal','EviAlert-C','EviAlert-W']
out.append('TABLE recall')
for b in BK:
    for m in M:
        r=S[b]['methods'][m]
        out.append(f"| {NM[b]} | {m} | {f(r['r05'])} ({f(r['r05sd'])}) | {f(r['r1'])} ({f(r['r1sd'])}) | {f(r['r2'])} ({f(r['r2sd'])}) | {r['delay']:.1f} |")
out.append('TABLE tests')
for b in BK:
    out.append(NM[b]+' '+json.dumps({k:(v['best_baseline'],round(v['diff'],3),v['wins'],v['ties'],f"{v['p']:.2g}") for k,v in S[b]['tests'].items()}))
out.append(json.dumps(S['pooled']))
out.append('TABLE deployed (gboot) + compliance')
for b in BK:
    for m in ['MovAvg','MovAvg-Cal','SelMovAvg','EviAlert-C','EviAlert-W']:
        r=S[b]['methods'][m]
        out.append(f"| {NM[b]} | {m} | naive {r['naive_compliant']:.2f} | pois {r['poisson_compliant']:.2f} | gboot {r['gboot_compliant']:.2f} feas {r['gboot_feasible']:.2f} | FA {r['gboot_fa']:.2f} | rec {f(r['gboot_recall'])} | stale {r['gboot_stale']:.2f} | ie {r['gboot_ie']:.3f} | naiveFA {r['naive_fa']:.2f} poisFA {r['poisson_fa']:.2f}")
for m in ['Frame','MajVote']:
    for b in BK:
        r=S[b]['methods'][m]; out.append(f"{m} {NM[b]} gboot feas {r['gboot_feasible']:.2f} comp {r['gboot_compliant']:.2f} naive feas {r['naive_feasible']:.2f}")
pool=lambda m,k: np.mean([S[b]['methods'][m][k] for b in BK])
out.append('POOLED compliance/stale/FA/recall (gboot)')
for m in ['Frame','MajVote','MovAvg','SelMovAvg','MovAvg-Cal','EviAlert-C','EviAlert-W','EviAlert-W-noCal','EviAlert-W-noClip','EviAlert-W-noGate']:
    out.append(f"{m}: naive {pool(m,'naive_compliant'):.3f} pois {pool(m,'poisson_compliant'):.3f} gboot {pool(m,'gboot_compliant'):.3f} | FA {pool(m,'gboot_fa'):.3f} rec {pool(m,'gboot_recall'):.3f} stale {pool(m,'gboot_stale'):.2f} staleOr {pool(m,'stale_or'):.2f} ie {pool(m,'gboot_ie'):.3f} feas {pool(m,'gboot_feasible'):.2f}")
out.append('TABLE ablation')
for b in BK:
    for m in ['EviAlert-W','EviAlert-W-noCal','EviAlert-W-noClip','EviAlert-W-noGate']:
        r=S[b]['methods'][m]
        out.append(f"| {NM[b]} | {m} | {f(r['r05'])} | {f(r['r1'])} | {f(r['r2'])} | {r['delay']:.1f} | staleOr {r['stale_or']:.2f} | gstale {r['gboot_stale']:.2f} | gfeas {r['gboot_feasible']:.2f} | gcomp {r['gboot_compliant']:.2f}")
out.append('LEAD')
for b in BK:
    r=S[b]['methods']['EviAlert-W']; out.append(f"{NM[b]} lead {r['lead']:.1f} watch {r['watch_share']:.3f}")
out.append('META '+json.dumps({b:S[b]['meta'] for b in BK}))
# robustness
for fn in ['missing','persist','budget']:
    d=json.load(open(f'results/robust_{fn}.json'))
    out.append('ROBUST '+fn)
    for key,L in d.items():
        for m in L[0].keys():
            R1=np.mean([e[m]['recall@1.0'] for e in L]); 
            st=np.mean([e[m]['gboot']['stale'] for e in L]); comp=np.mean([e[m]['gboot']['compliant'] for e in L])
            ie=np.mean([e[m]['gboot'].get('insuff_share',np.nan) for e in L])
            fa=np.mean([e[m]['gboot']['fa'] for e in L]); rec=np.mean([e[m]['gboot']['recall'] for e in L])
            out.append(f"{key} {m}: R@.5 {np.mean([e[m]['recall@0.5'] for e in L]):.3f} R@1 {R1:.3f} R@2 {np.mean([e[m]['recall@2.0'] for e in L]):.3f} gstale {st:.2f} gcomp {comp:.2f} gFA {fa:.2f} grec {rec:.3f} ie {ie:.3f} n={len(L)}")
open('results/numbers.txt','w').write('\n'.join(out))
print('\n'.join(out))
