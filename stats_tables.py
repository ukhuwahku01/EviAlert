import json, numpy as np
from scipy import stats
BK=['resnet50','vit_b16','convnext_t','swin_t','effnet_b0']
NAMES={'resnet50':'ResNet-50','vit_b16':'ViT-B/16','convnext_t':'ConvNeXt-T','swin_t':'Swin-T','effnet_b0':'EfficientNet-B0'}
BASE=['Frame','MajVote','MovAvg','SelMovAvg','MovAvg-Cal']
out={}
for b in BK:
    R=json.load(open(f'results/main_{b}.json'))
    def arr(m,k): return np.array([r['methods'][m][k] for r in R],float)
    def arr2(m,rule,k): return np.array([r['methods'][m][rule][k] for r in R],float)
    fr={k:np.array([r['frame'][k] for r in R]) for k in R[0]['frame']}
    meta={k:np.array([r['meta'][k] for r in R]) for k in ['distress_time_share','missing_share','events_per_hour','same_identity_share','n_cal_img','n_eval_img']}
    best=max(BASE,key=lambda m: arr(m,'recall@1.0').mean())
    tests={}
    for x in ['recall@0.5','recall@1.0','recall@2.0']:
        a=arr('EviAlert-W',x); bb=np.vstack([arr(m,x) for m in BASE])
        bestx=BASE[int(np.argmax(bb.mean(1)))]
        d=a-arr(bestx,x)
        p=stats.wilcoxon(a,arr(bestx,x)).pvalue if np.any(d!=0) else 1.0
        tests[x]=dict(best_baseline=bestx,diff=float(d.mean()),wins=int((d>0).sum()),ties=int((d==0).sum()),p=float(p))
    out[b]=dict(frame={k:[float(v.mean()),float(v.std())] for k,v in fr.items()},
                meta={k:float(v.mean()) for k,v in meta.items()},
                tests=tests,
                methods={m:dict(r05=float(arr(m,'recall@0.5').mean()),r05sd=float(arr(m,'recall@0.5').std()),
                                 r1=float(arr(m,'recall@1.0').mean()),r1sd=float(arr(m,'recall@1.0').std()),
                                 r2=float(arr(m,'recall@2.0').mean()),r2sd=float(arr(m,'recall@2.0').std()),
                                 delay=float(np.nanmean(arr2(m,'oracle','delay_med'))),
                                 stale_or=float(arr2(m,'oracle','stale').mean()),
                                 **{f'{rule}_{k}':float(np.nanmean(arr2(m,rule,k))) for rule in ['naive','poisson','gboot'] for k in ['fa','recall','compliant','feasible']},
                                 gboot_stale=float(arr2(m,'gboot','stale').mean()), gboot_ie=float(arr2(m,'gboot','insuff_share').mean()),
                                 lead=float(np.nanmean(arr(m,'watch_lead_med'))) if 'watch_lead_med' in R[0]['methods'][m] else None,
                                 watch_share=float(np.nanmean(arr(m,'watch_share'))) if 'watch_share' in R[0]['methods'][m] else None)
                         for m in R[0]['methods']})
    print(b, {k:(v['best_baseline'],round(v['diff'],3),v['wins'],f"{v['p']:.4f}") for k,v in tests.items()})
# pooled tests across backbones (100 paired splits) vs per-backbone best baseline at R@1
for x in ['recall@0.5','recall@1.0','recall@2.0']:
    A=[];B=[]
    for b in BK:
        R=json.load(open(f'results/main_{b}.json'))
        bestx=out[b]['tests'][x]['best_baseline']
        A+= [r['methods']['EviAlert-W'][x] for r in R]; B+=[r['methods'][bestx][x] for r in R]
    A=np.array(A);B=np.array(B)
    print('pooled',x,'diff',round((A-B).mean(),3),'wins',int((A>B).sum()),'/100 p',stats.wilcoxon(A,B).pvalue)
    out.setdefault('pooled',{})[x]=dict(diff=float((A-B).mean()),wins=int((A>B).sum()),ties=int((A==B).sum()),p=float(stats.wilcoxon(A,B).pvalue))
# pooled vs MovAvg-Cal (fixed comparator)
for x in ['recall@0.5','recall@1.0','recall@2.0']:
    A=[];B=[]
    for b in BK:
        R=json.load(open(f'results/main_{b}.json'))
        A+= [r['methods']['EviAlert-W'][x] for r in R]; B+=[r['methods']['MovAvg-Cal'][x] for r in R]
    A=np.array(A);B=np.array(B)
    out['pooled'][x+'_vs_MovAvgCal']=dict(diff=float((A-B).mean()),rel=float((A.mean()-B.mean())/B.mean()),wins=int((A>B).sum()),p=float(stats.wilcoxon(A,B).pvalue))
    print('vs MovAvg-Cal',x,out['pooled'][x+'_vs_MovAvgCal'])
# compliance pooled
for m in ['MovAvg','MovAvg-Cal','SelMovAvg','EviAlert-W','EviAlert-C']:
    c={rule:np.mean([out[b]['methods'][m][f'{rule}_compliant'] for b in BK]) for rule in ['naive','poisson','gboot']}
    print('compliance',m,{k:round(v,3) for k,v in c.items()})
# stale pooled
for m in ['MovAvg','MovAvg-Cal','SelMovAvg','EviAlert-W','EviAlert-W-noGate']:
    print('stale gboot',m, round(np.mean([out[b]['methods'][m]['gboot_stale'] for b in BK]),2))
json.dump(out,open('results/summary_tables.json','w'),indent=1)
