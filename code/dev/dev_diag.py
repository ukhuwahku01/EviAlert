import os
import numpy as np, sys
sys.path.insert(0,'.')
from evialert import core
from evialert.experiments import load_oof, window_sum, window_mean, locf, GRID01
from evialert.simulate import make_sessions, DISTRESS
from sklearn.metrics import roc_auc_score
U = os.environ.get('INFANTFER_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results'))
for model in ['resnet50','convnext_t']:
    P,y,g=load_oof(U+'/runs',U+'/clean_index.csv',model)
    d=np.isin(y,DISTRESS)
    print(model,'frame AUROC distress', round(roc_auc_score(d, core.target_prob(P,DISTRESS)),3))
    rng=np.random.default_rng(0); ug=np.unique(g); rng.shuffle(ug); cg=set(ug[:len(ug)//2])
    ic=np.array([x in cg for x in g]); ci=np.where(ic)[0]; ei=np.where(~ic)[0]
    Ts=core.fit_temperature(P[ci],y[ci]); Pc=core.temperature_scale(P,Ts); pr=d[ci].mean()
    L=core.evidence(Pc,DISTRESS,pr,None)
    for c in range(4): print('  class',c,'mean LLR',round(L[ei][y[ei]==c].mean(),2),'sd',round(L[ei][y[ei]==c].std(),2))
    Sc=make_sessions(ci,y,g,300,rng); Se=make_sessions(ei,y,g,600,rng)
    T=180; res=[]
    def run(name,Ecal,Eev,grid,acc,gate,usec,usee):
        on,_=core.run_policy(Ecal,usec,grid,refractory=24,gate=gate,accumulate=acc)
        sc=core.score_alerts(on,Sc['events'],T,5,6)
        h=core.ltt_threshold(grid,sc['fa'],sc['hours'],1.0)
        ub=core.poisson_ucb(sc['fa'][list(grid).index(h)],sc['hours'])
        on2,_=core.run_policy(Eev,usee,[h],refractory=24,gate=gate,accumulate=acc)
        s2=core.score_alerts(on2,Se['events'],T,5,6)
        print(f'  {name:28s} h={h:7.3f} ucb_cal={ub:5.2f} FA/h={s2["fa"][0]/s2["hours"]:5.2f} rec={s2["det"][0]/s2["tot"]:.3f} delay={np.median(s2["delays"][0]) if s2["delays"][0] else np.nan}')
    for W in [4,6,8,12]:
        pdc=locf(core.target_prob(P[Sc['img']],DISTRESS),Sc['usable']); pde=locf(core.target_prob(P[Se['img']],DISTRESS),Se['usable'])
        run(f'MovAvg W={W}',window_mean(pdc,W),window_mean(pde,W),GRID01,False,False,Sc['usable'],Se['usable'])
    for c in [1,2,3]:
        Ec=core.evidence(Pc[Sc['img']],DISTRESS,pr,c); Ee=core.evidence(Pc[Se['img']],DISTRESS,pr,c)
        for W in [4,6,8,12]:
            run(f'WinSum c={c} W={W}',window_sum(Ec,W,Sc['usable']),window_sum(Ee,W,Se['usable']),np.round(np.linspace(0.3,3*W,60),3),False,True,Sc['usable'],Se['usable'])
        for k in [0,0.25,0.5,1.0,1.5]:
            Gc=np.round(np.geomspace(0.3,40,60),3)
            on=None
            run(f'CUSUM c={c} k={k}',Ec-k,Ee-k,Gc,True,True,Sc['usable'],Se['usable'])
