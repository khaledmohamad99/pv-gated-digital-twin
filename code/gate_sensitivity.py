"""One-at-a-time sensitivity of the physics-gate / calibration hyper-parameters (final twin + PA, LightGBM, LODO).
For every setting the twin features are rebuilt causally on the clean signal and on the +/-10 % irradiance-gain signals;
the LODO model trained on clean days is evaluated on the held-out day (clean) and, for fault days 9-16, on the perturbed day."""
import sys,os,subprocess,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import PHYS, phys_augment
CODE=os.path.dirname(os.path.abspath(__file__))
NOM=dict(EPS_I='0.015',EPS_V='0.02',DELTA_I='0.15',HALF='240',BUF='10')
GRID=[('nominal',{})]+[(f'{k}={v}',{k:v}) for k,vals in [('EPS_I',['0.0075','0.03']),('EPS_V',['0.01','0.04']),
      ('DELTA_I',['0.10','0.25']),('HALF',['120','480']),('BUF',['5','20'])] for v in vals]
def build(tag,over,twinf,out):
    env=dict(os.environ,CALMODE='gated',REANCHOR='1',MORNING='0',TWINF=twinf,OUTF=out,**{**NOM,**over})
    subprocess.run([sys.executable,os.path.join(CODE,'lazz_features.py')],env=env,check=True,stdout=subprocess.DEVNULL)
PART=os.environ.get('PART','')   # '' = full grid; 'a'/'b' = two halves for two parallel processes
SEL={'':GRID,'a':GRID[:6],'b':GRID[6:]}[PART]
R=[]; OUT=f'res_gate_sensitivity{"_"+PART if PART else ""}.csv'
for tag,over in SEL:
    f={c:f'sens_tmp{PART}_{c}.parquet' for c in ['clean','g090','g110']}
    build(tag,over,'lazz_twin.parquet',f['clean'])
    for c in ['g090','g110']: build(tag,over,f'lazz_twin_{c}.parquet',f[c])
    D={c:pd.read_parquet(p) for c,p in f.items()}; d=D['clean']; rng=np.random.default_rng(0)
    pred=np.full(len(d),-1); Yr={c:[] for c in D}; Pr={c:[] for c in D}
    for k in sorted(d.day.unique()):
        tr=d[d.day!=k]; tr=tr[(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)]
        X=phys_augment(tr[PHYS].values,PHYS,rng); y=np.tile(tr.y.values,3)
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0).fit(X,y)
        te=np.flatnonzero(d.day.values==k); pred[te]=m.predict(d[PHYS].values[te])
        if k>=9:
            for c,dc in D.items():
                s=dc[dc.day==k]; Yr[c].append(s.y.values); Pr[c].append(m.predict(s[PHYS].values))
    yt=d.y.values; pc=f1_score(yt,pred,average=None)
    r=dict(setting=tag,macroF1=f1_score(yt,pred,average='macro'),F1_DG=pc[2],F1_SH=pc[4],
           day9_DG_recall=float((pred[(d.day.values==9)&(yt==2)]==2).mean()))
    for c in D: r[f'faultdays_{c}']=f1_score(np.concatenate(Yr[c]),np.concatenate(Pr[c]),average='macro')
    R.append(r); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
    pd.DataFrame(R).to_csv(OUT,index=False)
