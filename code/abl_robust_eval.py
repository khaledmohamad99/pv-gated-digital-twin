import sys,json; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import *
R=[]
for name,clean,pert,aug in [('A0 no calibration','lazz_feat_abl_A0.parquet','lazz_feat_abl_A0_{}.parquet',False),('A5 final twin (calibrated + augmentation)','lazz_feat2.parquet','lazz_feat_{}.parquet',True)]:
    D={'clean':pd.read_parquet(clean)}
    for t in ['g090','g110']: D[t]=pd.read_parquet(pert.format(t))
    rng=np.random.default_rng(0); Y={c:[] for c in D}; P={c:[] for c in D}
    for k in range(9,17):
        tr=D['clean'][D['clean'].day!=k]; tr=tr[(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)]
        X=tr[PHYS].values; y=tr.y.values
        if aug: X=phys_augment(X,PHYS,rng); y=np.tile(y,3)
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0).fit(X,y)
        for c,df in D.items():
            te=df[df.day==k]; P[c].append(m.predict(te[PHYS].values)); Y[c].append(te.y.values)
    for c in D:
        R.append(dict(config=name,condition=c,macroF1=f1_score(np.concatenate(Y[c]),np.concatenate(P[c]),average='macro'))); print(R[-1],flush=True)
pd.DataFrame(R).to_csv('./res_ablation_robust.csv',index=False)
