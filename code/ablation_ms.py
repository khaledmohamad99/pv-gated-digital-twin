"""Multi-seed ablation (3 seeds x leave-one-day-out). Seed controls the training subsample and the model."""
import sys,json; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score
from common import *
cfgs=[('A0 datasheet twin (no calibration)','lazz_feat_abl_A0.parquet',False,PHYS),
      ('A1 + online calibration, no physics gate','lazz_feat_abl_A1.parquet',False,PHYS),
      ('A2 + common/differential physics gate','lazz_feat_abl_A2.parquet',False,PHYS),
      ('A3 + change-point re-anchoring','lazz_feat_abl_A3.parquet',False,PHYS),
      ('A4 + fast/morning re-anchoring','lazz_feat_abl_A4.parquet',False,PHYS),
      ('RAW reference','lazz_feat2.parquet',False,RAWW)]
R=[]
for seed in [0,1,2]:
    for name,f,aug,cols in cfgs:
        d=pd.read_parquet(f); rng=np.random.default_rng(seed); pred=np.full(len(d),-1)
        for k in sorted(d.day.unique()):
            tr=d[d.day!=k]; tr=tr[(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)]; te=np.flatnonzero(d.day.values==k)
            X=tr[cols].values; y=tr.y.values
            if aug: X=phys_augment(X,cols,rng); y=np.tile(y,3)
            m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=seed).fit(X,y)
            pred[te]=m.predict(d[cols].values[te])
        yt=d.y.values; pc=f1_score(yt,pred,average=None)
        R.append(dict(seed=seed,config=name,macroF1=f1_score(yt,pred,average='macro'),F1_SC=pc[1],F1_DG=pc[2],F1_SH=pc[4],
                      day9_DG_recall=float((pred[(d.day.values==9)&(yt==2)]==2).mean())))
        print(R[-1],flush=True); pd.DataFrame(R).to_csv('./res_ablation_ms.csv',index=False)
R=pd.DataFrame(R); print(R.groupby('config')[['macroF1','F1_DG','day9_DG_recall']].agg(['mean','std']).round(4).to_string())
