"""Robustness to irradiance-sensor gain error (x0.90 / x1.10) at test time; models trained on clean data."""
import sys,json; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.neural_network import MLPClassifier; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import *
rng=np.random.default_rng(0)
D={'clean':pd.read_parquet('lazz_feat2.parquet')}
for t in ['g090','g110']: D[t]=add_raw_window(pd.read_parquet(f'lazz_feat_{t}.parquet'))
mk={'LightGBM':lambda: lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0),
    'MLP':lambda: make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=200,early_stopping=True,random_state=0))}
R=[]
for fs,cols in [('RAW',RAWW),('PHYS+PA',PHYS)]:
    for mn in mk:
        Y={c:[] for c in D}; Pp={c:[] for c in D}
        for k in range(9,17):
            tr=D['clean'][D['clean'].day!=k]; tr=tr[(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)]
            X=tr[cols].values; y=tr.y.values
            if fs=='PHYS+PA': X=phys_augment(X,cols,rng); y=np.tile(y,3)
            m=mk[mn]().fit(X,y)
            for c,df in D.items():
                te=df[df.day==k]; Pp[c].append(m.predict(te[cols].values)); Y[c].append(te.y.values)
        for c in D:
            yt=np.concatenate(Y[c]); yp=np.concatenate(Pp[c])
            R.append(dict(features=fs,model=mn,condition=c,macroF1=f1_score(yt,yp,average='macro'),perclass=f1_score(yt,yp,average=None).tolist()))
            print(R[-1]['features'],mn,c,round(R[-1]['macroF1'],4),np.round(R[-1]['perclass'],3),flush=True)
json.dump(R,open('res_lazz_robust.json','w'),indent=1)
