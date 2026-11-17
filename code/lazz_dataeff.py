"""Data efficiency: train with only j labelled fault days (+ healthy/shading days 1-8), test on held-out fault days."""
import sys,json,itertools; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.neural_network import MLPClassifier; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import *
d=pd.read_parquet('lazz_feat2.parquet'); rng=np.random.default_rng(1)
FD=[12,13,14,15,16]   # days containing all four fault types
mk={'LightGBM':lambda: lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0),
    'MLP':lambda: make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=200,early_stopping=True,random_state=0))}
R=[]
for j in [1,2]:
    combos=list(itertools.combinations(FD,j))
    for comb in combos:
        trd=d[d.day.isin(list(range(1,9))+list(comb))]; ted=d[d.day.isin([k for k in FD if k not in comb])]
        trd=trd[(trd.y.isin([1,2,3]))|(rng.random(len(trd))<0.2)]
        for fs,cols in [('RAW',RAWW),('PHYS+PA',PHYS)]:
            for mn in mk:
                X=trd[cols].values; y=trd.y.values
                if fs=='PHYS+PA': X=phys_augment(X,cols,rng); y=np.tile(y,3)
                p=mk[mn]().fit(X,y).predict(ted[cols].values)
                R.append(dict(j=j,train=list(comb),features=fs,model=mn,macroF1=f1_score(ted.y,p,average='macro')))
                print(R[-1],flush=True)
R=pd.DataFrame(R); R.to_csv('res_lazz_dataeff.csv',index=False)
print(R.groupby(['j','features','model']).macroF1.agg(['mean','std','min']).round(4))
