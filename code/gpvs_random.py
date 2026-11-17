"""Leaky random 70/30 window split on GPVS-Faults (reported only to quantify leakage); 5 random seeds."""
import sys,json; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
exec(open('../code/gpvs_cls.py').read().split('K=8')[0])
y=F.y.values; out={'STAT':[],'PHYS':[]}
for seed in range(5):
    tr,te=train_test_split(np.arange(len(F)),test_size=0.3,random_state=seed,stratify=y)
    for fs in out:
        X=F[FS[fs]].values.astype(np.float32)
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=2,random_state=0).fit(X[tr],y[tr])
        out[fs].append(f1_score(y[te],m.predict(X[te]),average='macro'))
res={fs:dict(macroF1=float(np.mean(v)),macroF1_sd=float(np.std(v)),runs=v) for fs,v in out.items()}
print({k:(round(v['macroF1'],3),round(v['macroF1_sd'],3)) for k,v in res.items()})
json.dump(res,open('res_gpvs_random.json','w'),indent=1)
