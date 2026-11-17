import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from common import *
d=pd.read_parquet('lazz_feat2.parquet')
rng=np.random.default_rng(0)
def models():
    return {
     'LightGBM':lambda: lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0),
     'XGBoost':lambda: XGBClassifier(n_estimators=300,learning_rate=0.1,max_depth=6,n_jobs=1,random_state=0,tree_method='hist'),
     'RandomForest':lambda: RandomForestClassifier(n_estimators=100,min_samples_leaf=2,max_samples=0.3,class_weight='balanced',n_jobs=1,random_state=0),
     'MLP':lambda: make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=200,early_stopping=True,random_state=0)),
    }
FS={'RAW':RAWW,'PHYS':PHYS,'PHYS+PA':PHYS,'HYBRID':sorted(set(RAWW+PHYS))}
def train_idx(tr):
    # subsample abundant classes for speed; all fault samples kept
    keep=(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)
    return tr[keep]
def sw(y):
    c=np.bincount(y,minlength=5).astype(float); w=len(y)/(5*np.maximum(c,1)); return w[y]
def run(protocol,fs,mname):
    X=d[FS[fs]].values.astype(np.float32); y=d.y.values
    pred=np.full(len(d),-1)
    if protocol=='LODO':
        folds=[(np.flatnonzero(d.day!=k),np.flatnonzero(d.day==k)) for k in sorted(d.day.unique())]
    else:  # random split (leaky, for contrast)
        p=rng.permutation(len(d)); n=int(.8*len(d)); folds=[(p[:n],p[n:])]
    for tr,te in folds:
        trd=d.iloc[tr]; trd=train_idx(trd); ix=trd.index.values
        m=models()[mname]()
        Xtr,ytr=X[ix],y[ix]
        if fs=='PHYS+PA': Xtr=phys_augment(Xtr,FS[fs],rng); ytr=np.tile(ytr,3)
        if mname=='XGBoost': m.fit(Xtr,ytr,sample_weight=sw(ytr))
        else: m.fit(Xtr,ytr)
        pred[te]=m.predict(X[te])
    msk=pred>=0
    yt,yp=y[msk],pred[msk]
    return dict(protocol=protocol,features=fs,model=mname,acc=accuracy_score(yt,yp),bacc=balanced_accuracy_score(yt,yp),
                macroF1=f1_score(yt,yp,average='macro'),perclassF1=f1_score(yt,yp,average=None).tolist(),
                cm=confusion_matrix(yt,yp,labels=range(5)).tolist(),
                det_acc=accuracy_score(yt>0,yp>0)),pred
res=[];P={}
PLAN=[('RAW','LightGBM'),('RAW','XGBoost'),('RAW','RandomForest'),('RAW','MLP'),
      ('PHYS+PA','LightGBM'),('PHYS+PA','XGBoost'),('PHYS+PA','RandomForest'),('PHYS+PA','MLP'),
      ('HYBRID','LightGBM'),('HYBRID','XGBoost'),('HYBRID','MLP')]
for fs,mn in PLAN:
    t=time.time(); r,p=run('LODO',fs,mn); r['sec']=time.time()-t; res.append(r); P[f'LODO_{fs}_{mn}']=p
    print('LODO',fs,mn,round(r['macroF1'],4),flush=True)
    json.dump(res,open('res_lazz_cls_all.json','w'),indent=1); np.savez_compressed('pred_lazz.npz',**P)
for fs in ['RAW','PHYS+PA']:   # leaky random split, reported only to quantify leakage
    r,p=run('RANDOM',fs,'LightGBM'); res.append(r); print('RANDOM',fs,round(r['macroF1'],4),flush=True)
json.dump(res,open('res_lazz_cls_all.json','w'),indent=1)
