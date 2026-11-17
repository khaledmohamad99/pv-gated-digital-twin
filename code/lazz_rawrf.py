import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from common import *
from scipy.stats import wilcoxon
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

r,p=run('LODO','RAW','RandomForest'); print('RAW RF',round(r['macroF1'],4),flush=True)
np.savez_compressed('pred_lazz_rawrf.npz',LODO_RAW_RandomForest=p)
json.dump([r],open('res_lazz_rawrf.json','w'),indent=1)

# 95 % day-bootstrap interval (300 resamples) and paired Wilcoxon test against PHYS+PA random forest (fault days 9-16)
y=d.y.values; day=d.day.values; b=np.random.default_rng(0); days=np.unique(day); B=[]
for _ in range(300):
    s=b.choice(days,len(days)); m=np.concatenate([np.flatnonzero(day==k) for k in s]); B.append(f1_score(y[m],p[m],average='macro'))
P=np.load('pred_lazz.npz')['LODO_PHYS+PA_RandomForest']
per=lambda q: np.array([f1_score(y[day==k],q[day==k],average='macro',labels=sorted(set(y[day==k]))) for k in range(9,17)])
z,x=per(P),per(p); r['ci95']=np.percentile(B,[2.5,97.5]).tolist(); r['vs_PHYS+PA_RF']=dict(mean_diff=float((z-x).mean()),wins=int((z>x).sum()),p=float(wilcoxon(z,x).pvalue))
json.dump([r],open('res_lazz_rawrf.json','w'),indent=1); print(r['ci95'],r['vs_PHYS+PA_RF'])
