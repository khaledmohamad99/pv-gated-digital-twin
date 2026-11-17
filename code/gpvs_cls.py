import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier
F=pd.read_pickle('gpvs_feat.pkl').reset_index(drop=True)
SIG=['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf']
STAT=[s+'_'+k for s in SIG for k in ['mean','std','min','max']]
PB=['P_pv','P_ac','Q_ac','eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','dVdI','dPdV','Vdc_ripple','dc_balance','freq_dev','Vpv_mean','Ipv_mean','Vdc_mean','Iabc_mean']
INV=['eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','Vdc_ripple','dc_balance','freq_dev']
PHYS=INV+['z_'+c for c in PB]+['r_'+c for c in PB]
FS={'STAT':STAT,'PHYS':PHYS,'HYBRID':STAT+PHYS}
F=F.replace([np.inf,-np.inf],np.nan).fillna(0)
for c in PHYS: F[c]=F[c].clip(-1e4,1e4)
def models():
    return {'LightGBM':lambda: lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=2,random_state=0),
     'XGBoost':lambda: XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,n_jobs=2,random_state=0),
     'RandomForest':lambda: RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=2,random_state=0),
     'MLP':lambda: make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=400,early_stopping=True,random_state=0))}
K=8
def blocked_split(F,frac=0.6,gap=3):
    tr=[];te=[]
    for (f,y),g in F.groupby(['file','y']):
        ix=g.sort_values('t_end').index.values; n=int(len(ix)*frac)
        tr+=list(ix[:n]); te+=list(ix[n+gap:])
    return np.array(tr),np.array(te)
def evaluate(tr,te,fs,mn):
    X=F[FS[fs]].values.astype(np.float32); y=F.y.values
    m=models()[mn]()
    if mn=='XGBoost':
        c=np.bincount(y[tr],minlength=K); w=(len(tr)/(K*np.maximum(c,1)))[y[tr]]; m.fit(X[tr],y[tr],sample_weight=w)
    else: m.fit(X[tr],y[tr])
    p=m.predict(X[te]); return p,m
res=[]; preds={}
for proto in ['BLOCKED','M->L','L->M']:
    if proto=='BLOCKED': tr,te=blocked_split(F)
    elif proto=='M->L': tr=np.flatnonzero(F['mode']=='M'); te=np.flatnonzero(F['mode']=='L')
    else: tr=np.flatnonzero(F['mode']=='L'); te=np.flatnonzero(F['mode']=='M')
    for fs in FS:
        for mn in models():
            t=time.time(); p,_=evaluate(tr,te,fs,mn); yt=F.y.values[te]
            r=dict(protocol=proto,features=fs,model=mn,acc=accuracy_score(yt,p),bacc=balanced_accuracy_score(yt,p),macroF1=f1_score(yt,p,average='macro'),
                   perclassF1=f1_score(yt,p,average=None,labels=range(K)).tolist(),det_acc=accuracy_score(yt>0,p>0),cm=confusion_matrix(yt,p,labels=range(K)).tolist(),sec=time.time()-t)
            res.append(r); preds[f'{proto}|{fs}|{mn}']=(te,p)
            print(proto,fs,mn,round(r['macroF1'],4),round(r['acc'],4),np.round(r['perclassF1'],2),flush=True)
json.dump(res,open('res_gpvs_cls.json','w'),indent=1)
pd.to_pickle(preds,'pred_gpvs.pkl')
