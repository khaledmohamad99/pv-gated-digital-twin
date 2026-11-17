"""XAI (SHAP physics-consistency + stability), open-set (leave-one-fault-out), sensor-robustness, latency."""
import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb, shap
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import spearmanr
from common import *
d=pd.read_parquet('lazz_feat2.parquet')
rng=np.random.default_rng(0)
FSETS={'RAW':RAWW,'PHYS':PHYS}
def mk(): return lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0)
def sub(tr): return tr[(tr.y.isin([1,2,3]))|(rng.random(len(tr))<0.2)]
OUT={}
# physically expected measurement channel per class (first principles):
#  short-circuit / series degradation -> string VOLTAGE ; open-circuit / shading -> string CURRENT
CH={'RAW':{'V':[f for f in RAWW if f.startswith('vdc')],'I':[f for f in RAWW if f.startswith('idc')]},
    'PHYS':{'V':['cv_min','cv_max','n_bypass','R_extra','dcv','cv_min_sd30','cv_min_mean30','dcv_sd30','dcv_mean30'],
            'I':['ci_min','ci_max','dci','ci_min_sd30','ci_min_mean30','dci_sd30','dci_mean30']}}
EXP={1:'V',2:'V',3:'I',4:'I'}   # OC reported but excluded from the mean: breaker opening collapses BOTH V and I in this plant
fault_days=[12,13,14,15,16]
# ---------- 1) SHAP physics consistency + stability across leave-one-day-out models
for fs,cols in FSETS.items():
    imps=[]; pcs={c:[] for c in EXP}
    for k in fault_days:
        tr=sub(d[d.day!=k]); te=d[(d.day==k)]
        X=tr[cols].values; y=tr.y.values
        if fs=='PHYS': X=phys_augment(X,cols,rng); y=np.tile(y,3)
        m=mk().fit(X,y)
        # explain a stratified sample of test points predicted as each fault class
        p=m.predict(te[cols].values); 
        idx=np.concatenate([np.flatnonzero(p==c)[:400] for c in range(5)])
        sv=shap.TreeExplainer(m).shap_values(te[cols].values[idx])
        sv=np.stack(sv,axis=-1) if isinstance(sv,list) else sv   # (n,features,classes)
        g=np.abs(sv).mean(axis=(0,2)); imps.append(g/g.sum())
        for c,ch in EXP.items():
            rows=np.flatnonzero(p[idx]==c)
            if len(rows)==0: continue
            a=np.abs(sv[rows,:,c]).mean(0)
            iV=[cols.index(f) for f in CH[fs]['V']]; iI=[cols.index(f) for f in CH[fs]['I']]
            pcs[c].append(a[iV if ch=='V' else iI].sum()/(a[iV].sum()+a[iI].sum()+1e-12))
    imps=np.array(imps); rho=[spearmanr(imps[i],imps[j])[0] for i in range(len(imps)) for j in range(i+1,len(imps))]
    OUT[f'shap_{fs}']=dict(PCS={CLASSES[c]:float(np.mean(v)) for c,v in pcs.items()},PCS_mean=float(np.mean([np.mean(pcs[c]) for c in [1,2,4]])),
                          stability_rho_mean=float(np.mean(rho)),stability_rho_min=float(np.min(rho)),
                          global_importance=dict(zip(cols,imps.mean(0).round(4).tolist())))
    print(fs,'PCS',OUT[f'shap_{fs}']['PCS'],'stab',round(np.mean(rho),3),flush=True)
# ---------- 2) open-set: leave-one-fault-type-out (unseen fault at test time)
#  score_known = max softmax prob ; physics novelty = classifier says normal/known but twin residual says abnormal
def twin_anomaly(df):
    # class-agnostic physics residual magnitude (healthy twin agreement = 0)
    return np.maximum.reduce([np.abs(1-df.ci_min),np.abs(1-df.cv_min),np.abs(df.dci),np.abs(df.dcv)])
OS=[]
for unk in [1,2,3,4]:
    for fs,cols in FSETS.items():
        P=[];Y=[];A=[]
        for k in fault_days:
            tr=sub(d[(d.day!=k)&(d.y!=unk)]); te=d[d.day==k]
            X=tr[cols].values; y=tr.y.values
            if fs=='PHYS': X=phys_augment(X,cols,rng); y=np.tile(y,3)
            m=mk().fit(X,y)
            pr=m.predict_proba(te[cols].values); P.append(pr); Y.append(te.y.values); A.append(np.asarray(twin_anomaly(te)))
        P=np.vstack(P);Y=np.concatenate(Y);A=np.concatenate(A)
        msp=P.max(1); isu=(Y==unk)
        auc_msp=roc_auc_score(isu,-msp)
        # physics-guided open-set score: twin residual high while classifier is not confident in a known FAULT
        pk=P.copy(); 
        known_fault_conf=pk[:,1:].max(1)
        score=A*(1-known_fault_conf)
        auc_phys=roc_auc_score(isu,score)
        # operating point: flag 'unknown' if twin anomaly >0.08 and classifier max prob <0.9 or predicts normal
        pred_lab=P.argmax(1)
        flag=(A>0.08)&((pred_lab==0)|(msp<0.9))
        OS.append(dict(unknown=CLASSES[unk],features=fs,AUROC_MSP=auc_msp,AUROC_physics=auc_phys,
                       unk_recall=float(flag[isu].mean()),false_unknown_on_normal=float(flag[Y==0].mean()),
                       false_unknown_on_known_faults=float(flag[(Y>0)&~isu].mean())))
        print(OS[-1],flush=True)
OUT['open_set']=OS
json.dump(OUT,open('res_lazz_xai_open.json','w'),indent=1)
# ---------- 3) latency (single-sample, end-to-end feature+inference) on PHYS LightGBM
m=mk().fit(sub(d)[PHYS].values,sub(d).y.values); x1=d[PHYS].values[:1]
t=time.perf_counter(); 
for _ in range(2000): m.predict(x1)
lat=(time.perf_counter()-t)/2000*1e3
xb=d[PHYS].values[:100000]; t=time.perf_counter(); m.predict(xb); thr=100000/(time.perf_counter()-t)
OUT['latency']=dict(single_ms=lat,throughput_samples_per_s=thr)
json.dump(OUT,open('res_lazz_xai_open.json','w'),indent=1); print(OUT['latency'])
