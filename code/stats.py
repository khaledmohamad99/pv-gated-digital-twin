import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
d=pd.read_parquet('./lazz_feat2.parquet'); y=d.y.values; day=d.day.values
P=np.load('./pred_lazz.npz'); out={}
FD=list(range(9,17)); rng=np.random.default_rng(0)
def mf1(mask,p): return f1_score(y[mask],p[mask],average='macro',labels=sorted(set(y[mask])))
for k in P.files:
    p=P[k]; per=[mf1(day==dd,p) for dd in FD]
    # block bootstrap over days (all 16 days resampled)
    boots=[]
    days=np.unique(day)
    for b in range(300):
        s=rng.choice(days,len(days)); m=np.concatenate([np.flatnonzero(day==dd) for dd in s])
        boots.append(f1_score(y[m],p[m],average='macro'))
    out[k]=dict(per_fault_day=per,ci95=[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))])
pairs=[('LODO_RAW_MLP','LODO_HYBRID_MLP'),('LODO_RAW_MLP','LODO_PHYS+PA_MLP'),('LODO_RAW_XGBoost','LODO_PHYS+PA_XGBoost'),('LODO_RAW_LightGBM','LODO_PHYS+PA_LightGBM'),('LODO_RAW_LightGBM','LODO_HYBRID_LightGBM'),('LODO_RAW_XGBoost','LODO_HYBRID_XGBoost')]
T={}
for a,b in pairs:
    if a in out and b in out:
        x=np.array(out[a]['per_fault_day']); z=np.array(out[b]['per_fault_day'])
        T[f'{b} vs {a}']=dict(mean_diff=float((z-x).mean()),wins=int((z>x).sum()),p_wilcoxon=float(wilcoxon(z,x).pvalue))
out['tests']=T; json.dump(out,open('./res_stats_v2.json','w'),indent=1)
print(json.dumps(T,indent=1)); print({k:v['ci95'] for k,v in out.items() if k!='tests'})
