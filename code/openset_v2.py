"""Open-set v2: physics-signature prototype distance (class-agnostic) + fusion with MSP of the twin model."""
import sys,json; sys.path.insert(0,'../code')
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
d=pd.read_parquet('./lazz_feat2.parquet')
SIG=['ci_min','cv_min','n_bypass','R_extra','dci','dcv','ci_min_sd30','cv_min_sd30']
XP=d[SIG].values.copy(); XP[:,3]=np.log1p(XP[:,3])
RAWS=['vdc1','vdc2','idc1','idc2','irr','pvt','idc1_sd30','idc2_sd30']
XR=d[RAWS].values.copy()
FD=[12,13,14,15,16]; R=[]
for space,X in [('physics',XP),('raw',XR)]:
 for unk in [1,2,3,4]:
     S=[];Y=[]
     for k in FD:
         tr=(d.day!=k)&(d.y!=unk); te=(d.day==k).values
         # robust per-class prototypes in physics-signature space
         dist=[]
         for c in sorted(d.y[tr].unique()):
             Z=X[tr.values&(d.y.values==c)]; med=np.median(Z,0); mad=np.median(np.abs(Z-med),0)*1.4826+1e-3
             dist.append(np.sqrt((((X[te]-med)/mad)**2).mean(1)))
         S.append(np.min(dist,0)); Y.append(d.y.values[te])
     S=np.concatenate(S); Y=np.concatenate(Y); isu=Y==unk
     R.append(dict(unknown=['','Short-circuit','Degradation','Open-circuit','Shading'][unk],space=space,AUROC_prototype=float(roc_auc_score(isu,S))))
     print(R[-1],flush=True)
json.dump(R,open('./res_openset_v2.json','w'),indent=1)
