"""Cross-mode transfer on GPVS: physics-normalised features + unsupervised domain adaptation (CORAL, transductive)."""
import numpy as np, pandas as pd, json, lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy import linalg
F=pd.read_pickle('gpvs_feat.pkl').reset_index(drop=True).replace([np.inf,-np.inf],np.nan).fillna(0)
SIG=['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf']
STAT=[s+'_'+k for s in SIG for k in ['mean','std','min','max']]
PB=['P_pv','P_ac','Q_ac','eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','dVdI','dPdV','Vdc_ripple','dc_balance','freq_dev','Vpv_mean','Ipv_mean','Vdc_mean','Iabc_mean']
INV=['eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','Vdc_ripple','dc_balance','freq_dev']
ZR=['z_'+c for c in PB]+['r_'+c for c in PB]
for c in ZR+INV: F[c]=F[c].clip(-1e3,1e3)
SETS={'STAT':STAT,'PHYS':INV+ZR,'PHYS-z':ZR}
def coral(Xs,Xt,eps=1.0):
    ms,mt=Xs.mean(0),Xt.mean(0); ss,st=Xs.std(0)+1e-9,Xt.std(0)+1e-9
    A=(Xs-ms)/ss; B=(Xt-mt)/st
    Cs=np.cov(A,rowvar=False)+eps*np.eye(A.shape[1]); Ct=np.cov(B,rowvar=False)+eps*np.eye(A.shape[1])
    W=np.real(linalg.fractional_matrix_power(Cs,-0.5)@linalg.fractional_matrix_power(Ct,0.5))
    return A@W, B
def per_run_std(X,files):
    # unsupervised per-run normalisation using the healthy start-up (first 2 s) of each run
    Z=X.copy()
    for f in np.unique(files):
        m=files==f; cal=m&(F.t_end.values<=2.0); mu=X[cal].mean(0); sd=X[cal].std(0)+1e-6; Z[m]=(X[m]-mu)/sd
    return Z
mk={'LightGBM':lambda: lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0),
    'RandomForest':lambda: RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=1,random_state=0)}
R=[]
for src,tgt in [('M','L'),('L','M')]:
    s=(F['mode']==src).values; t=(F['mode']==tgt).values; ys=F.y.values[s]; yt=F.y.values[t]
    for fs,cols in SETS.items():
        X=F[cols].values.astype(float)
        for da in ['none','run-norm','CORAL','run-norm+CORAL']:
            Xs,Xt=X[s],X[t]
            if 'run-norm' in da: Z=per_run_std(X,F.file.values); Xs,Xt=Z[s],Z[t]
            if 'CORAL' in da: Xs,Xt=coral(Xs,Xt)
            for mn in mk:
                p=mk[mn]().fit(Xs,ys).predict(Xt)
                R.append(dict(direction=f'{src}->{tgt}',features=fs,DA=da,model=mn,macroF1=f1_score(yt,p,average='macro'),det_F1=f1_score(yt>0,p>0)))
                print(R[-1],flush=True)
R=pd.DataFrame(R); R.to_csv('res_gpvs_da.csv',index=False)
print(R.groupby(['features','DA','model'])[['macroF1','det_F1']].mean().round(3).sort_values('macroF1').to_string())
