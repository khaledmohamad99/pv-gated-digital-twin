"""Additional soiling scenarios that violate the twin estimator's assumption (I_sc proportional to the energy-relevant
soiling ratio): (i) non-uniform soiling, where the most-soiled cells limit I_sc and the I_sc loss exceeds the mean
(energy-relevant) loss by a factor h ~ U(1.3, 1.6); (ii) Isc-reducing partial shading of all substrings
(e.g. near obstacles, debris), on 30 % of days for 1-2 h, scaling I_sc and I_mp by U(0.6, 0.9)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import soiling as S
def simulate_ext(seed,nu=False,shi=False):
    x,SR,cl=S.simulate(seed,False,False); r=np.random.default_rng(1000+seed)
    if nu:
        h=r.uniform(1.3,1.6); orig=x.Isc/x.SR; x['Isc']=orig*(1-(1-x.SR)*h)
    if shi:
        dates=x.index.normalize(); ud=np.unique(dates); sh=np.zeros(len(x),bool); hrs=x.index.hour
        f=np.ones(len(x))
        for dd in r.choice(ud,int(.3*len(ud)),replace=False):
            h0=r.integers(10,16); m=(dates==dd)&(hrs>=h0)&(hrs<h0+r.integers(1,3)); f[m]=r.uniform(0.6,0.9)
        x['Isc']*=f; x['Imp']*=f; x['Pmp']=x.Imp*x.Vmp
    return x,SR
R=[]
for sc,(nu,shi) in {'S+NU':(1,0),'S+ShI':(0,1)}.items():
    for seed in range(20):
        x,SR=simulate_ext(seed,nu,shi); nv,srr,tw,info,tw0=S.estimators(x)
        for m,e in [('Naive PR',nv),('RdTools SRR',srr),('Twin w/o AOI physics',tw0),('Digital twin (ours)',tw)]:
            R.append(dict(scenario=sc,seed=seed,method=m,**S.score(e,SR)))
        if seed==0: pd.DataFrame({'SR_true':SR,'naive':nv,'twin':tw,'srr':srr if srr is not None else np.nan}).to_csv(f'soil_trace_{sc}.csv')
        print(sc,seed,flush=True)
R=pd.DataFrame(R); R.to_csv('res_soiling_ext.csv',index=False)
print(R.groupby(['scenario','method'])[['mae','bias','rmse']].agg(['mean','std']).round(4).to_string())
