"""Semi-synthetic soiling benchmark on 9 months of real TECNALIA reference-module data.
Soiling, Rs-degradation and shading are injected with physically consistent effects (SDM), then
three estimators recover the soiling ratio: naive PR, NREL RdTools SRR, and the digital twin."""
import sys; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, pvlib, json, warnings; warnings.filterwarnings('ignore')
from rdtools import soiling as rds
from tec_params import load
k=1.380649e-23;q=1.602176634e-19
d=load('JaSolar3').dropna(subset=['G','T']).set_index('ts').sort_index()
d=d[~d.index.duplicated()]
# --- QC: drop days whose irradiance is inconsistent with solar geometry (timestamp glitches)
loc=pvlib.location.Location(43.2962,-2.8706,tz='Europe/Madrid')
idx=d.index.tz_localize('Europe/Madrid',ambiguous='NaT',nonexistent='NaT')
d=d[~idx.isna()]; idx=idx[~idx.isna()]
sp=loc.get_solarposition(idx)
d['elev']=sp.apparent_elevation.values
d['aoi']=pvlib.irradiance.aoi(36,180,sp.apparent_zenith.values,sp.azimuth.values)
d['iam']=pvlib.iam.physical(d.aoi.values).clip(0.05,1)
bad=(d.G>50)&(d.elev<0); dd=d.index.normalize()
badday=bad.groupby(dd).mean()>0.02
d=d[~badday.reindex(dd).values]
d=d[d['T'].between(-20,90)]
QC=dict(days_removed=int(badday.sum()),days_kept=int((~badday).sum()))
# JaSolar3 SDM state identified by the twin (median of daily estimates)
P=pd.read_csv('tec_params.csv'); P=P[(P.module=='JaSolar3')&(P.cost<1e-6)]
RS0,RSH0,I00=P.Rs.median(),P.Rsh.median(),P.I0.median()
def mpp_ratio(G,T,dRs):
    a=1.2*60*k*(T+273.15)/q; IL=9.98*G/1000*(1+0.0006*(T-25))
    o0=pvlib.pvsystem.singlediode(IL,I00,RS0,RSH0,a); o1=pvlib.pvsystem.singlediode(IL,I00,RS0+dRs,RSH0,a)
    return np.asarray(o1['i_mp']/o0['i_mp']),np.asarray(o1['v_mp']/o0['v_mp']),np.asarray(o1['i_sc']/o0['i_sc'])
day=(d.G>400)
dates=d.index.normalize(); udays=np.unique(dates)
def simulate(seed,deg=False,shade=False):
    r=np.random.default_rng(seed); x=d.copy()
    # soiling: rate 0.1-1.0 %/day, cleanings every 15-60 days (70-100 % effectiveness)
    sr=np.ones(len(udays)); s=1.0; nxt=14+r.integers(15,60); rate=r.uniform(0.001,0.01); cleans=[]
    for i in range(len(udays)):
        if i==nxt: s=s+(1-s)*r.uniform(0.7,1.0); nxt=i+r.integers(15,60); rate=r.uniform(0.001,0.01); cleans.append(i)
        sr[i]=s
        if udays[i]>=udays[0]+np.timedelta64(14,'D'): s=max(s-rate*r.uniform(0.5,1.5),0.6)   # commissioning: clean for 14 days
    SR=pd.Series(sr,index=udays); x['SR']=SR.reindex(dates).values
    x['Imp']*=x.SR; x['Isc']*=x.SR; x['Pmp']=x.Imp*x.Vmp
    if deg:   # series-resistance growth (corrosion), 0 -> 0.8 ohm over the period
        dRs=0.8*np.linspace(0,1,len(x)); ri,rv,rsc=mpp_ratio(np.clip(x.G.values,1,None),x['T'].values,dRs)
        x['Imp']*=ri; x['Vmp']*=rv; x['Isc']*=rsc; x['Pmp']=x.Imp*x.Vmp
    if shade: # transient partial shading on 30 % of days, 1-2 h, one substring bypassed
        sh=np.zeros(len(x),bool); hrs=x.index.hour
        for dd in r.choice(udays,int(.3*len(udays)),replace=False):
            h0=r.integers(13,16); m=(dates==dd)&(hrs>=h0)&(hrs<h0+r.integers(1,3)); sh|=m
        x.loc[sh,'Vmp']*=2/3; x.loc[sh,'Imp']*=r.uniform(0.85,0.95); x['Pmp']=x.Imp*x.Vmp
    return x,SR,np.array(cleans)
def estimators(x):
    t0=x.index[0]+pd.Timedelta(days=14); cal=x[(x.index<t0)&day]
    # --- naive: temperature-corrected PR relative to first 2 weeks
    pr=x.Pmp/(x.G/1000*(1-0.0038*(x['T']-25)))
    prd=pr[day].groupby(dates[day]).median(); naive=prd/prd[prd.index<t0].median()
    # --- RdTools SRR on insolation-weighted daily PR
    try:
        expP=x.G/1000*(1-0.0038*(x['T']-25))*cal.Pmp.sum()/ (cal.G/1000*(1-0.0038*(cal['T']-25))).sum()
        pi=(x.Pmp/expP).where(day); ins=x.G.where(day)
        pi_d=(pi*ins).resample('D').sum()/ins.resample('D').sum(); ins_d=ins.resample('D').sum()
        pi_d=pi_d.dropna(); ins_d=ins_d.reindex(pi_d.index)
        pi_d=pi_d.asfreq('D'); ins_d=ins_d.asfreq('D'); pi_d.index=pi_d.index.tz_localize('Europe/Madrid'); ins_d.index=pi_d.index
        sa=rds.SRRAnalysis(pi_d,ins_d); res=sa.run(reps=200)
        srr=res[2]['soiling_ratio_perfect_clean']; srr.index=srr.index.tz_localize(None)
        srr_info=dict(srr_median=float(res[0]))
        srr=srr.reindex(naive.index) if srr is not None else None
    except Exception as e:
        srr=None; srr_info=dict(err=str(e)[:200])
    # --- digital twin: physics-structured expected Isc = k*G*(1+alpha(T-25)), calibrated on 2 clean weeks
    alpha=0.0006
    out={}
    for tag,use_iam in [('noIAM',False),('IAM',True)]:
        geo=x.iam if use_iam else 1.0
        expI=x.G/1000*(1+alpha*(x['T']-25))*geo
        kk=(cal.Isc/expI.loc[cal.index]).median()
        ci=x.Isc/(kk*expI)
        valid=day&ci.notna()&(x.FF.between(40,90))&(x.aoi<70)
        cid=ci[valid].groupby(dates[valid]).median()
        out[tag]=cid.rolling(3,min_periods=1).median()
    twin=out['IAM']; twin_noiam=out['noIAM']
    return naive,srr,twin,srr_info,twin_noiam
def score(est,SR):
    if est is None: return dict(mae=np.nan,bias=np.nan)
    e=(est-SR.reindex(est.index)).dropna(); e=e[e.index>=e.index[0]+pd.Timedelta(days=14)]
    return dict(mae=float(e.abs().mean()),bias=float(e.mean()),rmse=float(np.sqrt((e**2).mean())))
if __name__=='__main__':
    R=[]
    for sc,(dg,sh) in {'S':(0,0),'S+D':(1,0),'S+Sh':(0,1),'S+D+Sh':(1,1)}.items():
        for seed in range(20):
            x,SR,cl=simulate(seed,dg,sh); nv,srr,tw,info,tw0=estimators(x)
            for m,e in [('Naive PR',nv),('RdTools SRR',srr),('Twin w/o AOI physics',tw0),('Digital twin (ours)',tw)]:
                R.append(dict(scenario=sc,seed=seed,method=m,**score(e,SR)))
            if seed==0: pd.DataFrame({'SR_true':SR,'naive':nv,'twin':tw,'srr':srr if srr is not None else np.nan,'twin_noIAM':tw0}).to_csv(f'soil_trace_{sc}.csv')
        print(sc,flush=True)
    print(QC); json.dump(QC,open('tec_qc.json','w'))
    R=pd.DataFrame(R); R.to_csv('res_soiling.csv',index=False)
    print(R.groupby(['scenario','method'])[['mae','bias','rmse']].agg(['mean','std']).round(4).to_string())
