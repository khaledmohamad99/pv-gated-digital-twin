"""Decision layer (common random numbers across policies; RUNS stochastic years): twin-driven cleaning scheduling vs fixed schedules and PR-driven rule.
Observation errors of each estimator are taken from the S+D+Sh benchmark (empirical residuals)."""
import numpy as np, pandas as pd, json, glob
# empirical daily estimation errors (estimate - truth) from the soiling benchmark traces
tr=pd.read_csv('soil_trace_S+D+Sh.csv',index_col=0,parse_dates=True)
err_twin=(tr.twin-tr.SR_true).dropna().values; err_naive=(tr.naive-tr.SR_true).dropna().values
# daily reference energy of a 1 MWp plant built from the measured JaSolar3 daily energy (Bilbao) scaled up
import sys; sys.path.insert(0,'../code')
from tec_params import load
x=load('JaSolar3').dropna(subset=['Pmp']).set_index('ts'); x=x[x.Pmp>=0]
E=(x.Pmp.resample('D').sum()*10/60/1000)            # kWh/day of the 315 W module
E=E[E>0]*(1e6/315)                                  # kWh/day for 1 MWp (Bilbao profile)
# rescale to a Gulf-class specific yield of 1900 kWh/kWp/yr, keeping the measured day-to-day variability
E=(E/E.mean()*1900*1000/365).values
def run(policy,seed,price,C,rate_rng=(0.001,0.01),horizon=365):
    r=np.random.default_rng(seed); ro=np.random.default_rng(100000+seed); n=len(E) if horizon is None else horizon  # common random numbers: r drives the plant, ro the observation noise
    sr=1.0; acc=0.0; cost=0.0; lost=0.0; ncl=0; rate=r.uniform(*rate_rng); bias=0.0
    for t in range(n):
        e=E[t%len(E)]
        lost+=(1-sr)*e*price
        # natural cleaning (rare rain / wind) ~ every 60 days, partial
        if r.random()<1/60: sr=sr+(1-sr)*r.uniform(0.3,0.9)
        # observation of soiling ratio by each estimator
        if policy=='oracle': obs=sr
        elif policy=='twin': obs=sr+ro.choice(err_twin)
        elif policy=='naive': obs=sr+ro.choice(err_naive)+bias
        acc_loss=(1-min(obs,1.0))*e*price if policy in('oracle','twin','naive') else 0
        clean=False
        if policy.startswith('every'):
            clean=(t+1)%int(policy[5:])==0
        else:
            acc+=max(acc_loss,0)
            clean=acc>=C           # renewal-optimal rule: clean when accumulated soiling loss = cleaning cost

        if clean: sr=1.0; acc=0.0; cost+=C; ncl+=1
        sr=max(sr-rate*r.uniform(0.5,1.5),0.6)
    return dict(policy=policy,seed=seed,price=price,C=C,lost=lost,cleaning_cost=cost,total=lost+cost,n_clean=ncl)
R=[]
# literature-grounded parameters: tariffs 0.0135 (Al Dhafra), 0.0242 (Noor Abu Dhabi) and 0.06 USD/kWh (non-auction markets);
# cleaning cost per MWp per cycle: 32 USD (machinery) and 190 USD (manual), UAE study; soiling 0.1-1 %/day (dry climates)
for price in [0.0135,0.0242,0.06]:
    for C in [32,190]:
        for seed in range(int(__import__("os").environ.get("RUNS","50"))):
            for pol in ['oracle','twin','naive','every7','every14','every30','every60']:
                R.append(run(pol,seed,price,C))
R=pd.DataFrame(R); R.to_csv('res_cleaning.csv',index=False)
S=R.groupby(['price','C','policy']).total.mean().unstack()
best_fixed=S[['every7','every14','every30','every60']].min(axis=1)
S['twin_vs_bestfixed_%']=100*(best_fixed-S.twin)/best_fixed
S['twin_gap_to_oracle_%']=100*(S.twin-S.oracle)/S.oracle
S['naive_vs_twin_%']=100*(S.naive-S.twin)/S.twin
print(S.round(0).to_string())
json.dump(S.reset_index().to_dict(orient='records'),open('res_cleaning_summary.json','w'),indent=1)
