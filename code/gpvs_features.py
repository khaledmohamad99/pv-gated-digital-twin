"""Window features for GPVS-Faults (10 kHz). Window 0.1 s (5 grid cycles), hop 0.05 s."""
import numpy as np, pandas as pd, glob
FS=1e4; W=1000; H=500; CAL=2.0   # twin calibration on first 2 s (healthy) of every run
SIG=['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf']
def phasor(x):
    n=len(x); t=np.arange(n)/FS; e=np.exp(-2j*np.pi*50*t); return 2*np.mean(x*e)
def thd(x):
    X=np.abs(np.fft.rfft(x*np.hanning(len(x)))); f=np.fft.rfftfreq(len(x),1/FS)
    k=np.argmin(abs(f-50)); h=[np.argmin(abs(f-50*m)) for m in range(2,20)]
    return np.sqrt(np.sum(X[h]**2))/max(X[k],1e-9)
a=np.exp(2j*np.pi/3)
def win_feats(w):
    f={}
    for s in SIG:
        x=w[s].values; f[s+'_mean']=x.mean(); f[s+'_std']=x.std(); f[s+'_min']=x.min(); f[s+'_max']=x.max()
    I=w.Ipv.values; V=w.Vpv.values; ppv=I*V
    pac=(w.va*w.ia+w.vb*w.ib+w.vc*w.ic).values
    Ia,Ib,Ic=[phasor(w[c].values) for c in ['ia','ib','ic']]; Va,Vb,Vc=[phasor(w[c].values) for c in ['va','vb','vc']]
    I1=(Ia+a*Ib+a*a*Ic)/3; I2=(Ia+a*a*Ib+a*Ic)/3; V1=(Va+a*Vb+a*a*Vc)/3; V2=(Va+a*a*Vb+a*Vc)/3
    S=1.5*V1*np.conj(I1)
    f['P_pv']=ppv.mean(); f['P_ac']=pac.mean(); f['Q_ac']=S.imag
    f['eta']=pac.mean()/max(abs(ppv.mean()),1e-3)
    f['boost_ratio']=w.Vdc.mean()/max(abs(V.mean()),1e-3)
    f['I_unb']=abs(I2)/max(abs(I1),1e-6); f['V_unb']=abs(V2)/max(abs(V1),1e-6)
    f['I_thd']=np.mean([thd(w[c].values) for c in ['ia','ib','ic']]); f['V_thd']=np.mean([thd(w[c].values) for c in ['va','vb','vc']])
    f['V_zero']=np.abs((w.va+w.vb+w.vc).values).mean()/max(abs(V1),1e-6)       # sensor consistency (KVL)
    f['I_zero']=np.abs((w.ia+w.ib+w.ic).values).mean()/max(abs(I1),1e-6)       # KCL consistency
    f['Vabc_consist']=w.Vabc.mean()/max(abs(V1),1e-6)
    f['Iabc_consist']=w.Iabc.mean()/max(abs(I1),1e-6)
    dI=np.diff(I); dV=np.diff(V); 
    f['dVdI']=np.cov(dV,dI)[0,1]/max(np.var(dI),1e-9)                           # dynamic PV source impedance
    f['dPdV']=np.cov(np.diff(ppv),dV)[0,1]/max(np.var(dV),1e-9)                 # MPPT optimality (0 at MPP)
    f['Vdc_ripple']=w.Vdc.std()/max(w.Vdc.mean(),1e-3)
    f['dc_balance']=(ppv.mean()-pac.mean())/max(abs(ppv.mean()),1e-3)
    f['freq_dev']=abs(w.If.mean()-50)+abs(w.Vf.mean()-50)
    return f
PHYS_BASE=['P_pv','P_ac','Q_ac','eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','dVdI','dPdV','Vdc_ripple','dc_balance','freq_dev','Vpv_mean','Ipv_mean','Vdc_mean','Iabc_mean']
rows=[]
for p in sorted(glob.glob('gpvs_F*.pkl')):
    n=p[5:8]; d=pd.read_pickle(p); T=d.Time.values[-1]; onset=T/2
    starts=range(0,len(d)-W,H)
    F=[]
    for s in starts:
        w=d.iloc[s:s+W]; f=win_feats(w); f['t_end']=d.Time.values[s+W-1]; F.append(f)
    F=pd.DataFrame(F); F['file']=n; F['mode']=n[2]; F['fault']=int(n[1])
    F['y']=np.where((F.fault>0)&(F.t_end>=onset),F.fault,0); F['onset']=onset if n[1]!='0' else np.nan
    # twin calibration: relative deviation from the healthy start-up (first CAL s) of each run
    base=F[F.t_end<=CAL][PHYS_BASE]
    for c in PHYS_BASE:
        mu=base[c].mean(); sd=base[c].std()+1e-6
        F['z_'+c]=(F[c]-mu)/sd             # standardised twin residual
        F['r_'+c]=F[c]/(abs(mu)+1e-6)      # ratio to healthy twin state
    rows.append(F); print(n,len(F),flush=True)
pd.concat(rows).to_pickle('gpvs_feat.pkl')
