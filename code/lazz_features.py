"""Online-calibrated twin + physics features for Lazzaretti (causal: only past data used)."""
import numpy as np, pandas as pd
import os
d=pd.read_parquet(os.environ.get('TWINF','lazz_twin.parquet'))
d=d[d.irr>50].copy()
d['minute']=d.t//60
# instantaneous physics ratios
for s in '12':
    d['ri'+s]=d['idc'+s]/d.tw_Imp
    d['rv'+s]=d['vdc'+s]/d.tw_Vmp
# --- online calibration (causal, gated EWMA at 1-min resolution) ---
m=d.groupby('minute').agg(irr=('irr','median'),irr_sd=('irr','std'),ri1=('ri1','median'),ri2=('ri2','median'),rv1=('rv1','median'),rv2=('rv2','median')).reset_index()
import os
CALMODE=os.environ.get('CALMODE','gated'); REANCHOR=int(os.environ.get('REANCHOR','1'))
BUF=int(os.environ.get('BUF','5')); MORNING=int(os.environ.get('MORNING','1'))
HALF=int(os.environ.get('HALF','240'))  # half-life in accepted minutes (~4 h of stable daylight)
EPS_I=float(os.environ.get('EPS_I','0.015')); EPS_V=float(os.environ.get('EPS_V','0.02')); DELTA_I=float(os.environ.get('DELTA_I','0.15'))
lam=0.5**(1/HALF)
keys=['ri1','ri2','rv1','rv2']
th={k:1.0 for k in keys}; boot=[]; out=[]; acc=[]; cm_buf=[]; n_reanchor=0
last_day_start=-10**9; prev_min=None
for r in m.itertuples(index=False):
    if prev_min is None or r.minute-prev_min>120: last_day_start=r.minute   # first daylight minute of a new day
    prev_min=r.minute
    out.append([th[k] for k in keys])        # value BEFORE using this minute => causal
    stable=(r.irr>300) and (r.irr_sd<0.03*r.irr)
    obs={k:getattr(r,k) for k in keys}
    ok_range=all(0.7<obs[k]<1.3 for k in keys)
    if not (stable and ok_range): acc.append(0); continue
    if CALMODE=='none': acc.append(0); th={k:1.0 for k in keys}; continue
    if CALMODE=='ungated' and len(boot)>=30:
        for k in keys: th[k]=lam*th[k]+(1-lam)*obs[k]
        acc.append(1); continue
    if len(boot)<30:                        # bootstrap on first 30 stable, in-range minutes
        boot.append(obs); th={k:float(np.median([b[k] for b in boot])) for k in keys}; acc.append(1); continue
    di=[obs['ri1']/th['ri1']-1, obs['ri2']/th['ri2']-1]; dv=[obs['rv1']/th['rv1']-1, obs['rv2']/th['rv2']-1]
    differential_ok=abs(di[0]-di[1])<EPS_I and abs(dv[0]-dv[1])<EPS_V
    # physics gate: soiling / cleaning / irradiance-sensor drift are COMMON-mode, OPTICAL (current-only) effects;
    # string faults are DIFFERENTIAL and/or voltage effects -> never absorbed by the calibration
    if differential_ok and abs(np.mean(di))<DELTA_I and max(abs(dv[0]),abs(dv[1]))<EPS_V:
        if REANCHOR and abs(np.mean(di))>0.02:
            cm_buf.append(obs)                      # persistent common-mode optical shift (e.g. cleaning event)
            if len(cm_buf)>=(2 if (MORNING and r.minute-last_day_start<90) else BUF):                     # change-point-triggered re-anchoring of the CURRENT states only
                for k in ['ri1','ri2']: th[k]=float(np.median([b[k] for b in cm_buf]))
                cm_buf=[]; n_reanchor+=1
        else:
            cm_buf=[]
            for k in keys: th[k]=lam*th[k]+(1-lam)*obs[k]
        acc.append(1)
    else: acc.append(0)
print('re-anchoring events',n_reanchor)
cal=pd.DataFrame(out,columns=['th_'+k for k in th]); cal['minute']=m.minute.values; cal['accepted']=acc
d=d.merge(cal,on='minute',how='left')
# calibrated residual ratios (1.0 == twin agreement)
for s in '12':
    d['ci'+s]=d['ri'+s]/d['th_ri'+s]; d['cv'+s]=d['rv'+s]/d['th_rv'+s]
    d['cp'+s]=d['ci'+s]*d['cv'+s]
# string-symmetric physics features
d['ci_min']=d[['ci1','ci2']].min(axis=1); d['ci_max']=d[['ci1','ci2']].max(axis=1)
d['cv_min']=d[['cv1','cv2']].min(axis=1); d['cv_max']=d[['cv1','cv2']].max(axis=1)
d['cp_min']=d[['cp1','cp2']].min(axis=1)
d['cp_tot']=(d.idc1*d.vdc1+d.idc2*d.vdc2)/(2*d.tw_Pmp*d.th_ri1*d.th_rv1)
d['n_bypass']=8*(1-d.cv_min)                     # equivalent number of lost modules (V-drop)
Ip=np.where(d.cv1<d.cv2,d.idc1,d.idc2); Vexp=np.where(d.cv1<d.cv2,d.th_rv1,d.th_rv2)*d.tw_Vmp
d['R_extra']=np.clip((Vexp-np.minimum(d.vdc1*0+np.where(d.cv1<d.cv2,d.vdc1,d.vdc2),Vexp))/np.maximum(Ip,0.2),0,500)  # apparent added series resistance (ohm)
d['dci']=d.ci1-d.ci2; d['dcv']=d.cv1-d.cv2
d['kt']=d.irr/d.groupby('day').irr.transform(lambda x:x.rolling(3600,min_periods=1).max())  # sky variability proxy
d['soil_idx']=d.th_ri1/d.groupby('day').th_ri1.transform('first')  # slow current drift
# causal window statistics (30 s)
W=30
for c in ['ci_min','cv_min','cp_min','dci','dcv','irr']:
    r=d[c].rolling(W,min_periods=1)
    d[c+'_sd30']=r.std().fillna(0); d[c+'_mean30']=r.mean()
d['dirr']=d.irr.diff().fillna(0)
d.to_parquet(os.environ.get('OUTF','lazz_feat.parquet'))
print(d.shape); print(d.groupby('y')[['ci_min','cv_min','cp_min','n_bypass','R_extra','dci','dcv']].median().round(3).to_string())
print(cal.drop(columns='minute').describe().round(3).to_string())
