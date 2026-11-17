RAW=['vdc1','vdc2','idc1','idc2','irr','pvt']
RAWW=RAW+['irr_sd30','dirr']
import pandas as pd
def add_raw_window(d):
    for c in ['vdc1','vdc2','idc1','idc2']:
        r=d[c].rolling(30,min_periods=1); d[c+'_sd30']=r.std().fillna(0); d[c+'_m30']=r.mean()
    return d
RAWW=RAWW+[c+s for c in ['vdc1','vdc2','idc1','idc2'] for s in ['_sd30','_m30']]
PHYS=['ci_min','ci_max','cv_min','cv_max','cp_min','cp_tot','n_bypass','R_extra','dci','dcv',
      'ci_min_sd30','ci_min_mean30','cv_min_sd30','cv_min_mean30','cp_min_sd30','cp_min_mean30',
      'dci_sd30','dci_mean30','dcv_sd30','dcv_mean30','irr_sd30','kt','dirr']
CLASSES=['Normal','Short-circuit','Degradation','Open-circuit','Shading']
import numpy as np
CI_SCALE=['ci_min','ci_max','ci_min_mean30','cp_min','cp_tot','cp_min_mean30','ci_min_sd30','cp_min_sd30','dci','dci_mean30','dci_sd30']
CV_SCALE=['cv_min','cv_max','cv_min_mean30','cv_min_sd30']
def phys_augment(X,cols,rng,gi=(0.93,1.07),gv=(0.98,1.02),copies=2):
    """Physics-consistent augmentation: common-mode optical gain (soiling / irradiance-sensor error) scales all
    current-ratio features; common-mode voltage gain (temperature-sensor error) scales voltage-ratio features."""
    ii=[cols.index(c) for c in CI_SCALE if c in cols]; iv=[cols.index(c) for c in CV_SCALE if c in cols]
    ib=cols.index('n_bypass') if 'n_bypass' in cols else None
    out=[X]
    for _ in range(copies):
        Z=X.copy(); g=rng.uniform(*gi,size=(len(X),1)); h=rng.uniform(*gv,size=(len(X),1))
        Z[:,ii]*=g; Z[:,iv]*=h
        if ib is not None: Z[:,ib]=8*(1-Z[:,cols.index('cv_min')])
        out.append(Z)
    return np.vstack(out)
