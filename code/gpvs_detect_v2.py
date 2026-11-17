"""GPVS unsupervised real-time detection v2: change-point ground-truth onsets, twin Mahalanobis + CUSUM, fusion."""
import numpy as np, pandas as pd, json
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
F=pd.read_pickle('gpvs_feat.pkl').reset_index(drop=True).replace([np.inf,-np.inf],np.nan).fillna(0)
SIG=['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf']
STAT=[s+'_'+k for s in SIG for k in ['mean','std','min','max']]
PB=['P_pv','P_ac','Q_ac','eta','boost_ratio','I_unb','V_unb','I_thd','V_thd','V_zero','I_zero','Vabc_consist','Iabc_consist','dVdI','dPdV','Vdc_ripple','dc_balance','freq_dev','Vpv_mean','Ipv_mean','Vdc_mean','Iabc_mean']
CAL=2.0; CONSEC=3
# ---- 1) ground-truth onset by offline change-point (binary segmentation, mean shift on standardised STAT+PHYS)
def onset_cp(g):
    X=StandardScaler().fit_transform(g[STAT+PB].values); t=g.t_end.values; T=t[-1]
    best=(np.inf,None); cs=np.cumsum(X,0); cs2=np.cumsum(X**2,0); n=len(X)
    for i in range(int(.25*n),int(.95*n)):
        a=cs2[i-1]-cs[i-1]**2/i; b=(cs2[-1]-cs2[i-1])-(cs[-1]-cs[i-1])**2/(n-i); c=(a+b).sum()
        if c<best[0]: best=(c,t[i])
    return best[1]
ON={}
for f,g in F.groupby('file'):
    g=g.sort_values('t_end')
    if f[1]!='0': ON[f]=dict(nominal=float(g.t_end.values[-1]/2),changepoint=float(onset_cp(g)))
print(pd.DataFrame(ON).T.round(2).to_string())
def alarm_stream(score,thr):
    a=score>thr; run=0; out=np.zeros(len(a),bool)
    for i,v in enumerate(a): run=run+1 if v else 0; out[i]=run>=CONSEC
    return out
def cusum(D,k):
    S=np.zeros(len(D)); 
    for i in range(1,len(D)): S[i]=max(0,S[i-1]+D[i]-k)
    return S
def scores(g,method):
    cal=g[g.t_end<=CAL]
    if method in('PCA-T2/SPE','Fused (twin+PCA)','Fused + CUSUM (ours)'):
        ss=StandardScaler().fit(cal[STAT]); Xc=ss.transform(cal[STAT]); X=ss.transform(g[STAT])
        p=PCA(n_components=10).fit(Xc)
        def st(Z): T=p.transform(Z); R=Z-p.inverse_transform(T); return (T**2/p.explained_variance_).sum(1),(R**2).sum(1)
        t2c,spc=st(Xc); t2,sp=st(X); pca=np.maximum(t2/np.percentile(t2c,99),sp/np.percentile(spc,99)); pcac=np.maximum(t2c/np.percentile(t2c,99),spc/np.percentile(spc,99))
        if method=='PCA-T2/SPE': return pca,pcac
    Z=['z_'+c for c in PB]
    lw=LedoitWolf().fit(cal[Z].values); D=lw.mahalanobis(g[Z].values); Dc=lw.mahalanobis(cal[Z].values)
    D=D/np.percentile(Dc,99); Dc=Dc/np.percentile(Dc,99)
    if method=='Twin Mahalanobis': return D,Dc
    if method=='Twin Mahalanobis + CUSUM (ours)': return cusum(D,2.0),cusum(Dc,2.0)
    if method=='Fused (twin+PCA)': return np.maximum(D,pca),np.maximum(Dc,pcac)
    if method=='Fused + CUSUM (ours)': s=np.maximum(D,pca); c=np.maximum(Dc,pcac); return cusum(s,2.0),cusum(c,2.0)
res=[]
for method in ['PCA-T2/SPE','Twin Mahalanobis','Twin Mahalanobis + CUSUM (ours)','Fused (twin+PCA)','Fused + CUSUM (ours)']:
    S={f:(g.sort_values('t_end'),)+scores(g.sort_values('t_end'),method) for f,g in F.groupby('file')}
    def far(h):
        return np.mean([alarm_stream(sc,h)[g.t_end.values>CAL].mean() for f,(g,sc,cs) in S.items() if f[1]=='0'])
    ks=np.unique(np.r_[np.linspace(0.5,5,46),np.geomspace(5,500,80)]); kk=next((k for k in ks if far(k)<=0.01),ks[-1])
    for f,(g,sc,cs) in S.items():
        thr=kk; al=alarm_stream(sc,thr); t=g.t_end.values
        if f[1]=='0': res.append(dict(method=method,file=f,FAR=float(al[t>CAL].mean()))); continue
        for ref in ['nominal','changepoint']:
            on=ON[f][ref]; pre=(t>CAL)&(t<on); post=t>=on; hit=np.flatnonzero(al&post)
            res.append(dict(method=method,file=f,onset=ref,FAR=float(al[pre].mean()) if pre.any() else np.nan,detected=bool(len(hit)),
                            delay_ms=float((t[hit[0]]-on)*1000) if len(hit) else np.nan,TPR=float(al[post].mean())))
    print(method,'k',round(kk,2),flush=True)
R=pd.DataFrame(res); R.to_csv('res_gpvs_detect_v2.csv',index=False); json.dump(ON,open('gpvs_onsets.json','w'),indent=1)
q=R[R.onset=='changepoint']
print(q.pivot_table(index='file',columns='method',values='delay_ms').round(0).to_string())
print(q.groupby('method').agg(det=('detected','mean'),delay_med=('delay_ms','median'),TPR=('TPR','mean'),FAR_pre=('FAR','mean')).round(3))
print(R[R.file.str[1]=='0'].groupby('method').FAR.mean())
