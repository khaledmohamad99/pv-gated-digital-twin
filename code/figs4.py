import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('../code/figs1.py').read().split("OUT='./figs/'")[0])
OUT='./figs/'; W='./'
AB=pd.read_csv(W+'res_ablation_ms.csv'); g=AB.groupby('config',sort=False).macroF1.agg(['mean','std']).reset_index()
AR=pd.read_csv(W+'res_ablation_robust.csv'); RB=pd.DataFrame(json.load(open(W+'res_lazz_robust.json')))
fig,ax=plt.subplots(1,2,figsize=(7.4,3.0),gridspec_kw=dict(width_ratios=[1.35,1]))
lab=[c.split(' ',1)[0] for c in g.config]; cols=[C[0] if c.startswith('A') else C[3] for c in g.config]
ax[0].bar(range(len(g)),g['mean'],yerr=g['std'].fillna(0),color=cols,capsize=3,width=.65)
for i,(v,sd) in enumerate(zip(g['mean'],g['std'].fillna(0))): ax[0].text(i,v+sd+.002,f'{v:.3f}',ha='center',fontsize=6.5,color=INK,va='bottom')
ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels(['A0\nno cal.','A1\n+cal.','A2\n+gate','A3\n+re-anch.','A4\n+fast','A5 final\n(A3+aug.)','RAW'][:len(g)],fontsize=7)
ax[0].set_ylim(.9,1.0); ax[0].set_ylabel('Macro-F1 (LODO, mean ± sd)'); ax[0].set_title('(a) Ablation in nominal conditions',fontsize=9,loc='left')
conds=['g090','clean','g110']; xs=np.arange(3)
series=[('Twin without calibration (A0)',AR[AR.config=='A0 no calibration'].set_index('condition').macroF1,C[3]),
        ('Raw LightGBM',RB[(RB.features=='RAW')&(RB.model=='LightGBM')].set_index('condition').macroF1,'#9b9b9b'),
        ('Full twin (ours)',AR[AR.config.str.startswith('A5')].set_index('condition').macroF1,C[0])]
for j,(l,s,c) in enumerate(series):
    v=[s[k] for k in conds]; ax[1].plot(xs,v,marker='o',ms=7,lw=2,color=c,label=l,markeredgecolor='white',markeredgewidth=1.5)
ax[1].set_xticks(xs); ax[1].set_xticklabels(['−10 %','0','+10 %']); ax[1].set_xlabel('Irradiance-sensor gain error'); ax[1].set_ylabel('Macro-F1 (fault days)')
ax[1].set_ylim(.5,1.0); ax[1].legend(frameon=False,fontsize=7,loc='lower center'); ax[1].set_title('(b) Robustness: role of calibration',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_ablation.png',dpi=300); plt.close()
# transfer figure incl. CNN
DL=pd.DataFrame(json.load(open(W+'res_dl.json'))); q=DL[DL.dataset=='GPVS']
cm=lambda f,m: q[(q.features==f)&(q.model==m)&(q.protocol.isin(['M->L','L->M']))].macroF1.mean()
D=pd.read_csv(W+'res_gpvs_da.csv'); gg=D.groupby(['features','DA','model']).macroF1.mean()
best=lambda fs,da: gg.loc[fs,da].max()
items=[('Stat. features','STATnone',best('STAT','none'),C[3]),('Stat. + CORAL','',best('STAT','CORAL'),'#9b9b9b'),('1D-CNN raw','',cm('raw waveform (global)','1D-CNN'),'#c9c9c9'),
       ('Stat. + twin\ncalibration','',best('STAT','run-norm'),C[2]),('Twin residuals +\ncalibration','',best('PHYS-z','run-norm'),C[0]),('1D-CNN + twin\ncalibration','',cm('raw waveform (run-norm)','1D-CNN'),C[6])]
fig,ax=plt.subplots(figsize=(6.8,2.9))
for i,(l,_,v,c) in enumerate(items): ax.bar(i,v,.62,color=c); ax.text(i,v+.008,f'{v:.3f}',ha='center',fontsize=7,color=INK)
ax.set_xticks(range(len(items))); ax.set_xticklabels([l for l,*_ in items],fontsize=7); ax.set_ylim(0,.55)
ax.set_ylabel('Macro-F1, cross-mode\n(mean of M→L and L→M)'); ax.set_title('GPVS-Faults: transfer to an unseen operating mode',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_gpvs_transfer.png',dpi=300); plt.close(); print('figs4 ok')
