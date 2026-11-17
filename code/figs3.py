import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('../code/figs1.py').read().split("OUT='./figs/'")[0])
OUT='./figs/'; W='./'
D=pd.read_csv(W+'res_gpvs_da.csv'); g=D.groupby(['features','DA','model']).macroF1.mean().reset_index()
best=lambda fs,da: g[(g.features==fs)&(g.DA==da)].macroF1.max()
items=[('Statistical\nsignatures','STAT','none',C[3]),('+ CORAL\n(domain adapt.)','STAT','CORAL','#9b9b9b'),('+ twin start-up\ncalibration','STAT','run-norm',C[2]),('Twin residuals +\nstart-up calibration (ours)','PHYS-z','run-norm',C[0])]
fig,ax=plt.subplots(figsize=(6.2,2.8))
v=[best(fs,da) for _,fs,da,_ in items]
ax.bar(range(4),v,.6,color=[c for *_,c in items])
for i,x in enumerate(v): ax.text(i,x+.008,f'{x:.3f}',ha='center',fontsize=7.5,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([l for l,*_ in items],fontsize=7.5); ax.set_ylabel('Macro-F1, cross-mode\n(mean of M→L and L→M)'); ax.set_ylim(0,.5)
ax.set_title('GPVS-Faults: transfer to an unseen operating mode',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_gpvs_transfer.png',dpi=300); plt.close()
print('ok',v)
