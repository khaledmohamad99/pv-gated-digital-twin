import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('../code/figs1.py').read().split("OUT='./figs/'")[0])
OUT='./figs/'; W='./'
# ---- Lazzaretti benchmark (LODO)
R=pd.DataFrame(json.load(open(W+'res_lazz_cls_all.json')))
L=R[R.protocol=='LODO']
fig,ax=plt.subplots(figsize=(6.4,2.8)); models=['LightGBM','XGBoost','RandomForest','MLP']; fsl=['RAW','PHYS+PA']; w=.36
for j,fs in enumerate(fsl):
    v=[L[(L.features==fs)&(L.model==m)].macroF1.max() for m in models]
    ax.bar(np.arange(4)+(j-.5)*w,v,w-.04,color=[C[3],C[0]][j],label={'RAW':'Raw measurements','PHYS+PA':'Physics twin features (ours)'}[fs])
    for xi,vi in zip(np.arange(4)+(j-.5)*w,v): ax.text(xi,vi+.002,f'{vi:.3f}',ha='center',fontsize=6.5,color=INK)
ax.set_ylim(.9,1.0); ax.set_xticks(range(4)); ax.set_xticklabels(models); ax.set_ylabel('Macro-F1 (leave-one-day-out)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax.set_title('Fault identification, field plant (16 days, 1 Hz)',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_lazz_benchmark.png',dpi=300); plt.close()
# ---- data efficiency
DE=pd.read_csv(W+'res_lazz_dataeff.csv'); g=DE.groupby(['j','features','model']).macroF1.agg(['mean','std']).reset_index()
fig,ax=plt.subplots(1,2,figsize=(6.6,2.6),sharey=True)
for a,m in zip(ax,['LightGBM','MLP']):
    for j,fs in enumerate(['RAW','PHYS+PA']):
        q=g[(g.model==m)&(g.features==fs)]
        a.errorbar(q.j+(j-.5)*.08,q['mean'],yerr=q['std'],marker='o',ms=6,lw=2,capsize=3,color=[C[3],C[0]][j],label={'RAW':'Raw','PHYS+PA':'Physics twin (ours)'}[fs])
        for x,y in zip(q.j+(j-.5)*.08,q['mean']): a.text(x+(.07 if fs=='PHYS+PA' else -.07),y+(.006 if fs=='PHYS+PA' else -.006),f'{y:.3f}',ha=('left' if fs=='PHYS+PA' else 'right'),fontsize=6.5,color=INK,va='center')
    a.set_xticks([1,2]); a.set_xlabel('Labelled fault days in training'); a.set_title(m,fontsize=9,loc='left'); a.set_xlim(.7,2.5)
ax[0].set_ylabel('Macro-F1 on unseen fault days'); ax[0].legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout(); fig.savefig(OUT+'fig_data_efficiency.png',dpi=300); plt.close()
# ---- confusion matrices
P=np.load(W+'pred_lazz.npz'); d=pd.read_parquet(W+'lazz_feat2.parquet'); y=d.y.values
from sklearn.metrics import confusion_matrix
fig,ax=plt.subplots(1,2,figsize=(7.0,3.1))
for a,(k,t) in zip(ax,[('LODO_RAW_MLP','(a) Raw + MLP'),('LODO_HYBRID_MLP','(b) Twin + raw (hybrid) + MLP (ours)')]):
    p=P[k]; m=p>=0; cm=confusion_matrix(y[m],p[m],labels=range(5)); cmn=cm/cm.sum(1,keepdims=True)
    a.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
    for i in range(5):
        for j in range(5): a.text(j,i,f'{cmn[i,j]*100:.1f}',ha='center',va='center',fontsize=7,color='white' if cmn[i,j]>.6 else INK)
    a.set_xticks(range(5)); a.set_yticks(range(5)); a.set_xticklabels(['N','SC','DG','OC','SH']); a.set_yticklabels(['N','SC','DG','OC','SH'])
    a.set_xlabel('Predicted'); a.set_ylabel('True'); a.set_title(t,fontsize=9,loc='left'); a.grid(False)
fig.tight_layout(); fig.savefig(OUT+'fig_confusion.png',dpi=300); plt.close()
# ---- GPVS protocols
G=pd.DataFrame(json.load(open(W+'res_gpvs_cls.json'))); RS=json.load(open(W+'res_gpvs_random.json'))
fig,ax=plt.subplots(figsize=(6.4,2.8))
prot=['Random 70/30\n(leaky)','Blocked\ntemporal','MPPT→IPPT','IPPT→MPPT']
st=[RS['STAT']['macroF1']]+[G[(G.protocol==p)&(G.features=='STAT')].macroF1.max() for p in ['BLOCKED','M->L','L->M']]
ph=[RS['PHYS']['macroF1']]+[G[(G.protocol==p)&(G.features=='PHYS')].macroF1.max() for p in ['BLOCKED','M->L','L->M']]
for j,(v,lab,cc) in enumerate([(st,'Statistical signatures',C[3]),(ph,'Physics twin residuals (ours)',C[0])]):
    xs=np.arange(4)+(j-.5)*.36; ax.bar(xs,v,.32,color=cc,label=lab)
    for xi,vi in zip(xs,v): ax.text(xi,vi+.01,f'{vi:.2f}',ha='center',fontsize=7,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(prot,fontsize=7.5); ax.set_ylabel('Macro-F1 (best model)'); ax.set_ylim(0,1.05)
ax.legend(frameon=False,fontsize=7); ax.set_title('GPVS-Faults (10 kHz): evaluation protocol vs generalisation',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_gpvs_protocols.png',dpi=300); plt.close()
# ---- XAI
X=json.load(open(W+'res_lazz_xai_open.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
cl=['Short-circuit','Degradation','Shading']
for j,fs in enumerate(['RAW','PHYS']):
    v=[X[f'shap_{fs}']['PCS'][c] for c in cl]; xs=np.arange(3)+(j-.5)*.36
    ax[0].bar(xs,v,.32,color=[C[3],C[0]][j],label={'RAW':'Raw model','PHYS':'Physics twin model (ours)'}[fs])
    for xi,vi in zip(xs,v): ax[0].text(xi,vi+.01,f'{vi:.2f}',ha='center',fontsize=7,color=INK)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(cl); ax[0].set_ylim(0,1); ax[0].set_ylabel('Physics-consistency score (PCS)')
ax[0].legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2); ax[0].set_title('(a) SHAP attribution on the physically\n     expected measurement channel',fontsize=8.5,loc='left')
gi=pd.Series(X['shap_PHYS']['global_importance']).sort_values().tail(10)
ax[1].barh(range(len(gi)),gi.values,color=C[0]); ax[1].set_yticks(range(len(gi))); ax[1].set_yticklabels(gi.index,fontsize=7)
ax[1].set_xlabel('Mean |SHAP| share'); ax[1].set_title('(b) Global importance, physics twin model',fontsize=8.5,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_xai.png',dpi=300); plt.close()
# ---- open set
O=pd.DataFrame(X['open_set'])
fig,ax=plt.subplots(figsize=(6.0,2.7)); u=['Short-circuit','Degradation','Open-circuit','Shading']
a=[O[(O.unknown==k)&(O.features=='RAW')].AUROC_MSP.values[0] for k in u]
b=[O[(O.unknown==k)&(O.features=='PHYS')].AUROC_MSP.values[0] for k in u]
c=[O[(O.unknown==k)&(O.features=='PHYS')].AUROC_physics.values[0] for k in u]
for j,(v,lab,cc) in enumerate([(a,f'Raw model (mean {np.mean(a):.2f})',C[3]),(b,f'Physics twin model, ours (mean {np.mean(b):.2f})',C[0])]):
    xs=np.arange(4)+(j-.5)*.36; ax.bar(xs,v,.32,color=cc,label=lab)
    for xi,vi in zip(xs,v): ax.text(xi,vi+.01,f'{vi:.2f}',ha='center',fontsize=6.5,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([f'Unseen:\n{k}' for k in u],fontsize=7.5); ax.set_ylim(0,1.1); ax.set_ylabel('AUROC, max-softmax score\n(unseen fault vs rest)')
ax.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2); ax.set_title('Open-set: leave-one-fault-type-out',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_openset.png',dpi=300); plt.close()
print('figs2 ok')
