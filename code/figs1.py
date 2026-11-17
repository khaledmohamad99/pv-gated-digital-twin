import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.axisbelow':True,'font.family':'DejaVu Sans'})
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']; INK='#333333'
OUT='./figs/'; M={}
d=pd.read_parquet('./lazz_feat2.parquet')
# ---- twin accuracy on healthy operation
n=d[(d.y==0)&(d.irr>200)]
M['normal_inverter_off_frac']=float(((n.vdc1<100)|(n.vdc2<100)).mean())
n=n[(n.vdc1>100)&(n.vdc2>100)]   # exclude inverter start-up / off-MPP intervals
for s in '12':
    P=n['idc'+s]*n['vdc'+s]; raw=n.tw_Pmp; cal=n.tw_Pmp*n['th_ri'+s]*n['th_rv'+s]
    M[f'twin_nRMSE_raw_s{s}']=float(np.sqrt(((P-raw)**2).mean())/P.mean()); M[f'twin_nRMSE_cal_s{s}']=float(np.sqrt(((P-cal)**2).mean())/P.mean())
    M[f'twin_nMBE_raw_s{s}']=float((raw-P).mean()/P.mean()); M[f'twin_nMBE_cal_s{s}']=float((cal-P).mean()/P.mean())
    M[f'twin_R2_cal_s{s}']=float(1-((P-cal)**2).sum()/((P-P.mean())**2).sum())
st=d[(d.y==0)&(d.irr>300)&(d.irr_sd30<0.02*d.irr)&(d.vdc1>100)&(d.vdc2>100)]
for s_ in '12':
    P_=st['idc'+s_]*st['vdc'+s_]
    for k_,v_ in [('cal',st.tw_Pmp*st['th_ri'+s_]*st['th_rv'+s_]),('raw',st.tw_Pmp)]:
        M[f'twin_stable_nRMSE_{k_}_s'+s_]=float(np.sqrt(((P_-v_)**2).mean())/P_.mean()); M[f'twin_stable_nMBE_{k_}_s'+s_]=float((v_-P_).mean()/P_.mean())
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
smp=n.sample(20000,random_state=0)
ax[0].scatter(smp.tw_Pmp*smp.th_ri1*smp.th_rv1/1000,smp.idc1*smp.vdc1/1000,s=1,alpha=.25,color=C[0],rasterized=True)
mx=2.6; ax[0].plot([0,mx],[0,mx],color=INK,lw=1,ls='--'); ax[0].set_xlabel('Twin-predicted string power (kW)'); ax[0].set_ylabel('Measured string power (kW)')
ax[0].set_title(f"(a) Healthy operation, string 1  (R² = {M['twin_R2_cal_s1']:.3f})",fontsize=9,loc='left')
g=d.groupby('minute').agg(t=('t','first'),a=('th_ri1','last'),b=('th_ri2','last'),c=('th_rv1','last'),e=('th_rv2','last'))
x=g.t/86400
for col,lab,cc in [('a','Current state, string 1',C[0]),('b','Current state, string 2',C[1]),('c','Voltage state, string 1',C[2]),('e','Voltage state, string 2',C[3])]:
    ax[1].plot(x,g[col],lw=1.5,color=cc,label=lab)
ax[1].set_xlabel('Elapsed time (days)'); ax[1].set_ylabel('Calibration state θ (–)')
ax[1].set_title('(b) Online twin calibration states',fontsize=9,loc='left'); ax[1].legend(frameon=False,fontsize=7,loc='lower right')
ax[1].annotate('common-mode optical step\n(cleaning-like event)',xy=(8.6,1.06),xytext=(2.0,1.075),fontsize=7,arrowprops=dict(arrowstyle='->',color=INK,lw=.8),color=INK)
fig.tight_layout(); fig.savefig(OUT+'fig_twin_validation.png',dpi=300); plt.close()
# ---- physics signatures
cls=['Normal','Short-circuit','Degradation','Open-circuit','Shading']
fig,ax=plt.subplots(1,4,figsize=(7.4,2.6))
for a,(f,lab) in zip(ax,[('cv_min','Min. voltage ratio  $c_V$'),('ci_min','Min. current ratio  $c_I$'),('n_bypass','Equiv. lost modules  $N_b$'),('R_extra','Apparent added $R_s$ (Ω)')]):
    data=[d[d.y==k][f].clip(upper=20 if f=='R_extra' else None).sample(min(20000,(d.y==k).sum()),random_state=0).values for k in range(5)]
    bp=a.boxplot(data,showfliers=False,patch_artist=True,widths=.6,medianprops=dict(color=INK))
    for p,cc in zip(bp['boxes'],C): p.set_facecolor(cc); p.set_alpha(.85); p.set_edgecolor('white')
    a.set_xticks(range(1,6)); a.set_xticklabels(['N','SC','DG','OC','SH']); a.set_title(lab,fontsize=8)
fig.tight_layout(); fig.savefig(OUT+'fig_physics_signatures.png',dpi=300); plt.close()
for k in range(5):
    q=d[d.y==k]; M[f'sig_{cls[k]}']=dict(cv_min=float(q.cv_min.median()),ci_min=float(q.ci_min.median()),n_bypass=float(q.n_bypass.median()),R_extra=float(q.R_extra.median()))
# ---- TECNALIA parameters
P=pd.read_csv('./tec_params.csv'); P=P[P.cost<1e-6]; P['day']=pd.to_datetime(P.day)
med=P.groupby('module')[['Rs','Rsh','Isc_n','FF']].median(); M['tec_params']=med.round(3).to_dict(orient='index'); M['tec_converged']=float((pd.read_csv('./tec_params.csv').cost<1e-6).mean())
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
mods=list(med.index)
for i,mname in enumerate(mods):
    q=P[P.module==mname].set_index('day').Rs.rolling(7,min_periods=3).median()
    ax[0].plot(q.index,q.values,lw=1.5,color=C[i],label=mname)
ax[0].set_ylabel('Identified $R_s$ (Ω)'); ax[0].set_title('(a) Series resistance (7-day median)',fontsize=9,loc='left'); ax[0].legend(frameon=False,fontsize=7,ncol=2)
ax[0].tick_params(axis='x',rotation=30)
ax[1].scatter(med.Rs,med.Rsh,s=60,color=C[:len(mods)],edgecolor='white',linewidth=2,zorder=3)
for i,mname in enumerate(mods): ax[1].annotate(mname,(med.Rs.iloc[i],med.Rsh.iloc[i]),xytext=(5,4),textcoords='offset points',fontsize=7,color=INK)
ax[1].set_yscale('log'); ax[1].set_xlabel('$R_s$ (Ω)'); ax[1].set_ylabel('$R_{sh}$ (Ω, log)'); ax[1].set_title('(b) Twin state space of aged modules',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_tecnalia_params.png',dpi=300); plt.close()
# ---- soiling
T=pd.read_csv('./soil_trace_S+D+Sh.csv',index_col=0,parse_dates=True)
fig,ax=plt.subplots(1,2,figsize=(7.4,3.1),gridspec_kw=dict(width_ratios=[1.35,1]))
ax[0].plot(T.index,T.SR_true,color=INK,lw=2,label='True soiling ratio')
ax[0].plot(T.index,T.twin,color=C[0],lw=1.5,label='Digital twin (ours)')
ax[0].plot(T.index,T.srr,color=C[1],lw=1.5,label='RdTools SRR')
ax[0].plot(T.index,T.naive,color=C[3],lw=1.2,label='Naive PR')
ax[0].set_ylabel('Soiling ratio (–)'); ax[0].set_title('(a) Soiling + Rs-degradation + shading (seed 0)',fontsize=9,loc='left'); ax[0].legend(frameon=False,fontsize=7,loc='lower left'); ax[0].tick_params(axis='x',rotation=30)
R=pd.concat([pd.read_csv('./res_soiling.csv'),pd.read_csv('./res_soiling_ext.csv')]); S=R.groupby(['scenario','method']).mae.mean().unstack()
meth=['Naive PR','RdTools SRR','Digital twin (ours)']; sc=['S','S+Sh','S+D','S+D+Sh','S+ShI','S+NU']; w=.26
for j,mm in enumerate(meth):
    v=S.loc[sc,mm].values*100; b=ax[1].bar(np.arange(6)+(j-1)*w,v,w-.03,color=[C[3],C[1],C[0]][j],label=mm)
    for xi,vi in zip(np.arange(6)+(j-1)*w,v): ax[1].text(xi,vi+.1,f'{vi:.1f}',ha='center',fontsize=5,color=INK,rotation=90,va='bottom')
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(sc,fontsize=7,rotation=30,ha='right',rotation_mode='anchor'); ax[1].set_ylabel('MAE of soiling ratio (%)'); ax[1].set_title('(b) Mean of 20 Monte-Carlo runs',fontsize=9,loc='left'); ax[1].legend(frameon=False,fontsize=6.5,loc='upper left',ncol=1); ax[1].set_ylim(0,12.5)
fig.tight_layout(); fig.savefig(OUT+'fig_soiling.png',dpi=300); plt.close()
M['soiling_mae']=R.groupby(['scenario','method']).agg(mae=('mae','mean'),mae_sd=('mae','std'),bias=('bias','mean'),rmse=('rmse','mean')).round(4).reset_index().to_dict(orient='records')
# ---- cleaning
CL=pd.read_csv('./res_cleaning.csv'); q=CL[(CL.price==0.0242)&(CL.C==190)].groupby('policy').total.agg(['mean','std'])
order=['every7','every14','every30','every60','naive','twin','oracle']; lab=['Fixed 7 d','Fixed 14 d','Fixed 30 d','Fixed 60 d','PR-driven','Twin-driven\n(ours)','Oracle']
fig,ax=plt.subplots(figsize=(5.2,2.8)); cols=['#9b9b9b']*4+[C[3],C[0],INK]
b=ax.bar(range(7),q.loc[order,'mean']/1000,color=cols,width=.65)
for i,v in enumerate(q.loc[order,'mean']/1000): ax.text(i,v+.3,f'{v:.1f}',ha='center',fontsize=7,color=INK)
ax.set_xticks(range(7)); ax.set_xticklabels(lab,fontsize=7); ax.set_ylabel('Soiling loss + cleaning cost\n(kUSD per MWp per year)')
ax.set_title('Cleaning scheduling\n(tariff 0.0242 USD/kWh, 190 USD/MWp per cleaning)',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_cleaning.png',dpi=300); plt.close()
M['cleaning']=json.load(open('./res_cleaning_summary.json'))
json.dump(M,open('./metrics_part1.json','w'),indent=1)
print({k:v for k,v in M.items() if k.startswith('twin')})
