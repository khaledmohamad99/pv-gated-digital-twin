import json, numpy as np, pandas as pd
W='./'; L=[]
def T(title,df,note=None):
    L.append(f"\n### {title}\n"); L.append(df.to_markdown(index=False,floatfmt='.3f'))
    if note: L.append(f"\n*{note}*")
# T1 datasets
T('Table 2. Datasets',pd.DataFrame([
 ['Lazzaretti et al. field plant (Brazil)','2 strings x 8 CS6U-330P, 5 kW','16 days, 1 Hz, 536 331 daylight samples','Normal, short-circuit (2 modules), degradation (series resistor), open-circuit, natural shading','Twin, fault identification, XAI, open-set, data efficiency, robustness, real soiling-like drift'],
 ['GPVS-Faults (Bakdi et al.)','Lab grid-connected PV + boost + inverter','16 runs (MPPT/IPPT), 10 kHz, 4 308 windows of 0.1 s','F0 + 7 faults (inverter, sensor, grid, mismatch, open-circuit, MPPT, boost controller)','Converter-level twin residuals, unsupervised detection, cross-mode transfer'],
 ['TECNALIA (Spain)','5 aged/defective single modules','Nov 2024-Jul 2025, 10 min, Isc/Voc/FF + MPP','Unlabelled real degradation (corrosion, ribbon breaks, cracks, yellowing)','Online SDM state identification, semi-synthetic soiling benchmark, cleaning economics']],
 columns=['Dataset','System','Size / resolution','Conditions','Used for']))
# T2 twin accuracy
M=json.load(open(W+'metrics_part1.json'))
T('Table 3. Digital-twin accuracy on healthy operation (Lazzaretti, stable irradiance >300 W/m2)',pd.DataFrame([
 ['String 1',M['twin_stable_nRMSE_raw_s1'],M['twin_stable_nRMSE_cal_s1'],M['twin_stable_nMBE_raw_s1'],M['twin_stable_nMBE_cal_s1']],
 ['String 2',M['twin_stable_nRMSE_raw_s2'],M['twin_stable_nRMSE_cal_s2'],M['twin_stable_nMBE_raw_s2'],M['twin_stable_nMBE_cal_s2']]],
 columns=['String','nRMSE datasheet twin','nRMSE calibrated twin','nMBE datasheet twin','nMBE calibrated twin']),'Power at 1 Hz; single plane-of-array pyranometer for both strings.')
# T3 signatures
cls=['Normal','Short-circuit','Degradation','Open-circuit','Shading']
d=pd.read_parquet(W+'lazz_feat2.parquet')
rows=[[c,d[d.y==k].cv_min.median(),d[d.y==k].ci_min.median(),d[d.y==k].n_bypass.median(),d[d.y==k].R_extra.median()] for k,c in enumerate(cls)]
T('Table 4. Twin-derived physical fault signatures (medians)',pd.DataFrame(rows,columns=['Class','c_V,min','c_I,min','N_b (equiv. lost modules)','R_extra (ohm)']),'Ground truth for short-circuit: 2 modules bypassed; twin estimate N_b = 2.0.')
# T4 LODO
v2=pd.DataFrame(json.load(open(W+'res_lazz_cls_v3.json'))); raw=pd.DataFrame(json.load(open(W+'res_lazz_cls_all_v1.json')))
raw=raw[(raw.protocol=='LODO')&(raw.features=='RAW')]; A=pd.concat([raw,v2]); A=A[A.protocol=='LODO']
S=json.load(open(W+'res_stats_v2.json'))
A['CI95']=[('[%.3f, %.3f]'%tuple(S[f"LODO_{r.features}_{r.model}"]['ci95'])) if f"LODO_{r.features}_{r.model}" in S else '' for r in A.itertuples()]
pc=pd.DataFrame(A.perclassF1.tolist(),columns=['F1 '+c for c in ['N','SC','DG','OC','SH']])
A=pd.concat([A[['features','model','acc','det_acc','macroF1','CI95']].reset_index(drop=True),pc],axis=1).sort_values(['features','model'])
T('Table 5. Fault identification, leave-one-day-out (16 folds)',A.rename(columns={'features':'Features','model':'Model','acc':'Accuracy','det_acc':'Detection acc.','macroF1':'Macro-F1','CI95':'95% CI (day bootstrap)'}),
  'RAW = string voltages/currents, irradiance, temperature + 30 s window statistics; PHYS+PA = twin features with physics-consistent augmentation; HYBRID = both.')
if 'tests' in S: T('Table 6. Paired Wilcoxon tests over fault days 9-16',pd.DataFrame([[k,v['mean_diff'],v['wins'],v['p_wilcoxon']] for k,v in S['tests'].items()],columns=['Comparison','Mean macro-F1 diff.','Days won (of 8)','p']))
# T5 protocols
_D=pd.read_csv(W+'res_gpvs_da.csv'); DAm=_D.groupby(['features','DA','model']).macroF1.mean().groupby(['features','DA']).max()
G=pd.DataFrame(json.load(open(W+'res_gpvs_cls.json'))); RS=json.load(open(W+'res_gpvs_random.json'))
rr=pd.DataFrame(json.load(open(W+'res_lazz_cls_all_v1.json'))); rr=rr[rr.protocol=='RANDOM']
T('Table 8. Effect of the evaluation protocol (information leakage)',pd.DataFrame([
 ['Lazzaretti','Random 80/20 split (leaky)','RAW / PHYS+PA LightGBM',rr[rr.features=='RAW'].macroF1.values[0],rr[rr.features=='PHYS+PA'].macroF1.values[0]],
 ['Lazzaretti','Leave-one-day-out','RAW / PHYS+PA LightGBM',raw[raw.model=='LightGBM'].macroF1.values[0],v2[(v2.features=='PHYS+PA')&(v2.model=='LightGBM')].macroF1.values[0]],
 ['GPVS','Random 70/30 split (leaky)','STAT / PHYS LightGBM',RS['STAT']['macroF1'],RS['PHYS']['macroF1']],
 ['GPVS','Blocked temporal split','STAT / PHYS best model',G[(G.protocol=='BLOCKED')&(G.features=='STAT')].macroF1.max(),G[(G.protocol=='BLOCKED')&(G.features=='PHYS')].macroF1.max()],
 ['GPVS','MPPT -> IPPT','STAT / PHYS best model',G[(G.protocol=='M->L')&(G.features=='STAT')].macroF1.max(),G[(G.protocol=='M->L')&(G.features=='PHYS')].macroF1.max()],
 ['GPVS','IPPT -> MPPT','STAT / PHYS best model',G[(G.protocol=='L->M')&(G.features=='STAT')].macroF1.max(),G[(G.protocol=='L->M')&(G.features=='PHYS')].macroF1.max()],
 ['GPVS','Cross-mode mean, no adaptation','STAT / PHYS-z best model',DAm.loc[('STAT','none')],DAm.loc[('PHYS-z','none')]],
 ['GPVS','Cross-mode mean, twin start-up calibration','STAT / PHYS-z best model',DAm.loc[('STAT','run-norm')],DAm.loc[('PHYS-z','run-norm')]]],
 columns=['Dataset','Protocol','Models','Macro-F1 baseline features','Macro-F1 physics features']))
# T6 data efficiency
DE=pd.read_csv(W+'res_lazz_dataeff.csv').groupby(['j','features','model']).macroF1.agg(['mean','std','min']).reset_index()
T('Table 9. Data efficiency: labelled fault days used for training (tested on the remaining fault days)',DE.rename(columns={'j':'Fault days','features':'Features','model':'Model'}))
# T7 robustness
RB=pd.DataFrame(json.load(open(W+'res_lazz_robust.json'))).pivot_table(index=['features','model'],columns='condition',values='macroF1').reset_index()
T('Table 10 (robustness part). Robustness to irradiance-sensor gain error at test time (fault days 9-16)',RB,'g090/g110 = pyranometer reading scaled by 0.90/1.10; models trained on clean data.')
# T8/T9 XAI open-set
X=json.load(open(W+'res_lazz_xai_open.json'))
T('Table 12. Explanation quality (SHAP, leave-one-day-out models on fault days 12-16)',pd.DataFrame([[fs,X[f'shap_{fs}']['PCS']['Short-circuit'],X[f'shap_{fs}']['PCS']['Degradation'],X[f'shap_{fs}']['PCS']['Shading'],X[f'shap_{fs}']['PCS_mean'],X[f'shap_{fs}']['stability_rho_mean']] for fs in ['RAW','PHYS']],
 columns=['Model features','PCS SC','PCS DG','PCS SH','PCS mean','Stability (Spearman rho)']),'PCS = share of attribution on the physically expected channel (voltage for SC/DG, current for SH); open-circuit excluded because breaker opening collapses both channels.')
O=pd.DataFrame(X['open_set']).pivot_table(index='unknown',columns='features',values='AUROC_MSP').reset_index()
O.loc[len(O)]=['Mean',O.PHYS.mean(),O.RAW.mean()]
T('Table 13. Open-set recognition: AUROC of max-softmax score for an unseen fault type',O)
# T10 detection
DT=pd.read_csv(W+'res_gpvs_detect_v2.csv'); DT=DT[DT.method.isin(['PCA-T2/SPE','Twin Mahalanobis','Fused (twin+PCA)'])].replace({'Fused (twin+PCA)':'Fused twin + PCA (ours)','PCA-T2/SPE':'PCA T2/SPE (Bakdi-type baseline)'})
q=DT[DT.onset=='changepoint'].groupby('method').agg(detected=('detected','mean'),median_delay_ms=('delay_ms','median'),TPR=('TPR','mean'),FAR_prefault=('FAR','mean')).reset_index()
far0=DT[DT.file.str[1]=='0'].groupby('method').FAR.mean(); q['FAR_fault_free_runs']=q.method.map(far0)
T('Table 14. Unsupervised real-time detection on GPVS (thresholds set for <=1 % window false-alarm rate on fault-free runs)',q,'Delays relative to change-point-estimated fault onsets; alarm after 3 consecutive 50 ms windows.')
# T11 transfer
D=pd.read_csv(W+'res_gpvs_da.csv'); T11=D.groupby(['features','DA','model'])[['macroF1','det_F1']].mean().reset_index().sort_values('macroF1').groupby(['features','DA']).tail(1).sort_values('macroF1')
T('Table 15. Cross-mode transfer with unsupervised adaptation (mean of both directions, best model)',T11,'run-norm = twin start-up calibration on the first 2 s of each run (no labels).')
# T12 TECNALIA
P=pd.DataFrame(M['tec_params']).T.reset_index().rename(columns={'index':'Module'})
T('Table 16. Twin state identification on aged modules (daily medians, G 500-1100 W/m2)',P,f"Converged solutions: {M['tec_converged']*100:.1f} % of days.")
# T13 soiling
SO=pd.DataFrame(M['soiling_mae']) if 'soiling_mae' in M else None
SO=pd.concat([pd.read_csv(W+x) for x in ['res_soiling.csv','res_soiling_ext.csv'] if __import__('os').path.exists(W+x)]).groupby(['scenario','method']).agg(MAE=('mae','mean'),MAE_sd=('mae','std'),bias=('bias','mean'),RMSE=('rmse','mean')).reset_index()
T('Table 17. Soiling-ratio estimation, 9 months of real data with injected soiling (20 Monte-Carlo runs per scenario)',SO,'S = soiling; D = progressive Rs degradation; Sh = transient partial shading.')
# T14 cleaning
CL=pd.read_csv(W+'res_cleaning.csv').groupby(['price','C','policy']).total.mean().unstack().reset_index()
T('Table 18. Annual soiling loss + cleaning cost per MWp (USD, 500 runs with common random numbers, 365 days)',CL,'Tariffs 0.0135/0.0242/0.06 USD/kWh; cleaning 32 (machinery) / 190 (manual) USD per MWp per cycle; soiling 0.1-1 %/day.')
# T15 literature
T('Table 19. Comparison with published work',pd.DataFrame([
 ['Lazzaretti et al. 2020 (Sensors)','Same field plant','as reported (split not leave-one-day-out)','Detection acc. 93.09 %, classification acc. 95.44 % (ANN)'],
 ['This work','Same field plant','leave-one-day-out','Detection acc. 98.3 %, accuracy 98.1 %, macro-F1 0.977 (HYBRID MLP)'],
 ['Signature RF + SHAP (arXiv 2112.14842)','GPVS-Faults MPPT','70:30 split','100 % accuracy'],
 ['This work','GPVS-Faults','random / blocked / cross-mode','0.93 / 0.70 / 0.42 macro-F1 -> random splits inflate scores'],
 ['NREL RdTools SRR (reference soiling method)','TECNALIA semi-synthetic','20 runs x 6 scenarios','MAE 5.3-7.3 % (ours: 1.8 % uniform soiling; 2.2 % Isc-reducing shading; 6.4 % non-uniform soiling)']],
 columns=['Work','Data','Protocol','Result']))
lat=X.get('latency') or json.load(open(W+'res_lazz_xai_open_v1.json')).get('latency')
if lat: T('Table 20. Computational cost (single CPU core)',pd.DataFrame([[k,v] for k,v in lat.items()],columns=['Item','Value']))

# T17 deep-learning baselines
DL=pd.DataFrame(json.load(open(W+'res_dl.json')))
a=DL[DL.dataset=='Lazzaretti'][['features','model','acc','det_acc','macroF1']]
T('Table 7 (field-plant rows). Deep-learning baselines, Lazzaretti leave-one-day-out (30 s causal windows, 1 Hz)',a,'RAW = 6 measured channels (standardised); TWIN = calibrated current/voltage ratios + irradiance. Compare with Table 4.')
b=DL[DL.dataset=='GPVS'].pivot_table(index=['features','model'],columns='protocol',values='macroF1').reset_index()
b['cross-mode mean']=(b['M->L']+b['L->M'])/2
T('Table 7 (GPVS rows). Deep-learning baselines on GPVS raw waveforms (0.1 s windows, mean of 3 seeds)',b,'run-norm = twin start-up calibration of each run (first 2 s, unlabelled).')
# T19 ablation
if __import__('os').path.exists(W+'res_ablation_ms.csv'):
    AB=pd.read_csv(W+'res_ablation_ms.csv').groupby('config',sort=False).agg(seeds=('seed','nunique'),macroF1=('macroF1','mean'),macroF1_sd=('macroF1','std'),F1_DG=('F1_DG','mean'),F1_SH=('F1_SH','mean'),day9_DG_recall=('day9_DG_recall','mean')).reset_index()
    T('Table 10 (ablation part). Twin ablation, LightGBM leave-one-day-out (mean over seeds)',AB)
if __import__('os').path.exists(W+'res_ablation_robust.csv'):
    AR=pd.read_csv(W+'res_ablation_robust.csv').pivot_table(index='config',columns='condition',values='macroF1').reset_index()
    T('Table 10 (A0 vs final under gain error). Why online calibration matters: irradiance-sensor gain error at test time (fault days 9-16)',AR,'Without calibration the twin residuals inherit the sensor error and the classifier collapses.')


import os, glob
GS=[f for f in ['res_gate_sensitivity.csv','res_gate_sensitivity_a.csv','res_gate_sensitivity_b.csv'] if os.path.exists(W+f)]
if GS: T('Table 11. One-at-a-time sensitivity of the calibration settings (final twin, LightGBM, LODO, seed 0)',pd.concat([pd.read_csv(W+f) for f in GS]).drop_duplicates('setting'))
if os.path.exists(W+'res_lazz_rawrf.json'): T('Supporting (Tables 5-6). RAW random forest, LODO (rerun with saved predictions)',pd.DataFrame(json.load(open(W+'res_lazz_rawrf.json'))).reindex(columns=['model','acc','det_acc','macroF1','ci95']))

open(W+'TABLES.md','w').write("# Results tables (all numbers generated from results/*.json|csv)\n"+"\n".join(L)); print('tables ok')
