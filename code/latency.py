"""Single-core latency of the twin and the classifiers (Table 19)."""
import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import PHYS
from twin import twin_mpp
d=pd.read_parquet('lazz_feat2.parquet'); rng=np.random.default_rng(0)
tr=d[(d.y.isin([1,2,3]))|(rng.random(len(d))<0.2)]
X,y=tr[PHYS].values,tr.y.values; x1=d[PHYS].values[:1]; xb=d[PHYS].values[:100000]
out={}
for name,m in [('LightGBM',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,class_weight='balanced',verbose=-1,n_jobs=1,random_state=0)),
               ('MLP',make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=200,early_stopping=True,random_state=0)))]:
    m.fit(X,y); t=time.perf_counter()
    for _ in range(2000): m.predict(x1)
    out[f'{name}_single_ms']=(time.perf_counter()-t)/2000*1e3
    t=time.perf_counter(); m.predict(xb); out[f'{name}_throughput_per_s']=len(xb)/(time.perf_counter()-t)
G=np.full(10000,800.0); T=np.full(10000,40.0)
t=time.perf_counter(); twin_mpp(G,T); out['twin_sdm_us_per_sample_batched']=(time.perf_counter()-t)/len(G)*1e6
t=time.perf_counter()
for _ in range(200): twin_mpp(G[:1],T[:1])
out['twin_sdm_ms_single']=(time.perf_counter()-t)/200*1e3
json.dump(out,open('res_latency.json','w'),indent=1); print(out)
