"""Deep-learning baselines: 1D-CNN and LSTM on raw sequences vs twin-ratio sequences (Lazzaretti LODO) and on raw
waveform windows (GPVS blocked + cross-mode, with/without twin start-up calibration)."""
import sys,json,time; sys.path.insert(0,'../code')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
torch.set_num_threads(1); torch.manual_seed(0)
W='./'
class CNN(nn.Module):
    def __init__(s,c,k):
        super().__init__(); s.f=nn.Sequential(nn.Conv1d(c,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,64,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1)); s.o=nn.Linear(64,k)
    def forward(s,x): return s.o(s.f(x.transpose(1,2)).squeeze(-1))
class LSTM(nn.Module):
    def __init__(s,c,k): super().__init__(); s.l=nn.LSTM(c,64,batch_first=True); s.o=nn.Linear(64,k)
    def forward(s,x): h,_=s.l(x); return s.o(h[:,-1])
def train(M,X,y,K,epochs=6,bs=512,seed=0):
    torch.manual_seed(seed); m=M(X.shape[2],K); opt=torch.optim.Adam(m.parameters(),1e-3)
    c=np.bincount(y,minlength=K).astype(float); w=torch.tensor(len(y)/(K*np.maximum(c,1)),dtype=torch.float32)
    lf=nn.CrossEntropyLoss(weight=w); Xt=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y)
    g=torch.Generator().manual_seed(seed)
    for e in range(epochs):
        p=torch.randperm(len(y),generator=g)
        for i in range(0,len(y),bs):
            b=p[i:i+bs]; opt.zero_grad(); l=lf(m(Xt[b]),yt[b]); l.backward(); opt.step()
    return m
def predict(m,X,bs=4096):
    m.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),bs): out.append(m(torch.tensor(X[i:i+bs],dtype=torch.float32)).argmax(1).numpy())
    return np.concatenate(out)
RES=[]
# ---------------- Lazzaretti ----------------
d=pd.read_parquet(W+'lazz_feat2.parquet').reset_index(drop=True)
SEQ=30
CH={'RAW':['vdc1','vdc2','idc1','idc2','irr','pvt'],'TWIN':['ci1','ci2','cv1','cv2','irr']}
A={}
for k,cols in CH.items():
    X=d[cols].values.astype(np.float32)
    if k=='RAW': mu,sd=X.mean(0),X.std(0)+1e-6; X=(X-mu)/sd
    else: X=np.clip(X,-3,3); X[:,-1]=X[:,-1]/1000
    A[k]=X
day=d.day.values; y=d.y.values
# causal window index (same day only)
start=np.r_[0,np.flatnonzero(np.diff(day))+1]; first=np.zeros(len(d),int)
for s,e in zip(start,np.r_[start[1:],len(d)]): first[s:e]=s
IDX=np.clip(np.arange(len(d))[:,None]-np.arange(SEQ)[::-1][None,:],first[:,None],None)
rng=np.random.default_rng(0)
for feat in ['RAW','TWIN']:
    for name,M in [('1D-CNN',CNN),('LSTM',LSTM)]:
        t=time.time(); pred=np.full(len(d),-1)
        for k in sorted(np.unique(day)):
            tr=np.flatnonzero((day!=k)&((np.isin(y,[1,2,3]))|(rng.random(len(d))<0.2))); te=np.flatnonzero(day==k)
            m=train(M,A[feat][IDX[tr]],y[tr],5)
            pred[te]=predict(m,A[feat][IDX[te]])
        r=dict(dataset='Lazzaretti',protocol='LODO',features=feat,model=name,acc=accuracy_score(y,pred),macroF1=f1_score(y,pred,average='macro'),
               det_acc=accuracy_score(y>0,pred>0),perclassF1=f1_score(y,pred,average=None).tolist(),sec=time.time()-t)
        RES.append(r); print(r,flush=True); np.save(W+f'pred_dl_{feat}_{name}.npy',pred)
        json.dump(RES,open(W+'res_dl.json','w'),indent=1)
# ---------------- GPVS ----------------
F=pd.read_pickle(W+'gpvs_feat.pkl').reset_index(drop=True)
SIG=['Ipv','Vpv','Vdc','ia','ib','ic','va','vb','vc','Iabc','If','Vabc','Vf']
Xs=[];meta=[]
for f in sorted(F.file.unique()):
    g=pd.read_pickle(W+f'gpvs_{f}.pkl')[SIG].values.astype(np.float32)
    rows=F[F.file==f].sort_values('t_end')
    for j,(i0) in enumerate(range(0,len(g)-1000,500)):
        Xs.append(g[i0:i0+1000:5])
    meta.append(rows)
X=np.stack(Xs); Mt=pd.concat(meta).reset_index(drop=True); yg=Mt.y.values
assert len(X)==len(Mt)
def norm_global(Xtr,Xte):
    mu=Xtr.reshape(-1,13).mean(0); sd=Xtr.reshape(-1,13).std(0)+1e-6; return (Xtr-mu)/sd,(Xte-mu)/sd
def norm_run(X):   # twin start-up calibration: per run, statistics of the first 2 s (healthy)
    Z=X.copy()
    for f in Mt.file.unique():
        m=(Mt.file==f).values; c=m&(Mt.t_end.values<=2.0); mu=X[c].reshape(-1,13).mean(0); sd=X[c].reshape(-1,13).std(0)+1e-6; Z[m]=(X[m]-mu)/sd
    return Z
def blocked():
    tr=[];te=[]
    for (f,yy),g in Mt.groupby(['file','y']):
        ix=g.sort_values('t_end').index.values; n=int(len(ix)*.6); tr+=list(ix[:n]); te+=list(ix[n+3:])
    return np.array(tr),np.array(te)
for proto in ['BLOCKED','M->L','L->M']:
    if proto=='BLOCKED': tr,te=blocked()
    else: s,t=proto.split('->'); tr=np.flatnonzero(Mt['mode']==s); te=np.flatnonzero(Mt['mode']==t)
    for normtag in ['global','run-norm']:
        if normtag=='global': a,b=norm_global(X[tr],X[te])
        else: Z=norm_run(X); a,b=Z[tr],Z[te]
        for name,M in [('1D-CNN',CNN),('LSTM',LSTM)]:
            ps=[]
            for seed in range(3):
                m=train(M,a,yg[tr],8,epochs=30,bs=128,seed=seed); ps.append(f1_score(yg[te],predict(m,b),average='macro'))
            r=dict(dataset='GPVS',protocol=proto,features='raw waveform ('+normtag+')',model=name,macroF1=float(np.mean(ps)),macroF1_sd=float(np.std(ps)))
            RES.append(r); print(r,flush=True); json.dump(RES,open(W+'res_dl.json','w'),indent=1)
