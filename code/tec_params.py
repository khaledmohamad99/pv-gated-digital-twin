"""Online SDM parameter identification (digital-twin state estimation) on TECNALIA aged modules."""
import numpy as np, pandas as pd, glob
from scipy.optimize import least_squares
k=1.380649e-23;q=1.602176634e-19
CELLS={'Atersa':60,'JaSolar3':60,'NingboSolar':72,'Photowatt':72,'TrinaSolar':72}
def load(name):
    f=f'../data/TECNALIA/data_{name}/data_{name}.csv'
    d=pd.read_csv(f,sep=';',parse_dates=['Fecha']).replace(-9999,np.nan)
    d=d.rename(columns={'Fecha':'ts','Vmpp(V)':'Vmp','Impp(A)':'Imp','Temp. Mod (°C)':'T','Front GPOA (W/m²)':'G','Isc(A)':'Isc','Voc(A)':'Voc','Pmpp (W)':'Pmp'})
    return d[['ts','Vmp','Imp','T','G','Isc','Voc','FF','Pmp']]
def solve(Isc,Voc,Imp,Vmp,T,Ncell,n=1.2):
    a=n*Ncell*k*(T+273.15)/q
    def res(x):
        Rs,lRsh,lI0=x; Rsh=np.exp(lRsh); I0=np.exp(lI0)
        IL=Isc*(1+Rs/Rsh)+I0*(np.exp(Isc*Rs/a)-1)
        e2=IL-I0*(np.exp(Voc/a)-1)-Voc/Rsh
        ex=np.exp((Vmp+Imp*Rs)/a); e3=IL-I0*(ex-1)-(Vmp+Imp*Rs)/Rsh-Imp
        g=I0/a*ex+1/Rsh; dIdV=-g/(1+Rs*g); e4=Imp+Vmp*dIdV
        return [e2/Isc,e3/Isc,e4/Isc]
    best=None
    for x0 in ([0.3,np.log(300),np.log(1e-9)],[0.8,np.log(100),np.log(1e-7)],[0.1,np.log(1000),np.log(1e-11)]):
        try:
            r=least_squares(res,x0,bounds=([0,np.log(5),np.log(1e-16)],[5,np.log(1e5),np.log(1e-3)]))
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    if best is None: return np.nan,np.nan,np.nan,np.inf
    Rs,lRsh,lI0=best.x; return Rs,np.exp(lRsh),np.exp(lI0),best.cost
def main():
    out=[]
    for name in CELLS:
        d=load(name); s=d[(d.G>500)&(d.G<1100)&d.FF.between(40,90)&d.Isc.notna()].copy()
        s['hour']=s.ts.dt.floor('D')   # daily robust state: median of stable high-irradiance samples
        g=s.groupby('hour').median(numeric_only=True); g=g[s.groupby('hour').size()>=3]
        g=g.dropna(subset=["Isc","Voc","Imp","Vmp","T"])
        for t,r in g.iterrows():
            Rs,Rsh,I0,c=solve(r.Isc,r.Voc,r.Imp,r.Vmp,r['T'],CELLS[name])
            out.append(dict(module=name,day=t,G=r.G,T=r['T'],Isc_n=r.Isc/r.G*1000,Rs=Rs,Rsh=Rsh,I0=I0,cost=c,FF=r.FF,Pmp_n=r.Pmp/r.G*1000))
        print(name,len(g),flush=True)
    o=pd.DataFrame(out); o.to_csv('tec_params.csv',index=False)
    ok=o[o.cost<1e-6]
    print(ok.groupby('module')[['Rs','Rsh','Isc_n','FF','Pmp_n']].median().round(3).to_string())
    print('converged frac',(o.cost<1e-6).mean().round(3))

if __name__=='__main__': main()
