import sys; sys.path.insert(0,'../code')
import numpy as np, pandas as pd
from twin import twin_mpp
d=pd.read_parquet('lazz_raw.parquet')
day=d.irr>50
tw=twin_mpp(d.irr.values,d.pvt.values)
for k,v in tw.items(): d['tw_'+k]=np.where(day,v,np.nan)
d.to_parquet('lazz_twin.parquet')
n=d[day&(d.y==0)]
for s in '12':
    print('string',s,
      'I ratio med',np.round((n['idc'+s]/n.tw_Imp).median(),3),
      'V ratio med',np.round((n['vdc'+s]/n.tw_Vmp).median(),3),
      'P ratio med',np.round((n['idc'+s]*n['vdc'+s]/n.tw_Pmp).median(),3))
# per day ratio on normal
g=n.assign(r1=n.idc1/n.tw_Imp,r2=n.idc2/n.tw_Imp,v1=n.vdc1/n.tw_Vmp,v2=n.vdc2/n.tw_Vmp).groupby('day')[['r1','r2','v1','v2']].median().round(3)
print(g.to_string())
# lag check between irr and current
x=n.irr.values[:200000]; 
