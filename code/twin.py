"""Physics layer of the digital twin: CEC single-diode model of CS6U-330P (SAM database)."""
import numpy as np, pvlib
M=pvlib.pvsystem.retrieve_sam('CECMod')['Canadian_Solar_Inc__CS6U_330P']
NS=8
def sdm(G,T,m=M,IL_scale=1.0,Rs_scale=1.0,Rsh_scale=1.0):
    G=np.clip(np.asarray(G,float),1.0,None);T=np.asarray(T,float)
    IL,I0,Rs,Rsh,a=pvlib.pvsystem.calcparams_cec(G,T,m['alpha_sc'],m['a_ref'],m['I_L_ref'],m['I_o_ref'],m['R_sh_ref'],m['R_s'],m['Adjust'])
    return IL*IL_scale,I0,Rs*Rs_scale,Rsh*Rsh_scale,a
def twin_mpp(G,T,ns=NS,**kw):
    IL,I0,Rs,Rsh,a=sdm(G,T,**kw)
    o=pvlib.pvsystem.singlediode(IL,I0,Rs,Rsh,a,method='newton')
    return dict(Imp=np.asarray(o['i_mp']),Vmp=np.asarray(o['v_mp'])*ns,Isc=np.asarray(o['i_sc']),Voc=np.asarray(o['v_oc'])*ns,Pmp=np.asarray(o['p_mp'])*ns)
if __name__=='__main__':
    print({k:np.round(float(v[0]),2) for k,v in twin_mpp([1000],[25]).items()})
