import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(7.4,3.4)); ax.set_xlim(0,108); ax.set_ylim(0,48); ax.axis('off')
INK='#1f1f1f'; SUB='#5c5c5c'; BL='#2a78d6'; OR='#e39a00'; GR='#1a9e6e'; GY='#8a8a8a'; PU='#6a4fc8'
def box(x,y,w,h,title,sub=None,col=BL,fill='white',fs=6.2,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.0',fc=fill,ec=col,lw=1.2))
    if sub:
        ax.text(x+w/2,y+h*0.64,title,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',color=INK)
        ax.text(x+w/2,y+h*0.28,sub,ha='center',va='center',fontsize=5.0,color=SUB)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',color=INK,linespacing=1.2)
def group(x,y,w,h,label,col):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.6',fc='none',ec=col,lw=1.0,ls=(0,(4,2.5))))
    ax.text(x+w/2,y+h-1.8,label,ha='center',va='center',fontsize=6.0,weight='bold',color=col)
def arr(x1,y1,x2,y2,col=INK,rad=0.0,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=8,color=col,lw=1.0,ls=ls,
                 connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0))
# measurements
group(0.8,4,15.6,40,'Measurements',GY)
for i,(t,s) in enumerate([('String I, V','1 Hz – 10 kHz'),('G, T','irradiance, temp.'),('Module I–V','Isc, Voc, MPP')]):
    box(2.4,30.5-i*11.5,12.4,8,t,s,col=GY)
# twin
group(20,4,35,40,'Self-calibrating digital twin',BL)
box(22.5,24,13.5,11,'Single-diode','model',col=BL,fill='#eef4fc',fs=6.6)
box(39.5,24,13.5,11,'Physics gate','online calibration',col=BL,fill='#eef4fc',fs=6.6)
arr(36.2,29.5,39.3,29.5)
arr(46.2,35.3,29.3,35.3,col=BL,rad=0.45,ls='--'); ax.text(37.75,37.3,'state θ',ha='center',va='center',fontsize=5.2,color=BL)
arr(46.2,23.8,46.2,13.8,col=GY); arr(44.0,23.8,29.3,13.8,col=GY)
box(22.5,6.5,13.5,7,'Common-mode\n→ absorb',None,col=GR,fill='#e9f6ef',fs=5.4,bold=False)
box(39.5,6.5,13.5,7,'Differential\n→ keep',None,col=OR,fill='#fff3dc',fs=5.4,bold=False)
for i in range(3): arr(15.0,34.5-i*11.5,22.3,29.5)
# features
box(59,21.5,12,16,'Twin\nfeatures',None,col=BL,fill='#eef4fc',fs=6.6)
arr(53.2,29.5,58.8,29.5)
# outputs
group(75,4,32,40,'Intelligence & decision',GR)
rows=[('ML classifier','fault type',PU,'SHAP + PCS','explanation check'),
      ('State estimation','fault size',OR,'N_b, R_s, R_sh','fault magnitude'),
      ('Soiling estimator','I_sc ratio',GR,'Cleaning rule','loss = cost')]
for i,(t1,s1,c,t2,s2) in enumerate(rows):
    y=30-i*11.5
    box(77,y,14.4,8,t1,s1,col=c,fs=5.4); box(93.6,y,12.4,8,t2,s2,col=c,fs=5.4)
    arr(91.6,y+4,93.4,y+4)
    arr(71.2,29.5,76.8,y+4)
fig.savefig('./figs/fig_framework.png',dpi=400,bbox_inches='tight',pad_inches=0.04,facecolor='white'); print('ok')
