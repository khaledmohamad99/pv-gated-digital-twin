import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white'})
W,H=13.3/2.54,5.3/2.54
fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,133); ax.set_ylim(0,53); ax.axis('off')
INK='#1f1f1f'; BL='#2a78d6'; OR='#e39a00'; GR='#1a9e6e'; GY='#7a7a7a'; LB='#eaf2fc'
def panel(x,w,title,col):
    ax.add_patch(FancyBboxPatch((x,4.5),w,44,boxstyle='round,pad=0.6,rounding_size=2',fc='white',ec=col,lw=1.4))
    ax.text(x+w/2,45.0,title,ha='center',va='center',fontsize=7.2,weight='bold',color=col)
def arrow(x1,x2,y=27):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.4,mutation_scale=12))
# ---- panel 1: plant
panel(1.5,33,'PV plant data',GY)
for s,y0 in enumerate([30,17.5]):
    for i in range(4):
        x0=5+i*6.2
        ax.add_patch(Polygon([[x0,y0],[x0+5.4,y0],[x0+6.2,y0+7],[x0+0.8,y0+7]],closed=True,fc=(OR if (s==1 and i==1) else '#24508f'),ec='white',lw=0.6))
        for k in range(1,3): ax.plot([x0+k*1.8+0.3*k*0.9/2.7*2,x0+k*1.8+0.8],[y0,y0+7],color='#6f9bd6',lw=0.35)
    ax.text(31.5,y0+3.5,f'S{s+1}',fontsize=6,va='center',color=INK)
rng=np.random.default_rng(1)
for _ in range(30):
    yy=rng.choice([rng.uniform(30.5,36.5),rng.uniform(18,24)]); ax.add_patch(Circle((rng.uniform(6,29),yy),0.28,fc='#b08a4a',ec='none',alpha=.9))
ax.text(18,27.6,'dust: all strings | fault: one module',ha='center',va='center',fontsize=4.4,color=GY)
ax.text(18,13.0,'Faults, soiling and sensor drift\nlook alike in raw I, V, G signals',ha='center',va='center',fontsize=5.2,color=INK,linespacing=1.3)
ax.text(18,7.6,'3 public datasets: field (1 Hz),\nlab (10 kHz), aged modules',ha='center',va='center',fontsize=4.6,color=GY,linespacing=1.25)
arrow(35.4,39.2)
# ---- panel 2: twin
panel(40,42,'Self-calibrating digital twin',BL)
ax.add_patch(FancyBboxPatch((43.5,32.5),35,7.6,boxstyle='round,pad=0.3,rounding_size=1',fc=LB,ec=BL,lw=0.9))
ax.text(61,36.3,'Single-diode model → expected I, V',ha='center',va='center',fontsize=5.2,color=INK)
ax.text(61,28.7,'Physics gate',ha='center',va='center',fontsize=6.4,weight='bold',color=BL)
ax.add_patch(FancyBboxPatch((43.5,13.2),17,12,boxstyle='round,pad=0.3,rounding_size=1',fc='#e8f6ef',ec=GR,lw=0.9))
ax.text(52,22.4,'Common-mode',ha='center',fontsize=5.0,weight='bold',color=GR)
ax.text(52,17.6,'soiling, sensor\ndrift → absorb',ha='center',va='center',fontsize=5.2,color=INK,linespacing=1.25)
ax.add_patch(FancyBboxPatch((61.5,13.2),17,12,boxstyle='round,pad=0.3,rounding_size=1',fc='#fff3dc',ec=OR,lw=0.9))
ax.text(70,22.4,'Differential',ha='center',fontsize=5.0,weight='bold',color=OR)
ax.text(70,17.6,'string faults →\nkeep & quantify',ha='center',va='center',fontsize=4.8,color=INK,linespacing=1.25)
ax.text(61,8.4,'Explainable ML (SHAP +\nphysics-consistency score)',ha='center',va='center',fontsize=4.6,color=GY,linespacing=1.25)
arrow(82.9,86.7)
# ---- panel 3: results
panel(87.5,44,'Outcomes',GR)
tiles=[('0.977','macro-F1 fault ID\n(leave-one-day-out)',BL),('0.96 vs 0.57','macro-F1 under\n±10 % sensor error',BL),
       ('1.8 %','soiling MAE\n(uniform soiling)',GR),('4–11 % lower','cleaning cost vs\nfixed plan (simulated)',GR)]
for i,(big,small,c) in enumerate(tiles):
    x=90+(i%2)*20.5; y=25.5-(i//2)*18.5
    ax.add_patch(FancyBboxPatch((x,y),18.5,15.5,boxstyle='round,pad=0.3,rounding_size=1',fc='white',ec=c,lw=0.9))
    ax.text(x+9.25,y+11.2,big,ha='center',va='center',fontsize=7.6 if len(big)<8 else 6.0,weight='bold',color=c)
    ax.text(x+9.25,y+4.6,small,ha='center',va='center',fontsize=4.6,color=INK,linespacing=1.2)
for ext in ['png','tiff']:
    fig.savefig(f'figs/graphical_abstract.{ext}',dpi=400,facecolor='white')
fig.savefig('figs/graphical_abstract.pdf',facecolor='white')
print('ok')
