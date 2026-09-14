"""Journal figures. Categorical slots 1-3 (validated all-pairs); shape = secondary encoding."""
import math, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
D='/workspace/icbmehs_c3_skin/data/'; F='/workspace/icbmehs_c3_skin/figs/'
import os; os.makedirs(F,exist_ok=True)
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
INK,INK2,GRID='#0b0b0b','#52514e','#d8d7d2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
 'axes.edgecolor':INK2,'axes.linewidth':0.7,'xtick.color':INK2,'ytick.color':INK2,
 'text.color':INK,'axes.labelcolor':INK,'figure.dpi':300,'savefig.dpi':300,
 'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','savefig.pad_inches':0.02})
def grid(ax):
    ax.grid(True,color=GRID,linewidth=0.5,alpha=0.9); ax.set_axisbelow(True)
def stats(o,p): return r2_score(o,p), math.sqrt(mean_squared_error(o,p))

# ---------------- Fig 1: observed vs predicted, external test ----------------
t=pd.read_csv(D+'external_test_predictions.csv')
o=t['obs'].values
panels=[('Random forest',t['pred_Random forest'].values,BLUE),
        ('Potts-Guy (1992)',(-2.72+0.71*0+0)*0,None)]
a=pd.read_csv(D+'compounds_modelling.csv'); ite=np.load(D+'split_test_idx.npy')
pg=(-2.72+0.71*a['logP'].values-0.0061*a['MW'].values-math.log10(3600.0))[ite]
panels[1]=('Potts-Guy (1992)',pg,ORANGE)
fig,axs=plt.subplots(1,2,figsize=(6.6,3.3),sharex=True,sharey=True)
lims=(min(o.min(),pg.min())-0.4, max(o.max(),pg.max())+0.4)
for ax,(nm,p,c) in zip(axs,panels):
    grid(ax); ax.plot(lims,lims,color=INK2,lw=0.8,ls='--',zorder=1)
    ax.scatter(o,p,s=26,facecolor=c,edgecolor='white',linewidth=0.6,alpha=0.9,zorder=3)
    r2,rm=stats(o,p)
    ax.text(0.04,0.96,'%s\n$R^2$ = %.3f\nRMSE = %.3f'%(nm,r2,rm),transform=ax.transAxes,
            va='top',ha='left',fontsize=7.5,color=INK,
            bbox=dict(fc='white',ec=GRID,lw=0.5,boxstyle='round,pad=0.35'))
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel('Observed log $K_p$ (cm s$^{-1}$)')
axs[0].set_ylabel('Predicted log $K_p$ (cm s$^{-1}$)')
fig.savefig(F+'Fig1.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)

# ---------------- Fig 2: permutation importance ----------------
imp=pd.read_csv(D+'descriptor_importance.csv').head(10).iloc[::-1]
fig,ax=plt.subplots(figsize=(3.3,3.3)); grid(ax); ax.grid(axis='y',visible=False)
ax.barh(imp.descriptor,imp.imp_mean,xerr=imp.imp_sd,height=0.62,color=BLUE,
        error_kw=dict(ecolor=INK2,lw=0.7,capsize=2))
ax.set_xlabel('Decrease in $R^2$ when permuted'); ax.axvline(0,color=INK2,lw=0.7)
for y,(v,s) in enumerate(zip(imp.imp_mean,imp.imp_sd)):
    ax.text(v+s+0.008,y,'%.3f'%v,va='center',fontsize=6.5,color=INK2)
ax.set_xlim(right=imp.imp_mean.max()+imp.imp_sd.max()+0.07)
fig.savefig(F+'Fig2.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)

# ---------------- Fig 3: analgesic subgroup, leave-analgesics-out ----------------
an=pd.read_csv(D+'analgesic_predictions.csv')
OPIOID=['morphine','codeine','fentanyl','sufentanil','oxycodone','oxymorphone','hydrocodone',
        'hydromorphone','meperidine','etorphine','dihydro']
LA=['lidocaine','aminobenzoate','benzocaine']
def cls(n):
    l=str(n).lower()
    if any(t in l for t in LA): return 'Local anaesthetic'
    if any(t in l for t in OPIOID): return 'Opioid'
    return 'NSAID / non-opioid'
an['klass']=an.name.map(cls)
STY={'Opioid':(BLUE,'o'),'NSAID / non-opioid':(ORANGE,'s'),'Local anaesthetic':(AQUA,'^')}
pcol='pred_PG-form refit (logP, MW)'
fig,ax=plt.subplots(figsize=(4.4,4.0)); grid(ax)
lo=min(an.logkp.min(),an[pcol].min())-0.5; hi=max(an.logkp.max(),an[pcol].max())+0.5
ax.plot([lo,hi],[lo,hi],color=INK2,lw=0.8,ls='--',zorder=1)
for k,(c,m) in STY.items():
    s=an[an.klass==k]
    ax.scatter(s.logkp,s[pcol],s=42,marker=m,facecolor=c,edgecolor='white',
               linewidth=0.7,label='%s (n = %d)'%(k,len(s)),zorder=3)
for lab in ['Fentanyl','Sufentanil','Lidocaine','Morphine','Ibuprofen']:
    row=an[an.name.str.lower()==lab.lower()]
    if len(row): ax.annotate(lab,(row.logkp.iloc[0],row[pcol].iloc[0]),textcoords='offset points',
        xytext=(6,-2),fontsize=6.5,color=INK2)
r2,rm=stats(an.logkp,an[pcol])
ax.text(0.03,0.97,'$R^2$ = %.3f\nRMSE = %.3f\nn = %d'%(r2,rm,len(an)),transform=ax.transAxes,
        va='top',fontsize=7.5,bbox=dict(fc='white',ec=GRID,lw=0.5,boxstyle='round,pad=0.35'))
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
ax.set_xlabel('Observed log $K_p$ (cm s$^{-1}$)'); ax.set_ylabel('Predicted log $K_p$ (cm s$^{-1}$)')
ax.legend(loc='lower right',frameon=True,framealpha=1,edgecolor=GRID)
fig.savefig(F+'Fig3.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)

# ---------------- Fig 4: variance decomposition + layer stratification ----------------
vd=pd.read_csv(D+'variance_decomposition.csv'); ls=pd.read_csv(D+'layer_stratified.csv')
ls=ls[ls.R2.notna()]
fig,axs=plt.subplots(1,2,figsize=(6.6,3.0))
lbl=['Molecular\ndescriptors','Experimental\nprotocol','Both']
axs[0].bar(range(3),vd.R2,yerr=vd.R2_sd,width=0.6,color=[BLUE,ORANGE,AQUA],
           error_kw=dict(ecolor=INK2,lw=0.7,capsize=3))
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(lbl)
for i,v in enumerate(vd.R2): axs[0].text(i,v+vd.R2_sd.iloc[i]+0.012,'%.3f'%v,ha='center',fontsize=7,color=INK)
axs[0].set_ylabel('Cross-validated $R^2$'); axs[0].set_ylim(0,max(vd.R2+vd.R2_sd)+0.10)
NAMEMAP={'epidermis':'Epidermis only','other/mixed':'Mixed / unspecified'}
axs[1].bar(range(len(ls)),ls.R2,yerr=ls.R2_sd,width=0.5,color=[BLUE,ORANGE][:len(ls)],
           error_kw=dict(ecolor=INK2,lw=0.7,capsize=3))
axs[1].set_xticks(range(len(ls)))
axs[1].set_xticklabels(['%s\n(n = %d)'%(NAMEMAP.get(x,x),n) for x,n in zip(ls.layer,ls.n_compounds)])
for i,v in enumerate(ls.R2): axs[1].text(i,v+ls.R2_sd.iloc[i]+0.012,'%.3f'%v,ha='center',fontsize=7,color=INK)
axs[1].set_ylabel('Cross-validated $R^2$'); axs[1].set_ylim(0,max(ls.R2+ls.R2_sd)+0.10)
for ax in axs: grid(ax); ax.grid(axis='x',visible=False)
fig.savefig(F+'Fig4.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)

an.to_csv(D+'analgesic_predictions_classed.csv',index=False)
for f in sorted(os.listdir(F)):
    p=os.path.join(F,f); from PIL import Image
    im=Image.open(p); print('%-10s %s px  %.2f MB  mode=%s'%(f,im.size,os.path.getsize(p)/1e6,im.mode))
