import math, warnings, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
D='/workspace/icbmehs_c3_skin/data/'; F='/workspace/icbmehs_c3_skin/figs/'
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'; INK,INK2,GRID='#0b0b0b','#52514e','#d8d7d2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.edgecolor':INK2,
 'axes.linewidth':0.7,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,
 'axes.labelcolor':INK,'figure.dpi':300,'savefig.dpi':300,'axes.spines.top':False,
 'axes.spines.right':False,'savefig.bbox':'tight','savefig.pad_inches':0.02})
an=pd.read_csv(D+'analgesic_predictions_classed.csv')
pcol='pred_PG-form refit (logP, MW)'
STY={'Opioid':(BLUE,'o'),'NSAID / non-opioid':(ORANGE,'s'),'Local anaesthetic':(AQUA,'^')}
fig,ax=plt.subplots(figsize=(4.6,4.2))
ax.grid(True,color=GRID,linewidth=0.5); ax.set_axisbelow(True)
lo=min(an.logkp.min(),an[pcol].min())-0.5; hi=max(an.logkp.max(),an[pcol].max())+1.1
ax.plot([lo,hi],[lo,hi],color=INK2,lw=0.8,ls='--',zorder=1)
for k,(c,m) in STY.items():
    s=an[an.klass==k]
    ax.scatter(s.logkp,s[pcol],s=42,marker=m,facecolor=c,edgecolor='white',
               linewidth=0.7,label='%s (n = %d)'%(k,len(s)),zorder=3)
OFF={'Fentanyl':((0,-14),'center'),'Sufentanil':((10,-3),'left'),
     'Lidocaine':((-10,6),'right'),'Morphine':((10,-3),'left'),'Ibuprofen':((0,11),'center')}
for lab,(xy,ha) in OFF.items():
    row=an[an.name.str.lower()==lab.lower()]
    if len(row):
        ax.annotate(lab,(row.logkp.iloc[0],row[pcol].iloc[0]),textcoords='offset points',
                    xytext=xy,ha=ha,fontsize=6.5,color=INK2,zorder=5)
r2=r2_score(an.logkp,an[pcol]); rm=math.sqrt(mean_squared_error(an.logkp,an[pcol]))
ax.text(0.03,0.97,'$R^2$ = %.3f\nRMSE = %.3f\nn = %d'%(r2,rm,len(an)),transform=ax.transAxes,
        va='top',fontsize=7.5,bbox=dict(fc='white',ec=GRID,lw=0.5,boxstyle='round,pad=0.35'),zorder=5)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
ax.set_xlabel('Observed log $K_p$ (cm s$^{-1}$)'); ax.set_ylabel('Predicted log $K_p$ (cm s$^{-1}$)')
ax.legend(loc='lower right',frameon=True,framealpha=1,edgecolor=GRID)
fig.savefig(F+'Fig3.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)
from PIL import Image
im=Image.open(F+'Fig3.tiff')
if im.mode!='RGB':
    bg=Image.new('RGB',im.size,(255,255,255)); bg.paste(im,mask=im.split()[-1])
    bg.save(F+'Fig3.tiff',format='TIFF',compression='tiff_lzw',dpi=(300,300)); im=Image.open(F+'Fig3.tiff')
pv=im.copy(); pv.thumbnail((900,900)); pv.save(F+'prev_Fig3.png',optimize=True)
print('Fig3',im.size,im.mode,'%.2f MB'%(os.path.getsize(F+'Fig3.tiff')/1e6))
