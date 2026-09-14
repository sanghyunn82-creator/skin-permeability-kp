import math,warnings,json,os
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
D='/workspace/icbmehs_c3_skin/data/'; F='/workspace/icbmehs_c3_skin/figs/'; SEED=20260821
FEATS=['MW','logP','MR','TPSA','HBD','HBA','RotB','AromRings','Rings','HeavyAtoms',
       'FracCSP3','LabuteASA','BalabanJ','BertzCT','HeteroAtoms']
r=pd.read_csv(D+'records_with_descriptors.csv').dropna(subset=FEATS+['logkp'])
# (a) post-standardisation database overlap
hs=set(r.loc[r.src=='HuskinDB','csmiles']); ss=set(r.loc[r.src=='SkinPiX','csmiles'])
print('POST-STANDARDISATION: HuskinDB %d, SkinPiX %d, shared %d, SkinPiX-only %d, total %d'%(
    len(hs),len(ss),len(hs&ss),len(ss-hs),len(hs|ss)))
# (b) layer strata with the SAME repeated-KFold scheme used for the significance test
def lump(s):
    s=str(s).lower().strip()
    if 'stratum corneum' in s and 'without' not in s: return 'stratum corneum'
    if s.startswith('epidermis') and 'without' not in s and ',' not in s: return 'epidermis'
    if s=='dermis': return 'dermis'
    return 'other/mixed'
r['layer_g']=r['layer'].map(lump)
res={}
for lg in ['epidermis','other/mixed']:
    sub=r[r.layer_g==lg]
    ag=sub.groupby('csmiles').agg(logkp=('logkp','median'),**{f:(f,'first') for f in FEATS}).reset_index()
    rk=RepeatedKFold(n_splits=10,n_repeats=5,random_state=SEED); s=[]
    for tr,va in rk.split(ag):
        m=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=8)
        m.fit(ag.loc[tr,FEATS],ag.loc[tr,'logkp']); s.append(r2_score(ag.loc[va,'logkp'],m.predict(ag.loc[va,FEATS])))
    res[lg]=(len(ag),float(np.mean(s)),float(np.std(s)))
    print('%-14s n=%d  R2=%.3f +/- %.3f'%(lg,len(ag),np.mean(s),np.std(s)))
pd.DataFrame([(k,)+v for k,v in res.items()],columns=['layer','n_compounds','R2','R2_sd']).to_csv(D+'layer_stratified_rkf.csv',index=False)

# (c) redraw Fig 4 with matching numbers
BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'; INK,INK2,GRID='#0b0b0b','#52514e','#d8d7d2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'axes.edgecolor':INK2,'axes.linewidth':0.7,
 'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'axes.labelcolor':INK,
 'figure.dpi':300,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
 'savefig.bbox':'tight','savefig.pad_inches':0.02})
vd=pd.read_csv(D+'variance_decomposition.csv')
fig,axs=plt.subplots(1,2,figsize=(6.6,3.0))
axs[0].bar(range(3),vd.R2,yerr=vd.R2_sd,width=0.6,color=[BLUE,ORANGE,AQUA],
           error_kw=dict(ecolor=INK2,lw=0.7,capsize=3))
axs[0].set_xticks(range(3)); axs[0].set_xticklabels(['Molecular\ndescriptors','Experimental\nprotocol','Both'])
for i,v in enumerate(vd.R2): axs[0].text(i,v+vd.R2_sd.iloc[i]+0.012,'%.3f'%v,ha='center',fontsize=7,color=INK)
axs[0].set_ylabel('Cross-validated $R^2$'); axs[0].set_ylim(0,max(vd.R2+vd.R2_sd)+0.10)
ks=['epidermis','other/mixed']; lab=['Epidermis only','Mixed / unspecified']
vals=[res[k][1] for k in ks]; sds=[res[k][2] for k in ks]; ns=[res[k][0] for k in ks]
axs[1].bar(range(2),vals,yerr=sds,width=0.5,color=[BLUE,ORANGE],error_kw=dict(ecolor=INK2,lw=0.7,capsize=3))
axs[1].set_xticks(range(2)); axs[1].set_xticklabels(['%s\n(n = %d)'%(l,n) for l,n in zip(lab,ns)])
for i,v in enumerate(vals): axs[1].text(i,v+sds[i]+0.012,'%.3f'%v,ha='center',fontsize=7,color=INK)
axs[1].set_ylabel('Cross-validated $R^2$'); axs[1].set_ylim(0,max(np.array(vals)+np.array(sds))+0.10)
for ax in axs: ax.grid(True,color=GRID,linewidth=0.5); ax.set_axisbelow(True); ax.grid(axis='x',visible=False)
fig.savefig(F+'Fig4.tiff',format='tiff',pil_kwargs={'compression':'tiff_lzw'}); plt.close(fig)
from PIL import Image
im=Image.open(F+'Fig4.tiff')
if im.mode!='RGB':
    bg=Image.new('RGB',im.size,(255,255,255)); bg.paste(im,mask=im.split()[-1])
    bg.save(F+'Fig4.tiff',format='TIFF',compression='tiff_lzw',dpi=(300,300)); im=Image.open(F+'Fig4.tiff')
pv=im.copy(); pv.thumbnail((900,900)); pv.save(F+'prev_Fig4.png',optimize=True)
print('Fig4 redrawn',im.size,im.mode)
