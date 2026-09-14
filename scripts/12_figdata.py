"""Compute every panel input for the richer figures. Rendering happens locally."""
import math, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error
D='/workspace/icbmehs_c3_skin/data/'; O='/workspace/icbmehs_c3_skin/figdata/'
import os; os.makedirs(O,exist_ok=True)
SEED=20260821; LOG3600=math.log10(3600.0)
FEATS=['MW','logP','MR','TPSA','HBD','HBA','RotB','AromRings','Rings','HeavyAtoms',
       'FracCSP3','LabuteASA','BalabanJ','BertzCT','HeteroAtoms']
a=pd.read_csv(D+'compounds_modelling.csv').dropna(subset=FEATS+['logkp']).reset_index(drop=True)
itr=np.load(D+'split_train_idx.npy'); ite=np.load(D+'split_test_idx.npy')
X=a[FEATS].values; y=a['logkp'].values; analg=a['analgesic'].values.astype(bool)
pg_all=-2.72+0.71*a['logP'].values-0.0061*a['MW'].values-LOG3600

# ---------- Fig 1: external test predictions + residuals + AD ----------
rf=RandomForestRegressor(n_estimators=800,min_samples_leaf=2,random_state=SEED,n_jobs=8).fit(X[itr],y[itr])
Z=(X-X[itr].mean(0))/X[itr].std(0)
nn=NearestNeighbors(n_neighbors=6).fit(Z[itr])
d_tr=nn.kneighbors(Z[itr])[0][:,1:].mean(1); thr=float(np.percentile(d_tr,95))
d_te=nn.kneighbors(Z[ite])[0][:,:5].mean(1)
f1=pd.DataFrame({'name':a.loc[ite,'name'].values,'obs':y[ite],'rf':rf.predict(X[ite]),
                 'pg':pg_all[ite],'analgesic':analg[ite],'ad_dist':d_te,'in_ad':d_te<=thr})
f1['res_rf']=f1.obs-f1.rf; f1['res_pg']=f1.obs-f1.pg
f1.to_csv(O+'f1_external.csv',index=False)
json.dump({'ad_threshold':thr,'n_train':int(len(itr)),'n_test':int(len(ite)),
           'rf_r2':float(r2_score(f1.obs,f1.rf)),'rf_rmse':float(math.sqrt(mean_squared_error(f1.obs,f1.rf))),
           'pg_r2':float(r2_score(f1.obs,f1.pg)),'pg_rmse':float(math.sqrt(mean_squared_error(f1.obs,f1.pg)))},
          open(O+'f1_meta.json','w'),indent=1)
print('f1', f1.shape, 'AD threshold %.3f'%thr)

# ---------- Fig 2: importance + partial dependence ----------
imp=pd.read_csv(D+'descriptor_importance.csv'); imp.to_csv(O+'f2_importance.csv',index=False)
rows=[]
for feat in ['logP','MW']:
    j=FEATS.index(feat)
    pd_=partial_dependence(rf,X[itr],[j],grid_resolution=60,kind='average')
    for g,v in zip(pd_['grid_values'][0],pd_['average'][0]):
        rows.append({'descriptor':feat,'x':float(g),'yhat':float(v)})
pd.DataFrame(rows).to_csv(O+'f2_pd1d.csv',index=False)
jp,jm=FEATS.index('logP'),FEATS.index('MW')
pd2=partial_dependence(rf,X[itr],[(jp,jm)],grid_resolution=40,kind='average')
gx,gy=pd2['grid_values']; Zg=pd2['average'][0]
np.savez(O+'f2_pd2d.npz',logP=gx,MW=gy,z=Zg)
pd.DataFrame({'logP':a.logP,'MW':a.MW,'logkp':a.logkp,'analgesic':analg}).to_csv(O+'f2_points.csv',index=False)
print('f2 pd2d grid', Zg.shape)

# ---------- Fig 3: analgesic leave-analgesics-out, per drug ----------
an=pd.read_csv(D+'analgesic_predictions_classed.csv')
an[['name','klass','logkp','n_rec','logkp_min','logkp_max','ad_dist','in_ad',
    'pred_PG-form refit (logP, MW)','pred_Random forest','pred_PottsGuy','MW','logP']].to_csv(O+'f3_analgesics.csv',index=False)
print('f3', an.shape)

# ---------- Fig 4: variance decomposition, layer strata, reproducibility ----------
pd.read_csv(D+'variance_decomposition.csv').to_csv(O+'f4_variance.csv',index=False)
pd.read_csv(D+'layer_stratified_rkf.csv').to_csv(O+'f4_layers.csv',index=False)
r=pd.read_csv(D+'records_with_descriptors.csv')
g=r.groupby('csmiles').agg(nref=('ref','nunique'),lo=('logkp','min'),hi=('logkp','max'),
                           n=('logkp','size'),name=('name','first')).reset_index()
s=g[g.nref>1].copy(); s['spread']=s.hi-s.lo
sd=r.groupby('csmiles').apply(lambda d: d.groupby('ref')['logkp'].median().std()).rename('sd_between')
s=s.merge(sd.reset_index(),on='csmiles',how='left')
s[['name','nref','n','spread','sd_between']].to_csv(O+'f4_repro.csv',index=False)
print('f4 repro n=%d  median spread %.2f  median sd %.2f'%(len(s),s.spread.median(),s.sd_between.median()))
# layer composition for a stacked panel
def lump(x):
    x=str(x).lower().strip()
    if 'stratum corneum' in x and 'without' not in x: return 'Stratum corneum'
    if x.startswith('epidermis') and 'without' not in x and ',' not in x: return 'Epidermis'
    if x=='dermis': return 'Dermis'
    return 'Mixed / unspecified'
r['layer_g']=r['layer'].map(lump)
r.groupby('layer_g').agg(records=('logkp','size'),compounds=('csmiles','nunique'),
    median=('logkp','median'),q1=('logkp',lambda v:v.quantile(.25)),
    q3=('logkp',lambda v:v.quantile(.75))).reset_index().to_csv(O+'f4_layer_desc.csv',index=False)
print('wrote figdata to',O)
