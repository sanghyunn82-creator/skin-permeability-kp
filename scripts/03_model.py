"""QSPR models for human skin permeability, with analgesics as an external stratum."""
import json, math, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_predict
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

D='/workspace/icbmehs_c3_skin/data/'; SEED=20260821
LOG3600=math.log10(3600.0)
a=pd.read_csv(D+'compounds_modelling.csv')
FEATS=['MW','logP','MR','TPSA','HBD','HBA','RotB','AromRings','Rings','HeavyAtoms',
       'FracCSP3','LabuteASA','BalabanJ','BertzCT','HeteroAtoms']
a=a.dropna(subset=FEATS+['logkp']).reset_index(drop=True)
X=a[FEATS].values; y=a['logkp'].values; analg=a['analgesic'].values.astype(bool)
print('compounds: %d (analgesics %d)  features: %d'%(len(a),analg.sum(),len(FEATS)))

def sc(yt,yp): return dict(R2=r2_score(yt,yp), RMSE=math.sqrt(mean_squared_error(yt,yp)), MAE=mean_absolute_error(yt,yp), n=len(yt))
def line(tag,m): print('  %-34s n=%3d  R2=%6.3f  RMSE=%5.3f  MAE=%5.3f'%(tag,m['n'],m['R2'],m['RMSE'],m['MAE']))

# ---------- benchmark: Potts & Guy (1992) with published coefficients ----------
# logKp(cm/h) = -2.72 + 0.71*logKow - 0.0061*MW  ->  convert to cm/s
pg = -2.72 + 0.71*a['logP'].values - 0.0061*a['MW'].values - LOG3600
print('\n=== BENCHMARK: Potts & Guy (1992) published coefficients, applied to all %d compounds ==='%len(a))
line('Potts-Guy (literature)', sc(y,pg))
line('  -> analgesic subset', sc(y[analg],pg[analg]))

# ---------- split ----------
strat = pd.cut(y,bins=[-np.inf,-7.5,-6.5,-5.5,np.inf],labels=False).astype(str)+np.where(analg,'A','N')
vc=pd.Series(strat).value_counts(); strat=np.where(pd.Series(strat).map(vc).values<2,'rare',strat)
itr,ite=train_test_split(np.arange(len(a)),test_size=0.20,random_state=SEED,stratify=strat)
print('\nexternal split: train %d / test %d (test analgesics %d)'%(len(itr),len(ite),analg[ite].sum()))

MODELS={
 'PG-form refit (logP, MW)': (make_pipeline(StandardScaler(),LinearRegression()), ['logP','MW']),
 'MLR (all descriptors)':    (make_pipeline(StandardScaler(),LinearRegression()), FEATS),
 'Ridge (all descriptors)':  (make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,25))), FEATS),
 'Random forest':            (RandomForestRegressor(n_estimators=800,min_samples_leaf=2,random_state=SEED,n_jobs=8), FEATS),
 'Gradient boosting (XGB)':  (XGBRegressor(n_estimators=700,learning_rate=0.03,max_depth=4,subsample=0.8,
                                colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,n_jobs=8), FEATS),
}
cv=RepeatedKFold(n_splits=10,n_repeats=5,random_state=SEED)
results={}
print('\n=== 10-fold x 5-repeat cross-validation on training set (n=%d) ==='%len(itr))
for nm,(mdl,fs) in MODELS.items():
    Xf=a[fs].values
    yp=cross_val_predict(mdl,Xf[itr],y[itr],cv=cv.split(Xf[itr]) if False else 10,n_jobs=4)
    # repeated CV: average metrics across repeats
    r2s=[];rms=[]
    for tr,va in cv.split(Xf[itr]):
        m=mdl.__class__(**mdl.get_params()) if not hasattr(mdl,'steps') else make_pipeline(*[s[1].__class__(**s[1].get_params()) for s in mdl.steps])
        m.fit(Xf[itr][tr],y[itr][tr]); p=m.predict(Xf[itr][va])
        r2s.append(r2_score(y[itr][va],p)); rms.append(math.sqrt(mean_squared_error(y[itr][va],p)))
    results[nm]={'cv_R2':np.mean(r2s),'cv_R2_sd':np.std(r2s),'cv_RMSE':np.mean(rms),'cv_RMSE_sd':np.std(rms)}
    print('  %-26s CV R2 = %.3f +/- %.3f   CV RMSE = %.3f +/- %.3f'%(nm,np.mean(r2s),np.std(r2s),np.mean(rms),np.std(rms)))

print('\n=== external test set (n=%d), models refitted on full training set ==='%len(ite))
preds={}
for nm,(mdl,fs) in MODELS.items():
    Xf=a[fs].values; mdl.fit(Xf[itr],y[itr]); p=mdl.predict(Xf[ite]); preds[nm]=p
    results[nm].update({('ext_'+k):v for k,v in sc(y[ite],p).items()})
    line(nm,sc(y[ite],p))
line('Potts-Guy (literature) on same test', sc(y[ite],pg[ite]))

# ---------- leave-analgesics-out external validation ----------
print('\n=== LEAVE-ANALGESICS-OUT: train on %d non-analgesics, predict %d analgesics ==='%((~analg).sum(),analg.sum()))
lao={}
for nm,(mdl,fs) in MODELS.items():
    Xf=a[fs].values; mdl.fit(Xf[~analg],y[~analg]); p=mdl.predict(Xf[analg])
    lao[nm]=p; m=sc(y[analg],p); results[nm].update({('lao_'+k):v for k,v in m.items()}); line(nm,m)
line('Potts-Guy (literature)', sc(y[analg],pg[analg]))

best=min(results,key=lambda k:results[k]['lao_RMSE'])
print('\nbest by leave-analgesics-out RMSE: %s'%best)

# ---------- descriptor importance (RF, permutation on external test) ----------
rf=RandomForestRegressor(n_estimators=800,min_samples_leaf=2,random_state=SEED,n_jobs=8).fit(X[itr],y[itr])
pi=permutation_importance(rf,X[ite],y[ite],n_repeats=50,random_state=SEED,n_jobs=8)
imp=pd.DataFrame({'descriptor':FEATS,'imp_mean':pi.importances_mean,'imp_sd':pi.importances_std}).sort_values('imp_mean',ascending=False)
print('\n=== permutation importance (RF, external test) ===')
print(imp.head(10).to_string(index=False))

# ---------- applicability domain: kNN distance in standardised descriptor space ----------
Z=(X-X[itr].mean(0))/X[itr].std(0)
nn=NearestNeighbors(n_neighbors=6).fit(Z[itr])
dtr=nn.kneighbors(Z[itr])[0][:,1:].mean(1); thr=np.percentile(dtr,95)
dan=nn.kneighbors(Z[analg])[0][:,:5].mean(1)
print('\n=== applicability domain (mean distance to 5 nearest training compounds) ===')
print('training 95th percentile threshold = %.3f'%thr)
print('analgesics inside AD: %d / %d (%.0f%%)'%((dan<=thr).sum(),analg.sum(),100*(dan<=thr).mean()))
out=a.loc[analg].copy(); out['ad_dist']=dan; out['in_ad']=dan<=thr
for nm,p in lao.items(): out['pred_'+nm]=p
out['pred_PottsGuy']=pg[analg]
out.to_csv(D+'analgesic_predictions.csv',index=False)
pd.DataFrame(results).T.to_csv(D+'model_results.csv')
imp.to_csv(D+'descriptor_importance.csv',index=False)
np.save(D+'split_test_idx.npy',ite); np.save(D+'split_train_idx.npy',itr)
pd.DataFrame({'obs':y[ite],**{('pred_'+k):v for k,v in preds.items()},'analgesic':analg[ite],
              'name':a.loc[ite,'name'].values}).to_csv(D+'external_test_predictions.csv',index=False)
print('\nwrote model_results.csv, analgesic_predictions.csv, descriptor_importance.csv, external_test_predictions.csv')
