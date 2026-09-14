"""Is the low R2 chemistry or protocol? Layer stratification, covariate model, logP source."""
import math, warnings, re
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
D='/workspace/icbmehs_c3_skin/data/'; SEED=20260821; LOG3600=math.log10(3600.0)
FEATS=['MW','logP','MR','TPSA','HBD','HBA','RotB','AromRings','Rings','HeavyAtoms',
       'FracCSP3','LabuteASA','BalabanJ','BertzCT','HeteroAtoms']
r=pd.read_csv(D+'records_with_descriptors.csv')
r=r.dropna(subset=FEATS+['logkp']).reset_index(drop=True)
print('records: %d / compounds: %d'%(len(r),r.csmiles.nunique()))

def cvscore(X,y,g,model,k=10):
    gk=GroupKFold(n_splits=k); r2s=[];rms=[]
    for tr,va in gk.split(X,y,groups=g):
        m=model(); m.fit(X.iloc[tr] if hasattr(X,'iloc') else X[tr], y[tr])
        p=m.predict(X.iloc[va] if hasattr(X,'iloc') else X[va])
        r2s.append(r2_score(y[va],p)); rms.append(math.sqrt(mean_squared_error(y[va],p)))
    return np.mean(r2s),np.std(r2s),np.mean(rms)

# ---------- (a) how much variance is protocol? within-compound spread by layer ----------
print('\n=== (a) LAYER STRATIFICATION (compound-level medians within each layer) ===')
r['layer_c']=r['layer'].astype(str).str.lower().str.strip()
def lump(s):
    if 'stratum corneum' in s and 'without' not in s: return 'stratum corneum'
    if s.startswith('epidermis') and 'without' not in s and ',' not in s: return 'epidermis'
    if s=='dermis': return 'dermis'
    return 'other/mixed'
r['layer_g']=r['layer_c'].map(lump)
print(r.groupby('layer_g').agg(records=('logkp','size'),compounds=('csmiles','nunique'),
      median_logkp=('logkp','median')).round(2).to_string())
rows=[]
for lg,sub in r.groupby('layer_g'):
    a=sub.groupby('csmiles').agg(logkp=('logkp','median'),**{f:(f,'first') for f in FEATS}).reset_index()
    if len(a)<40: rows.append((lg,len(a),np.nan,np.nan,np.nan)); continue
    y=a['logkp'].values; X=a[FEATS].values; g=a['csmiles'].values
    m,sd,rm=cvscore(pd.DataFrame(X,columns=FEATS),y,g,
        lambda: RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=SEED,n_jobs=8))
    rows.append((lg,len(a),m,sd,rm))
print('\nRF, 10-fold grouped CV, within each layer:')
for lg,n,m,sd,rm in rows:
    print('  %-16s n=%3d  '%(lg,n) + ('R2=%6.3f +/- %.3f  RMSE=%.3f'%(m,sd,rm) if n>=40 else '(too few compounds)'))

# ---------- (b) record-level: chemistry only vs chemistry + protocol ----------
print('\n=== (b) RECORD-LEVEL variance decomposition (grouped CV by compound, no leakage) ===')
CAT=['layer_g','site','cell_type']; NUM=['donor_temp','donor_ph','acceptor_temp','acceptor_ph']
for c in CAT: r[c]=r[c].astype(str).str.lower().str.strip().replace({'nan':'unknown'})
y=r['logkp'].values; g=r['csmiles'].values
def mk(cols_num,cols_cat):
    def f():
        t=[]
        if cols_num: t.append(('n',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),cols_num))
        if cols_cat: t.append(('c',make_pipeline(SimpleImputer(strategy='constant',fill_value='unknown'),
                                OneHotEncoder(handle_unknown='ignore',min_frequency=5)),cols_cat))
        return make_pipeline(ColumnTransformer(t),
                RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=SEED,n_jobs=8))
    return f
SPECS=[('chemistry only (15 descriptors)',FEATS,[]),
       ('protocol only',NUM,CAT),
       ('chemistry + protocol',FEATS+NUM,CAT)]
res={}
for nm,cn,cc in SPECS:
    m,sd,rm=cvscore(r[cn+cc],y,g,mk(cn,cc))
    res[nm]=(m,sd,rm); print('  %-34s R2 = %6.3f +/- %.3f   RMSE = %.3f'%(nm,m,sd,rm))

# ---------- (c) computed Crippen logP vs experimental logKow (Cheruvu 2022) ----------
print('\n=== (c) COMPUTED vs EXPERIMENTAL logP in the Potts-Guy equation ===')
ch=pd.read_excel(D+'cheruvu2022_Table1.xlsx')
ch['nm']=ch['Compound'].astype(str).str.lower().str.strip()
exp=ch.groupby('nm').agg(logKow_exp=('logKowb','median'),MW_exp=('MWa','median')).reset_index()
a=pd.read_csv(D+'compounds_modelling.csv'); a['nm']=a['name'].str.lower().str.strip()
mg=a.merge(exp,on='nm',how='inner').dropna(subset=['logKow_exp'])
print('compounds with an experimental logKow match by name: %d'%len(mg))
print('  Crippen logP vs experimental logKow: Pearson r = %.3f, mean abs diff = %.2f'%(
    np.corrcoef(mg.logP,mg.logKow_exp)[0,1], (mg.logP-mg.logKow_exp).abs().mean()))
for lab,lp in [('computed Crippen logP',mg.logP.values),('experimental logKow',mg.logKow_exp.values)]:
    pg=-2.72+0.71*lp-0.0061*mg.MW.values-LOG3600
    print('  Potts-Guy with %-24s R2 = %6.3f  RMSE = %.3f'%(lab,r2_score(mg.logkp,pg),math.sqrt(mean_squared_error(mg.logkp,pg))))
pd.DataFrame(rows,columns=['layer','n_compounds','R2','R2_sd','RMSE']).to_csv(D+'layer_stratified.csv',index=False)
pd.DataFrame([(k,)+v for k,v in res.items()],columns=['model','R2','R2_sd','RMSE']).to_csv(D+'variance_decomposition.csv',index=False)
mg[['name','logkp','logP','logKow_exp','MW']].to_csv(D+'logp_comparison.csv',index=False)
print('\nwrote layer_stratified.csv, variance_decomposition.csv, logp_comparison.csv')
