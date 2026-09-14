"""Formal comparisons with exact p-values (journal requires exact p)."""
import math, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import GroupKFold, RepeatedKFold
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
def pf(p): return ('%.3g'%p) if p>=1e-4 else ('%.2e'%p)
out={}

# --- 1. external test: RF vs Potts-Guy(literature), paired absolute errors ---
t=pd.read_csv(D+'external_test_predictions.csv'); a=pd.read_csv(D+'compounds_modelling.csv')
ite=np.load(D+'split_test_idx.npy')
pg=(-2.72+0.71*a['logP'].values-0.0061*a['MW'].values-LOG3600)[ite]
e_rf=np.abs(t['obs'].values-t['pred_Random forest'].values); e_pg=np.abs(t['obs'].values-pg)
w=stats.wilcoxon(e_rf,e_pg); out['rf_vs_pg_ext']=dict(n=len(e_rf),med_rf=float(np.median(e_rf)),
    med_pg=float(np.median(e_pg)),W=float(w.statistic),p=float(w.pvalue))
print('1. External test (n=%d): |error| RF median %.3f vs Potts-Guy %.3f; Wilcoxon W=%.1f, p=%s'%(
    len(e_rf),np.median(e_rf),np.median(e_pg),w.statistic,pf(w.pvalue)))

# --- 2. analgesic extrapolation: PG-form refit vs random forest ---
an=pd.read_csv(D+'analgesic_predictions_classed.csv')
e_lin=np.abs(an.logkp-an['pred_PG-form refit (logP, MW)']); e_rf2=np.abs(an.logkp-an['pred_Random forest'])
w2=stats.wilcoxon(e_lin,e_rf2); out['lin_vs_rf_analgesic']=dict(n=len(an),med_lin=float(np.median(e_lin)),
    med_rf=float(np.median(e_rf2)),W=float(w2.statistic),p=float(w2.pvalue))
print('2. Analgesics (n=%d): |error| two-descriptor linear median %.3f vs random forest %.3f; Wilcoxon W=%.1f, p=%s'%(
    len(an),np.median(e_lin),np.median(e_rf2),w2.statistic,pf(w2.pvalue)))
# and vs Potts-Guy literature
w2b=stats.wilcoxon(e_lin,np.abs(an.logkp-an.pred_PottsGuy))
print('   linear vs Potts-Guy(literature): median %.3f vs %.3f; W=%.1f, p=%s'%(
    np.median(e_lin),np.median(np.abs(an.logkp-an.pred_PottsGuy)),w2b.statistic,pf(w2b.pvalue)))
out['lin_vs_pglit_analgesic']=dict(W=float(w2b.statistic),p=float(w2b.pvalue))

# --- 3. fold-wise chemistry vs protocol vs both (paired over identical folds) ---
r=pd.read_csv(D+'records_with_descriptors.csv').dropna(subset=FEATS+['logkp']).reset_index(drop=True)
CAT=['layer_g','site','cell_type']; NUM=['donor_temp','donor_ph','acceptor_temp','acceptor_ph']
def lump(s):
    s=str(s).lower().strip()
    if 'stratum corneum' in s and 'without' not in s: return 'stratum corneum'
    if s.startswith('epidermis') and 'without' not in s and ',' not in s: return 'epidermis'
    if s=='dermis': return 'dermis'
    return 'other/mixed'
r['layer_g']=r['layer'].map(lump)
for c in ['site','cell_type']: r[c]=r[c].astype(str).str.lower().str.strip().replace({'nan':'unknown'})
y=r['logkp'].values; g=r['csmiles'].values
def mk(cn,cc):
    t=[]
    if cn: t.append(('n',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),cn))
    if cc: t.append(('c',make_pipeline(SimpleImputer(strategy='constant',fill_value='unknown'),
                     OneHotEncoder(handle_unknown='ignore',min_frequency=5)),cc))
    return make_pipeline(ColumnTransformer(t),RandomForestRegressor(n_estimators=600,min_samples_leaf=2,random_state=SEED,n_jobs=8))
SPECS={'chem':(FEATS,[]),'proto':(NUM,CAT),'both':(FEATS+NUM,CAT)}
folds=list(GroupKFold(n_splits=10).split(r,y,groups=g)); scores={k:[] for k in SPECS}
for tr,va in folds:
    for k,(cn,cc) in SPECS.items():
        m=mk(cn,cc); m.fit(r.loc[tr,cn+cc],y[tr]); scores[k].append(r2_score(y[va],m.predict(r.loc[va,cn+cc])))
for k in scores: scores[k]=np.array(scores[k])
for A,B in [('proto','chem'),('both','chem'),('both','proto')]:
    w3=stats.wilcoxon(scores[A],scores[B])
    print('3. %s vs %s across 10 identical folds: mean R2 %.3f vs %.3f; Wilcoxon W=%.1f, p=%s'%(
        A,B,scores[A].mean(),scores[B].mean(),w3.statistic,pf(w3.pvalue)))
    out['fold_%s_vs_%s'%(A,B)]=dict(meanA=float(scores[A].mean()),meanB=float(scores[B].mean()),
        W=float(w3.statistic),p=float(w3.pvalue))

# --- 4. epidermis vs mixed: paired over repeated CV within each stratum (unpaired test) ---
res={}
for lg in ['epidermis','other/mixed']:
    sub=r[r.layer_g==lg]
    ag=sub.groupby('csmiles').agg(logkp=('logkp','median'),**{f:(f,'first') for f in FEATS}).reset_index()
    rk=RepeatedKFold(n_splits=10,n_repeats=5,random_state=SEED); s=[]
    for tr,va in rk.split(ag):
        m=RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=8)
        m.fit(ag.loc[tr,FEATS],ag.loc[tr,'logkp']); s.append(r2_score(ag.loc[va,'logkp'],m.predict(ag.loc[va,FEATS])))
    res[lg]=np.array(s)
u=stats.mannwhitneyu(res['epidermis'],res['other/mixed'],alternative='two-sided')
print('4. Epidermis-only vs mixed-layer, 50 CV folds each: mean R2 %.3f vs %.3f; Mann-Whitney U=%.1f, p=%s'%(
    res['epidermis'].mean(),res['other/mixed'].mean(),u.statistic,pf(u.pvalue)))
out['layer_epi_vs_mixed']=dict(mean_epi=float(res['epidermis'].mean()),mean_mixed=float(res['other/mixed'].mean()),
    U=float(u.statistic),p=float(u.pvalue))

# --- 5. computed vs experimental logP as Potts-Guy input ---
lp=pd.read_csv(D+'logp_comparison.csv')
e_c=np.abs(lp.logkp-(-2.72+0.71*lp.logP-0.0061*lp.MW-LOG3600))
e_e=np.abs(lp.logkp-(-2.72+0.71*lp.logKow_exp-0.0061*lp.MW-LOG3600))
w5=stats.wilcoxon(e_c,e_e); rp=stats.pearsonr(lp.logP,lp.logKow_exp)
print('5. Potts-Guy input (n=%d): |error| computed logP median %.3f vs experimental logKow %.3f; W=%.1f, p=%s'%(
    len(lp),np.median(e_c),np.median(e_e),w5.statistic,pf(w5.pvalue)))
print('   Crippen logP vs experimental logKow: r=%.3f, p=%s'%(rp[0],pf(rp[1])))
out['logp_source']=dict(n=int(len(lp)),W=float(w5.statistic),p=float(w5.pvalue),r=float(rp[0]),r_p=float(rp[1]))
json.dump(out,open(D+'stats_tests.json','w'),indent=1)
print('\nwrote stats_tests.json')
