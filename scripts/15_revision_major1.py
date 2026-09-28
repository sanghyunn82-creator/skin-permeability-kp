"""Reviewer 2, Major 1: does the median-per-compound reduction drive the results?

The reviewer's point is that collapsing each compound to a median log Kp throws
away the very between-study spread the paper's ceiling argument rests on, and
weights a 19-record compound the same as a 1-record compound (the pooled set
has 177 single-record compounds and a maximum of 19).

Four schemes are run side by side on ONE compound-level split, so the held-out
compounds are identical in every scheme and the only thing that changes is how
replicate records are handled:

  A  median   -- one row per compound, log Kp = median      (the published baseline)
  B  mean     -- one row per compound, log Kp = mean        (requested sensitivity)
  C  record   -- all 732 records, grouped CV by compound    (requested record-level)
  D  mixed    -- records with compound as a random intercept (requested mixed model)

Leakage control: every fold and the held-out split are defined on compounds,
never on records, so all records of a compound always travel together. This is
asserted, not assumed -- see check_no_leak().

Nothing here overwrites the published median-based outputs; everything is
written to revision/major1/.
"""
import json, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from xgboost import XGBRegressor
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = '/workspace/icbmehs_c3_skin/'
D = ROOT + 'data/'
OUT = ROOT + 'revision/major1/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260821
LOG3600 = math.log10(3600.0)
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']

# ---------------------------------------------------------------- data
rec = pd.read_csv(D + 'records_with_descriptors.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
cmp_med = pd.read_csv(D + 'compounds_modelling.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)

# the published split, defined on compounds -- reused verbatim by every scheme
itr = np.load(D + 'split_train_idx.npy')
ite = np.load(D + 'split_test_idx.npy')
train_cmp = set(cmp_med.loc[itr, 'csmiles'])
test_cmp = set(cmp_med.loc[ite, 'csmiles'])
assert not (train_cmp & test_cmp)
print('compounds: %d train / %d held-out   records: %d' % (len(train_cmp), len(test_cmp), len(rec)))

# mean-aggregated compound table, built from the same records
agg_mean = (rec.groupby('csmiles')
              .agg(logkp=('logkp', 'mean'), n_rec=('logkp', 'size'),
                   **{f: (f, 'first') for f in FEATS}).reset_index())
cmp_mean = cmp_med[['csmiles', 'name', 'analgesic']].merge(agg_mean, on='csmiles', how='inner')
cmp_mean = cmp_mean[cmp_med.columns.intersection(cmp_mean.columns).tolist() +
                    [c for c in cmp_mean.columns if c not in cmp_med.columns]]
assert len(cmp_mean) == len(cmp_med)
d = (cmp_mean.set_index('csmiles')['logkp'] - cmp_med.set_index('csmiles')['logkp']).abs()
print('mean vs median per compound: %d differ, max |diff| = %.3f log units' % ((d > 1e-9).sum(), d.max()))

# the 98-compound experimental-logKow subset used for the log P sensitivity analysis
lp = pd.read_csv(D + 'logp_comparison.csv')
lp_names = set(lp['name'])
print('experimental-logKow subset: %d compounds' % len(lp_names))

# ---------------------------------------------------------------- models
def models():
    return {
        'PG-form refit (logP, MW)': (make_pipeline(StandardScaler(), LinearRegression()), ['logP', 'MW']),
        'MLR (all descriptors)':    (make_pipeline(StandardScaler(), LinearRegression()), FEATS),
        'Ridge (all descriptors)':  (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))), FEATS),
        'Random forest':            (RandomForestRegressor(n_estimators=800, min_samples_leaf=2,
                                                           random_state=SEED, n_jobs=8), FEATS),
        'Gradient boosting (XGB)':  (XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=4,
                                                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                                  random_state=SEED, n_jobs=8), FEATS),
    }

def check_no_leak(groups, folds, tag):
    """A compound must never appear on both sides of any split."""
    for k, (tr, va) in enumerate(folds):
        if set(groups[tr]) & set(groups[va]):
            raise SystemExit('LEAK in %s fold %d' % (tag, k))
    return True

def grouped_cv_scores(df, y, groups, n_splits=10):
    """Grouped CV over whichever unit df carries; returns per-model fold scores."""
    gk = GroupKFold(n_splits=n_splits)
    folds = list(gk.split(df, y, groups=groups))
    check_no_leak(groups, folds, 'grouped CV')
    out = {}
    for nm, (mdl, fs) in models().items():
        X = df[fs].values
        r2s, rmses = [], []
        for tr, va in folds:
            m = clone(mdl).fit(X[tr], y[tr])
            p = m.predict(X[va])
            r2s.append(r2_score(y[va], p))
            rmses.append(math.sqrt(mean_squared_error(y[va], p)))
        out[nm] = dict(cv_R2=float(np.mean(r2s)), cv_R2_sd=float(np.std(r2s)),
                       cv_RMSE=float(np.mean(rmses)))
    return out

def fit_predict_heldout(df_tr, y_tr, df_te, y_te):
    out, preds = {}, {}
    for nm, (mdl, fs) in models().items():
        m = clone(mdl).fit(df_tr[fs].values, y_tr)
        p = m.predict(df_te[fs].values)
        preds[nm] = p
        out[nm] = dict(ho_R2=r2_score(y_te, p),
                       ho_RMSE=math.sqrt(mean_squared_error(y_te, p)),
                       ho_MAE=mean_absolute_error(y_te, p), ho_n=len(y_te))
    return out, preds

def pg_literature(df):
    return -2.72 + 0.71 * df['logP'].values - 0.0061 * df['MW'].values - LOG3600

# ---------------------------------------------------------------- schemes A/B
results = {}
for tag, tbl in [('A_median', cmp_med), ('B_mean', cmp_mean)]:
    t = tbl.reset_index(drop=True)
    y = t['logkp'].values
    g = t['csmiles'].values                      # one row per compound
    # Use the published index arrays, not boolean masks: they carry the row
    # ORDER the original run used, and a random forest's bootstrap draw depends
    # on it (held-out R2 moves by ~0.006 between the two orderings). Same rows
    # either way; this keeps scheme A numerically comparable to Table 2.
    tr_mask, te_mask = itr, ite

    cv = grouped_cv_scores(t, y, g)
    ho, _ = fit_predict_heldout(t.iloc[tr_mask], y[tr_mask], t.iloc[te_mask], y[te_mask])

    # log P subset: held-out-style evaluation restricted to the 98 compounds,
    # models trained on the training compounds outside that subset's test part
    sub = t['name'].isin(lp_names).values
    sub_cv = grouped_cv_scores(t[sub].reset_index(drop=True), y[sub], g[sub])

    for nm in ho:
        results[(tag, nm)] = {**cv[nm], **ho[nm], 'logp_sub_R2': sub_cv[nm]['cv_R2'],
                              'logp_sub_RMSE': sub_cv[nm]['cv_RMSE'], 'unit': 'compound',
                              'n_fit': len(tr_mask), 'n_ho': len(te_mask)}
    p = pg_literature(t)
    results[(tag, 'Potts-Guy (literature)')] = {
        'cv_R2': r2_score(y, p), 'cv_R2_sd': np.nan,
        'cv_RMSE': math.sqrt(mean_squared_error(y, p)),
        'ho_R2': r2_score(y[te_mask], p[te_mask]),
        'ho_RMSE': math.sqrt(mean_squared_error(y[te_mask], p[te_mask])),
        'ho_MAE': mean_absolute_error(y[te_mask], p[te_mask]), 'ho_n': len(te_mask),
        'logp_sub_R2': r2_score(y[sub], p[sub]),
        'logp_sub_RMSE': math.sqrt(mean_squared_error(y[sub], p[sub])),
        'unit': 'compound', 'n_fit': len(tr_mask), 'n_ho': len(te_mask)}
    print('%s done' % tag)

# ---------------------------------------------------------------- scheme C
t = rec.reset_index(drop=True)
y = t['logkp'].values
g = t['csmiles'].values                          # many rows per compound
tr_mask = t['csmiles'].isin(train_cmp).values
te_mask = t['csmiles'].isin(test_cmp).values
assert not (set(g[tr_mask]) & set(g[te_mask])), 'compound crossed the held-out split'

cv = grouped_cv_scores(t, y, g)
ho, ho_preds = fit_predict_heldout(t[tr_mask], y[tr_mask], t[te_mask], y[te_mask])
sub = t['name'].isin(lp_names).values
sub_cv = grouped_cv_scores(t[sub].reset_index(drop=True), y[sub], g[sub])
for nm in ho:
    results[('C_record', nm)] = {**cv[nm], **ho[nm], 'logp_sub_R2': sub_cv[nm]['cv_R2'],
                                 'logp_sub_RMSE': sub_cv[nm]['cv_RMSE'], 'unit': 'record',
                                 'n_fit': int(tr_mask.sum()), 'n_ho': int(te_mask.sum())}
p = pg_literature(t)
results[('C_record', 'Potts-Guy (literature)')] = {
    'cv_R2': r2_score(y, p), 'cv_R2_sd': np.nan,
    'cv_RMSE': math.sqrt(mean_squared_error(y, p)),
    'ho_R2': r2_score(y[te_mask], p[te_mask]),
    'ho_RMSE': math.sqrt(mean_squared_error(y[te_mask], p[te_mask])),
    'ho_MAE': mean_absolute_error(y[te_mask], p[te_mask]), 'ho_n': int(te_mask.sum()),
    'logp_sub_R2': r2_score(y[sub], p[sub]),
    'logp_sub_RMSE': math.sqrt(mean_squared_error(y[sub], p[sub])),
    'unit': 'record', 'n_fit': int(tr_mask.sum()), 'n_ho': int(te_mask.sum())}
print('C_record done')

res = pd.DataFrame(results).T
res.index.names = ['scheme', 'model']
res.to_csv(OUT + 'scheme_comparison.csv')

# ---------------------------------------------------------------- scheme D
# Variance components first: an intercept-only model partitions the record-level
# variance into between-compound and within-compound (residual) parts. This is
# the quantity the ceiling argument rests on.
rec_s = rec.copy()
for f in FEATS:
    rec_s[f] = (rec_s[f] - rec_s[f].mean()) / rec_s[f].std()

null = MixedLM.from_formula('logkp ~ 1', groups='csmiles', data=rec_s).fit(reml=True)
vc_between = float(null.cov_re.iloc[0, 0])
vc_within = float(null.scale)
icc = vc_between / (vc_between + vc_within)

mm = {}
mm['null (intercept only)'] = dict(between=vc_between, within=vc_within, icc=icc,
                                   sd_between=math.sqrt(vc_between), sd_within=math.sqrt(vc_within))
for nm, fs in [('PG-form (logP, MW)', ['logP', 'MW']), ('MLR (all descriptors)', FEATS)]:
    f = 'logkp ~ ' + ' + '.join(fs)
    m = MixedLM.from_formula(f, groups='csmiles', data=rec_s).fit(reml=True)
    b, w = float(m.cov_re.iloc[0, 0]), float(m.scale)
    mm[nm] = dict(between=b, within=w, icc=b / (b + w),
                  sd_between=math.sqrt(b), sd_within=math.sqrt(w))

# held-out performance of the mixed model: predict test compounds from fixed
# effects only (their random intercept is unknown at prediction time -- which is
# exactly the situation for a new compound)
tr_s = rec_s[rec_s['csmiles'].isin(train_cmp)]
te_s = rec_s[rec_s['csmiles'].isin(test_cmp)]
for nm, fs in [('PG-form (logP, MW)', ['logP', 'MW']), ('MLR (all descriptors)', FEATS)]:
    f = 'logkp ~ ' + ' + '.join(fs)
    m = MixedLM.from_formula(f, groups='csmiles', data=tr_s).fit(reml=True)
    fe = m.fe_params
    pred = fe['Intercept'] + sum(fe[k] * te_s[k].values for k in fs)
    mm[nm]['ho_R2_fixed_only'] = r2_score(te_s['logkp'].values, pred)
    mm[nm]['ho_RMSE_fixed_only'] = math.sqrt(mean_squared_error(te_s['logkp'].values, pred))
    mm[nm]['ho_n_records'] = int(len(te_s))

pd.DataFrame(mm).T.to_csv(OUT + 'mixed_effects.csv')
json.dump({'variance_components': mm,
           'n_records': int(len(rec)), 'n_compounds': int(rec.csmiles.nunique()),
           'records_per_compound': {'median': float(rec.groupby('csmiles').size().median()),
                                    'max': int(rec.groupby('csmiles').size().max()),
                                    'singletons': int((rec.groupby('csmiles').size() == 1).sum())}},
          open(OUT + 'mixed_effects.json', 'w'), indent=1)

print('\n=== variance components (intercept-only, records) ===')
print('  between-compound var %.4f (SD %.3f)' % (vc_between, math.sqrt(vc_between)))
print('  within-compound var  %.4f (SD %.3f)' % (vc_within, math.sqrt(vc_within)))
print('  ICC = %.3f  -> %.0f%% of record-level variance is between compounds' % (icc, 100 * icc))
print('\nwrote', OUT)
