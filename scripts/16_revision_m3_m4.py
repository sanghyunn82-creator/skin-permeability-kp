"""Reviewer 2, Major 3 and Major 4.

M3 -- "External performance rests on one 80:20 split with a fixed seed ... Please
report a distribution of external R2 over many splits ... so the model ranking in
Table 2 can be effectively judged."
    -> 200 repeated stratified compound-level 80:20 splits. For each model we
       report the median held-out R2, a 95% percentile interval, and how often
       the model takes rank 1, so the ranking can be judged rather than assumed.

M4 -- "the other fold-wise tests treat 50 repeated cross-validation folds as
independent samples, which they are not. Please use a corrected test, or report
effect size intervals and drop the p-values."
    -> Both. Nadeau-Bengio corrected resampled t-tests (variance inflated by
       1/k + n_test/n_train, which the naive test ignores), alongside effect
       sizes with bootstrap intervals so the result stands without a p-value.

Fold-level values come from figdata/f4a_folds.csv and f4b_folds.csv, produced by
scripts/13_foldlevel.py, which reproduced every published aggregate exactly.

Writes to revision/major1/../ (revision/) -- nothing published is overwritten.
"""
import json, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from xgboost import XGBRegressor

ROOT = '/workspace/icbmehs_c3_skin/'
D, F = ROOT + 'data/', ROOT + 'figdata/'
OUT = ROOT + 'revision/m3_m4/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260821
LOG3600 = math.log10(3600.0)
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']
N_SPLITS = 200

# =====================================================================
# M3: distribution of held-out R2 over many splits
# =====================================================================
a = pd.read_csv(D + 'compounds_modelling.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
y = a['logkp'].values
analg = a['analgesic'].values.astype(bool)
pg_all = -2.72 + 0.71 * a['logP'].values - 0.0061 * a['MW'].values - LOG3600

# same stratification the published split used
strat = pd.cut(y, bins=[-np.inf, -7.5, -6.5, -5.5, np.inf], labels=False).astype(str) + np.where(analg, 'A', 'N')
vc = pd.Series(strat).value_counts()
strat = np.where(pd.Series(strat).map(vc).values < 2, 'rare', strat)

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

names = list(models().keys()) + ['Potts-Guy (literature)']
r2_mat = {n: [] for n in names}
rmse_mat = {n: [] for n in names}
print('M3: %d repeated stratified 80:20 compound-level splits' % N_SPLITS)
for s in range(N_SPLITS):
    itr, ite = train_test_split(np.arange(len(a)), test_size=0.20,
                                random_state=SEED + s, stratify=strat)
    # one row per compound, so a compound cannot straddle the split by construction
    for nm, (mdl, fs) in models().items():
        X = a[fs].values
        m = clone(mdl).fit(X[itr], y[itr])
        p = m.predict(X[ite])
        r2_mat[nm].append(r2_score(y[ite], p))
        rmse_mat[nm].append(math.sqrt(mean_squared_error(y[ite], p)))
    r2_mat['Potts-Guy (literature)'].append(r2_score(y[ite], pg_all[ite]))
    rmse_mat['Potts-Guy (literature)'].append(math.sqrt(mean_squared_error(y[ite], pg_all[ite])))
    if (s + 1) % 50 == 0:
        print('   %d/%d' % (s + 1, N_SPLITS))

R2 = pd.DataFrame(r2_mat)
RM = pd.DataFrame(rmse_mat)
R2.to_csv(OUT + 'm3_heldout_R2_by_split.csv', index=False)
RM.to_csv(OUT + 'm3_heldout_RMSE_by_split.csv', index=False)

# rank 1 frequency, and pairwise "how often does A beat B"
ranks = R2.rank(axis=1, ascending=False)
summary = pd.DataFrame({
    'median_R2': R2.median(),
    'lo95': R2.quantile(0.025),
    'hi95': R2.quantile(0.975),
    'sd': R2.std(),
    'median_RMSE': RM.median(),
    'rank1_pct': (ranks == 1).mean() * 100,
    'mean_rank': ranks.mean(),
}).sort_values('median_R2', ascending=False)
summary.to_csv(OUT + 'm3_summary.csv')
print('\n=== M3 held-out R2 across %d splits ===' % N_SPLITS)
print(summary.round(3).to_string())

rf, gb = 'Random forest', 'Gradient boosting (XGB)'
print('\nRF beats GBM in %.1f%% of splits; median gap %.3f (95%% interval %.3f to %.3f)'
      % ((R2[rf] > R2[gb]).mean() * 100, (R2[rf] - R2[gb]).median(),
         (R2[rf] - R2[gb]).quantile(0.025), (R2[rf] - R2[gb]).quantile(0.975)))
published = 0.312
pct = (R2[rf] < published).mean() * 100
print('published single-split RF R2 = %.3f sits at the %.0fth percentile of its own distribution'
      % (published, pct))

# =====================================================================
# M4: corrected tests for fold-wise comparisons
# =====================================================================
def nadeau_bengio(diff, n_train, n_test):
    """Corrected resampled t-test. The naive test uses var/k; folds share
    training data, so the variance is inflated by (1/k + n_test/n_train)."""
    k = len(diff)
    m = float(np.mean(diff))
    v = float(np.var(diff, ddof=1))
    if v == 0:
        return dict(mean=m, t=np.nan, p=np.nan, df=k - 1, corrected_se=0.0)
    se = math.sqrt(v * (1.0 / k + n_test / n_train))
    t = m / se
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return dict(mean=m, t=t, p=p, df=k - 1, corrected_se=se)

def boot_ci(x, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

m4 = {}
fa = pd.read_csv(F + 'f4a_folds.csv')     # 10 grouped folds, record level (732 records)
fb = pd.read_csv(F + 'f4b_folds.csv')     # 50 repeated folds per stratum, compound level

# --- paired record-level comparisons (Fig. 4a), 10 grouped folds over 732 records
n_rec = 732
n_test_r, n_train_r = n_rec / 10.0, n_rec * 9 / 10.0
pub_paired = {'protocol_vs_chemistry': ('protocol', 'chemistry', 0.556640625),
              'both_vs_chemistry':     ('both', 'chemistry', 0.005859375),
              'both_vs_protocol':      ('both', 'protocol', 0.4921875)}
for key, (A, B, p_pub) in pub_paired.items():
    d = (fa[A] - fa[B]).values
    nb = nadeau_bengio(d, n_train_r, n_test_r)
    w = stats.wilcoxon(fa[A], fa[B])
    lo, hi = boot_ci(d)
    m4[key] = dict(mean_diff=nb['mean'], ci95_lo=lo, ci95_hi=hi,
                   corrected_t=nb['t'], corrected_p=nb['p'], df=nb['df'],
                   published_wilcoxon_p=p_pub, naive_wilcoxon_p=float(w.pvalue),
                   k_folds=len(d))
    print('\n%s: mean diff %+.3f (95%% CI %+.3f to %+.3f)' % (key, nb['mean'], lo, hi))
    print('   published Wilcoxon p = %.4f  ->  Nadeau-Bengio corrected p = %.4f' % (p_pub, nb['p']))

# --- skin-layer comparison (Fig. 4b), 50 repeated folds per stratum, unpaired
n_epi, n_mix = 192, 199
diff_layer = fb['epidermis'].values.mean() - fb['other_mixed'].values.mean()
# unpaired, but both sides are repeated CV -> correct each side's variance, then combine
k = len(fb)
def corrected_var(x, n_total):
    n_test = n_total / 10.0
    n_train = n_total * 9 / 10.0
    return float(np.var(x, ddof=1)) * (1.0 / k + n_test / n_train)
ve = corrected_var(fb['epidermis'].values, n_epi)
vm = corrected_var(fb['other_mixed'].values, n_mix)
se = math.sqrt(ve + vm)
t = diff_layer / se
p_corr = 2 * stats.t.sf(abs(t), df=k - 1)
u = stats.mannwhitneyu(fb['epidermis'], fb['other_mixed'], alternative='two-sided')
lo_e, hi_e = boot_ci(fb['epidermis'].values)
lo_m, hi_m = boot_ci(fb['other_mixed'].values)
# Cohen's d on the fold values (descriptive effect size)
pooled_sd = math.sqrt((np.var(fb['epidermis'], ddof=1) + np.var(fb['other_mixed'], ddof=1)) / 2)
m4['layer_epidermis_vs_mixed'] = dict(
    mean_epi=float(fb['epidermis'].mean()), epi_ci95=[lo_e, hi_e],
    mean_mixed=float(fb['other_mixed'].mean()), mixed_ci95=[lo_m, hi_m],
    mean_diff=float(diff_layer), corrected_t=float(t), corrected_p=float(p_corr),
    df=k - 1, cohens_d=float(diff_layer / pooled_sd),
    published_mannwhitney_p=3.5505478185193758e-06, naive_mannwhitney_p=float(u.pvalue),
    k_folds=k)
print('\nlayer epidermis vs mixed: %.3f (95%% CI %.3f-%.3f) vs %.3f (95%% CI %.3f-%.3f)'
      % (fb['epidermis'].mean(), lo_e, hi_e, fb['other_mixed'].mean(), lo_m, hi_m))
print('   difference %+.3f, Cohen d = %.2f' % (diff_layer, diff_layer / pooled_sd))
print('   published Mann-Whitney p = 3.55e-06  ->  corrected t-test p = %.4f' % p_corr)

json.dump(m4, open(OUT + 'm4_corrected_tests.json', 'w'), indent=1)
pd.DataFrame(m4).T.to_csv(OUT + 'm4_corrected_tests.csv')
print('\nwrote', OUT)
