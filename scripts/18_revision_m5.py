"""Reviewer 2, Major 5.

"The two strata differ not only in skin layer but in n, chemical composition and
contributing laboratories. Matching on descriptor distribution, or a permutation
test preserving compound identity, would guard against attributing to skin layer
what is a difference in chemical space."

Both are done.

(1) Permutation test preserving compound identity. Compounds are relabelled at
    random into two pseudo-strata of exactly the observed sizes (192 / 199) and
    the whole stratified-CV comparison is rerun, building a null distribution for
    the R2 gap under "layer label carries no information". A compound keeps all
    of its own records and descriptors; only the label moves.

(2) Descriptor matching. Each epidermis compound is matched to the nearest
    unused mixed-stratum compound in standardised 15-descriptor space
    (Mahalanobis-style on the standardised scale, greedy nearest neighbour), and
    the comparison is rerun on the matched pairs only, so the two strata are
    forced to occupy the same chemical space.

The permutation null uses 150-tree forests and a single 5-fold split per draw to
stay inside the revision deadline; the observed statistic is recomputed under
exactly the same settings, so observed and null are like-for-like even though
both sit below the published run's precision. The published
50-fold estimate is reported alongside for reference but is not what the null is
built against.

Writes to revision/m5/. Nothing published is overwritten.
"""
import json, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats

ROOT = '/workspace/icbmehs_c3_skin/'
D = ROOT + 'data/'
OUT = ROOT + 'revision/m5/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260821
N_PERM = 200
N_TREES = 150
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']

rec = pd.read_csv(D + 'records_with_descriptors.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)

def lump(s):
    s = str(s).lower().strip()
    if 'stratum corneum' in s and 'without' not in s: return 'stratum corneum'
    if s.startswith('epidermis') and 'without' not in s and ',' not in s: return 'epidermis'
    if s == 'dermis': return 'dermis'
    return 'other/mixed'

rec['layer_g'] = rec['layer'].map(lump)
# compound-level table per stratum, exactly as the published analysis built it
def compound_table(sub):
    return (sub.groupby('csmiles')
              .agg(logkp=('logkp', 'median'), **{f: (f, 'first') for f in FEATS})
              .reset_index())

epi = compound_table(rec[rec.layer_g == 'epidermis'])
mix = compound_table(rec[rec.layer_g == 'other/mixed'])
print('epidermis %d compounds / mixed %d compounds' % (len(epi), len(mix)))
overlap = set(epi.csmiles) & set(mix.csmiles)
print('compounds appearing in BOTH strata: %d' % len(overlap))

def cv_r2(tbl, seed=SEED, k=5, trees=N_TREES):
    """Mean R2 over a single k-fold split. One row per compound, so folds are
    compound-disjoint by construction."""
    y = tbl['logkp'].values
    X = tbl[FEATS].values
    out = []
    for tr, va in KFold(n_splits=k, shuffle=True, random_state=seed).split(X):
        m = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                  random_state=SEED, n_jobs=8).fit(X[tr], y[tr])
        out.append(r2_score(y[va], m.predict(X[va])))
    return float(np.mean(out))

obs_epi, obs_mix = cv_r2(epi), cv_r2(mix)
obs_gap = obs_epi - obs_mix
print('\nobserved (150 trees, single 5-fold): epidermis %.3f, mixed %.3f, gap %+.3f'
      % (obs_epi, obs_mix, obs_gap))
print('published (400 trees, 50 repeated folds): 0.443 vs 0.213, gap +0.231')

# ---------------------------------------------------------------- (1) permutation
# Pool the compounds that carry a layer label, keep each compound's own data,
# and shuffle only which stratum the compound is assigned to.
pool = pd.concat([epi.assign(_src='epi'), mix.assign(_src='mix')], ignore_index=True)
n_epi = len(epi)
rng = np.random.default_rng(SEED)
null = []
for i in range(N_PERM):
    idx = rng.permutation(len(pool))
    a = pool.iloc[idx[:n_epi]].reset_index(drop=True)
    b = pool.iloc[idx[n_epi:]].reset_index(drop=True)
    null.append(cv_r2(a, seed=SEED + i) - cv_r2(b, seed=SEED + i))
    if (i + 1) % 50 == 0:
        print('   permutation %d/%d' % (i + 1, N_PERM), flush=True)
null = np.array(null)
p_perm = (np.sum(np.abs(null) >= abs(obs_gap)) + 1) / (N_PERM + 1)
print('\n=== permutation test (compound identity preserved, %d draws) ===' % N_PERM)
print('  null gap: mean %+.3f, SD %.3f, 2.5-97.5%% %+.3f to %+.3f'
      % (null.mean(), null.std(), np.percentile(null, 2.5), np.percentile(null, 97.5)))
print('  observed gap %+.3f  ->  two-sided permutation p = %.4f' % (obs_gap, p_perm))
pd.DataFrame({'null_gap': null}).to_csv(OUT + 'm5_permutation_null.csv', index=False)

# ---------------------------------------------------------------- (2) matching
both = pd.concat([epi[FEATS], mix[FEATS]], ignore_index=True)
mu, sd = both.mean(), both.std().replace(0, 1)
Ze = ((epi[FEATS] - mu) / sd).values
Zm = ((mix[FEATS] - mu) / sd).values
used = np.zeros(len(Zm), dtype=bool)
pairs = []
for i in range(len(Ze)):
    d = np.linalg.norm(Zm - Ze[i], axis=1)
    d[used] = np.inf
    j = int(np.argmin(d))
    if np.isinf(d[j]):
        break
    used[j] = True
    pairs.append((i, j, float(d[j])))
pi = [p[0] for p in pairs]; pj = [p[1] for p in pairs]
epi_m = epi.iloc[pi].reset_index(drop=True)
mix_m = mix.iloc[pj].reset_index(drop=True)
print('\n=== descriptor-matched comparison (%d pairs) ===' % len(pairs))
print('  mean matching distance %.3f (standardised units)' % np.mean([p[2] for p in pairs]))

bal = []
for f in FEATS:
    t, pv = stats.ttest_ind(epi_m[f], mix_m[f])
    smd = (epi_m[f].mean() - mix_m[f].mean()) / math.sqrt(
        (epi_m[f].var(ddof=1) + mix_m[f].var(ddof=1)) / 2 + 1e-12)
    t0, pv0 = stats.ttest_ind(epi[f], mix[f])
    smd0 = (epi[f].mean() - mix[f].mean()) / math.sqrt(
        (epi[f].var(ddof=1) + mix[f].var(ddof=1)) / 2 + 1e-12)
    bal.append(dict(descriptor=f, smd_before=smd0, smd_after=smd, p_after=pv))
bal = pd.DataFrame(bal)
bal.to_csv(OUT + 'm5_covariate_balance.csv', index=False)
print('  |SMD| > 0.2 before matching: %d / 15   after matching: %d / 15'
      % ((bal.smd_before.abs() > 0.2).sum(), (bal.smd_after.abs() > 0.2).sum()))

m_epi, m_mix = cv_r2(epi_m), cv_r2(mix_m)
print('  matched epidermis %.3f vs matched mixed %.3f  ->  gap %+.3f'
      % (m_epi, m_mix, m_epi - m_mix))

json.dump(dict(observed=dict(epi=obs_epi, mixed=obs_mix, gap=obs_gap,
                             n_epi=len(epi), n_mix=len(mix),
                             compounds_in_both_strata=len(overlap)),
               permutation=dict(n=N_PERM, null_mean=float(null.mean()),
                                null_sd=float(null.std()), p=float(p_perm),
                                null_lo=float(np.percentile(null, 2.5)),
                                null_hi=float(np.percentile(null, 97.5))),
               matched=dict(n_pairs=len(pairs), epi=m_epi, mixed=m_mix,
                            gap=m_epi - m_mix,
                            mean_distance=float(np.mean([p[2] for p in pairs])),
                            imbalanced_before=int((bal.smd_before.abs() > 0.2).sum()),
                            imbalanced_after=int((bal.smd_after.abs() > 0.2).sum())),
               settings=dict(trees=N_TREES, folds=10, note='reduced settings for the '
                             'permutation null; observed statistic recomputed identically')),
          open(OUT + 'm5_results.json', 'w'), indent=1)
print('\nwrote', OUT)
