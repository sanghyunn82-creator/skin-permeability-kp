"""Recover the per-fold R2 values behind Fig. 4a and 4b.

08_stats.py computed these folds but saved only their means, SDs and test
statistics. This script re-runs the identical computation (same seed, same
data, same model configuration, same fold generators) and writes the
fold-level values out, so that Fig. 4a/4b can show the distribution rather
than only mean +/- SD.

It refuses to write anything unless every aggregate it reproduces matches the
published value in data/stats_tests.json to within 1e-9. If the environment
has drifted, the run aborts rather than silently producing figures that
disagree with the manuscript.
"""
import math, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from scipy import stats

D = '/workspace/icbmehs_c3_skin/data/'
F = '/workspace/icbmehs_c3_skin/figdata/'
SEED = 20260821
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']

published = json.load(open(D + 'stats_tests.json'))

r = pd.read_csv(D + 'records_with_descriptors.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
CAT = ['layer_g', 'site', 'cell_type']
NUM = ['donor_temp', 'donor_ph', 'acceptor_temp', 'acceptor_ph']

def lump(s):
    s = str(s).lower().strip()
    if 'stratum corneum' in s and 'without' not in s: return 'stratum corneum'
    if s.startswith('epidermis') and 'without' not in s and ',' not in s: return 'epidermis'
    if s == 'dermis': return 'dermis'
    return 'other/mixed'

r['layer_g'] = r['layer'].map(lump)
for c in ['site', 'cell_type']:
    r[c] = r[c].astype(str).str.lower().str.strip().replace({'nan': 'unknown'})
y = r['logkp'].values
g = r['csmiles'].values

def mk(cn, cc):
    t = []
    if cn: t.append(('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), cn))
    if cc: t.append(('c', make_pipeline(SimpleImputer(strategy='constant', fill_value='unknown'),
                     OneHotEncoder(handle_unknown='ignore', min_frequency=5)), cc))
    return make_pipeline(ColumnTransformer(t),
                         RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                               random_state=SEED, n_jobs=8))

# ---- Fig. 4a: chemistry / protocol / both over 10 identical grouped folds ----
SPECS = {'chem': (FEATS, []), 'proto': (NUM, CAT), 'both': (FEATS + NUM, CAT)}
folds = list(GroupKFold(n_splits=10).split(r, y, groups=g))
scores = {k: [] for k in SPECS}
for tr, va in folds:
    for k, (cn, cc) in SPECS.items():
        m = mk(cn, cc)
        m.fit(r.loc[tr, cn + cc], y[tr])
        scores[k].append(r2_score(y[va], m.predict(r.loc[va, cn + cc])))
scores = {k: np.array(v) for k, v in scores.items()}

# ---- Fig. 4b: epidermis vs mixed over 50 repeated folds each ----
res = {}
for lg in ['epidermis', 'other/mixed']:
    sub = r[r.layer_g == lg]
    ag = sub.groupby('csmiles').agg(logkp=('logkp', 'median'),
                                    **{f: (f, 'first') for f in FEATS}).reset_index()
    rk = RepeatedKFold(n_splits=10, n_repeats=5, random_state=SEED)
    s = []
    for tr, va in rk.split(ag):
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=8)
        m.fit(ag.loc[tr, FEATS], ag.loc[tr, 'logkp'])
        s.append(r2_score(ag.loc[va, 'logkp'], m.predict(ag.loc[va, FEATS])))
    res[lg] = np.array(s)

# ---------------- reproduction check ----------------
TOL = 1e-9
checks = []
for A, B in [('proto', 'chem'), ('both', 'chem'), ('both', 'proto')]:
    w = stats.wilcoxon(scores[A], scores[B])
    p = published['fold_%s_vs_%s' % (A, B)]
    checks += [
        ('fold_%s_vs_%s meanA' % (A, B), scores[A].mean(), p['meanA']),
        ('fold_%s_vs_%s meanB' % (A, B), scores[B].mean(), p['meanB']),
        ('fold_%s_vs_%s W' % (A, B), float(w.statistic), p['W']),
        ('fold_%s_vs_%s p' % (A, B), float(w.pvalue), p['p']),
    ]
u = stats.mannwhitneyu(res['epidermis'], res['other/mixed'], alternative='two-sided')
p = published['layer_epi_vs_mixed']
checks += [
    ('layer mean_epi', res['epidermis'].mean(), p['mean_epi']),
    ('layer mean_mixed', res['other/mixed'].mean(), p['mean_mixed']),
    ('layer U', float(u.statistic), p['U']),
    ('layer p', float(u.pvalue), p['p']),
]

bad = [(n, a, b) for n, a, b in checks if not (abs(a - b) <= TOL or (b != 0 and abs(a - b) / abs(b) <= 1e-9))]
for n, a, b in checks:
    mark = 'OK ' if (n, a, b) not in bad else 'FAIL'
    print('%s %-28s recomputed=%.12g published=%.12g' % (mark, n, a, b))

if bad:
    raise SystemExit('\nABORTED: %d aggregate(s) did not reproduce. Fold-level values NOT written; '
                     'Fig. 4 left unchanged.' % len(bad))

print('\nAll %d aggregates reproduced exactly. Writing fold-level values.' % len(checks))
pd.DataFrame({'fold': np.arange(1, 11), 'chemistry': scores['chem'],
              'protocol': scores['proto'], 'both': scores['both']}
             ).to_csv(F + 'f4a_folds.csv', index=False)
pd.DataFrame({'fold': np.arange(1, 51), 'epidermis': res['epidermis'],
              'other_mixed': res['other/mixed']}).to_csv(F + 'f4b_folds.csv', index=False)
print('wrote', F + 'f4a_folds.csv', 'and', F + 'f4b_folds.csv')
