"""Reviewer 2, Major 8.

"Only 2D descriptors are used, yet much of the Discussion turns on ionizable
drugs and receptor pH. Computing pKa and the fraction ionized is inexpensive
and directly relevant. Please also test whether MW carries information
independent of LabuteASA."

Part 1 -- ionization. Site pKa values come from the Dimorphite-DL substructure
library (Ropp et al., J Cheminform 2019;11:14), which pairs SMARTS patterns with
empirical pKa means. For each compound we take the strongest acidic and strongest
basic site and compute the fraction ionized by Henderson-Hasselbalch, at pH 7.4
and at each record's own receptor pH where one was reported. Treating the
molecule as monoprotic in each direction is a simplification and is reported as
such; compounds with no matched ionizable site are carried as neutral.

Part 2 -- is MW redundant given LabuteASA? Their Spearman rho is 0.97 (Figure S3),
so the question is whether MW adds anything once LabuteASA is in the model. Tested
three ways: variance inflation, the partial correlation of MW with log Kp given
LabuteASA, and a nested model comparison under the same grouped CV used elsewhere.

RDKit 2026.03.6 is used here for structure parsing only. The 15 modelling
descriptors are the ones already stored in data/ (computed with 2026.03.5); they
are not recomputed, so nothing published shifts.

Writes to revision/m8/. Nothing published is overwritten.
"""
import json, math, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import clone
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = '/workspace/icbmehs_c3_skin/'
D = ROOT + 'data/'
OUT = ROOT + 'revision/m8/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260821
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']

# ============================================================ pKa site library
import dimorphite_dl
SMARTS_FILE = os.path.join(os.path.dirname(dimorphite_dl.__file__),
                           'smarts', 'site_substructures.smarts')
sites = []
for line in open(SMARTS_FILE):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split('\t')
    if len(parts) < 4:
        continue
    name, smarts = parts[0], parts[1]
    try:
        pka = float(parts[3])
    except ValueError:
        continue
    if pka <= -500:                      # dimorphite's sentinel for "never ionizes"
        continue
    patt = Chem.MolFromSmarts(smarts)
    if patt is not None:
        sites.append((name, patt, pka))
print('pKa site library: %d usable patterns' % len(sites))

# Acid vs base: dimorphite's SMARTS for an acidic site carries the ionizable
# proton explicitly, so a pattern containing an explicit [H] on the reacting
# atom is a proton donor. Names are used only to resolve the handful that this
# rule cannot settle, and every assignment is written out for inspection.
BASIC_HINT = re.compile(r'amine|amidine|guanidin|imid|pyridin|azide|indol|amino', re.I)
ACIDIC_HINT = re.compile(r'carboxyl|phenol|sulf|phosph|thiol|tetrazol|acid|barbit|imide|nitro', re.I)

def classify(name, smarts_str):
    if ACIDIC_HINT.search(name):
        return 'acid'
    if BASIC_HINT.search(name):
        return 'base'
    return 'acid' if '-[H]' in smarts_str or '[H]' in smarts_str else 'base'

site_rows = []
for line in open(SMARTS_FILE):
    p = line.strip().split('\t')
    if len(p) >= 4:
        try:
            pk = float(p[3])
        except ValueError:
            continue
        if pk > -500:
            site_rows.append(dict(name=p[0], smarts=p[1], pka=pk, kind=classify(p[0], p[1])))
pd.DataFrame(site_rows).to_csv(OUT + 'pka_site_library_used.csv', index=False)
kinds = {r['name']: r['kind'] for r in site_rows}

def ionization(smiles, ph):
    """Strongest acidic / basic site and the resulting fraction ionized."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    acid_pkas, base_pkas = [], []
    for name, patt, pka in sites:
        if m.HasSubstructMatch(patt):
            (acid_pkas if kinds.get(name) == 'acid' else base_pkas).append(pka)
    # most acidic = lowest pKa; most basic = highest pKa
    pka_a = min(acid_pkas) if acid_pkas else np.nan
    pka_b = max(base_pkas) if base_pkas else np.nan
    fa = 1.0 / (1.0 + 10 ** (pka_a - ph)) if acid_pkas else 0.0   # fraction deprotonated
    fb = 1.0 / (1.0 + 10 ** (ph - pka_b)) if base_pkas else 0.0   # fraction protonated
    if acid_pkas and base_pkas:
        cls = 'ampholyte'
    elif acid_pkas:
        cls = 'acid'
    elif base_pkas:
        cls = 'base'
    else:
        cls = 'neutral'
    return dict(pka_acid=pka_a, pka_base=pka_b, frac_anion=fa, frac_cation=fb,
                frac_ionized=max(fa, fb), frac_neutral=1.0 - max(fa, fb), ion_class=cls)

# ============================================================ apply
cmp = pd.read_csv(D + 'compounds_modelling.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
rows = []
for s in cmp['csmiles']:
    r = ionization(s, 7.4)
    rows.append(r if r else dict(pka_acid=np.nan, pka_base=np.nan, frac_anion=np.nan,
                                 frac_cation=np.nan, frac_ionized=np.nan,
                                 frac_neutral=np.nan, ion_class='unparsed'))
ion = pd.DataFrame(rows)
cmp_i = pd.concat([cmp, ion], axis=1)
cmp_i.to_csv(OUT + 'compounds_with_ionization.csv', index=False)
print('\n=== ionization class at pH 7.4 (n = %d compounds) ===' % len(cmp_i))
print(cmp_i['ion_class'].value_counts().to_string())
print('matched an ionizable site: %d / %d (%.0f%%)'
      % ((cmp_i.ion_class.isin(['acid', 'base', 'ampholyte'])).sum(), len(cmp_i),
         100 * (cmp_i.ion_class.isin(['acid', 'base', 'ampholyte'])).mean()))
print('median fraction ionized at pH 7.4 = %.3f' % cmp_i['frac_ionized'].median())

# record-level: use each record's own receptor pH where reported
rec = pd.read_csv(D + 'records_with_descriptors.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
cache = {}
fi = []
for smi, ph in zip(rec['csmiles'], rec['acceptor_ph']):
    use_ph = ph if pd.notna(ph) else 7.4
    key = (smi, round(float(use_ph), 2))
    if key not in cache:
        cache[key] = ionization(smi, float(use_ph))
    r = cache[key]
    fi.append(r['frac_ionized'] if r else np.nan)
rec_i = rec.copy()
rec_i['frac_ionized_at_receptor_pH'] = fi
rec_i['receptor_pH_used'] = np.where(rec['acceptor_ph'].notna(), rec['acceptor_ph'], 7.4)
rec_i['receptor_pH_reported'] = rec['acceptor_ph'].notna()
rec_i.to_csv(OUT + 'records_with_ionization.csv', index=False)
print('records with a reported receptor pH: %d / %d' % (rec['acceptor_ph'].notna().sum(), len(rec)))

# ---- does ionization add anything? grouped CV, compound level
y = cmp_i['logkp'].values
g = cmp_i['csmiles'].values
ok = cmp_i['frac_ionized'].notna().values
def grouped_cv(df, cols, y, g, model, k=10):
    gk = GroupKFold(n_splits=k)
    r2s, rmses = [], []
    for tr, va in gk.split(df, y, groups=g):
        assert not (set(g[tr]) & set(g[va]))
        m = clone(model).fit(df.iloc[tr][cols].values, y[tr])
        p = m.predict(df.iloc[va][cols].values)
        r2s.append(r2_score(y[va], p)); rmses.append(math.sqrt(mean_squared_error(y[va], p)))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(rmses))

RF = RandomForestRegressor(n_estimators=800, min_samples_leaf=2, random_state=SEED, n_jobs=8)
LIN = make_pipeline(StandardScaler(), LinearRegression())
ION = ['frac_ionized', 'frac_anion', 'frac_cation']
tbl = {}
for tag, cols, mdl in [
        ('RF, 15 descriptors', FEATS, RF),
        ('RF, 15 + ionization', FEATS + ION, RF),
        ('PG-form (logP, MW)', ['logP', 'MW'], LIN),
        ('PG-form + ionization', ['logP', 'MW'] + ION, LIN)]:
    r2, sd, rm = grouped_cv(cmp_i[ok].reset_index(drop=True), cols, y[ok], g[ok], mdl)
    tbl[tag] = dict(cv_R2=r2, cv_R2_sd=sd, cv_RMSE=rm, n=int(ok.sum()), n_features=len(cols))
    print('  %-24s grouped-CV R2 = %.3f +/- %.3f   RMSE = %.3f' % (tag, r2, sd, rm))

# ============================================================ MW vs LabuteASA
print('\n=== does MW carry information independent of LabuteASA? ===')
rho, p_rho = stats.spearmanr(cmp_i['MW'], cmp_i['LabuteASA'])
print('Spearman rho(MW, LabuteASA) = %.4f  (p = %.3g)' % (rho, p_rho))

# partial correlation of MW with log Kp, controlling for LabuteASA
def partial_corr(x, y_, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y_ - np.polyval(np.polyfit(z, y_, 1), z)
    return stats.pearsonr(rx, ry)
pr, pp = partial_corr(cmp_i['MW'].values, y, cmp_i['LabuteASA'].values)
r_simple, _ = stats.pearsonr(cmp_i['MW'].values, y)
print('r(MW, logKp) = %+.3f   ->   partial r given LabuteASA = %+.3f (p = %.3g)'
      % (r_simple, pr, pp))

Xv = cmp_i[['MW', 'LabuteASA', 'MR', 'HeavyAtoms']].values
vifs = {c: variance_inflation_factor(Xv, i)
        for i, c in enumerate(['MW', 'LabuteASA', 'MR', 'HeavyAtoms'])}
print('VIF among the size descriptors: ' + ', '.join('%s %.1f' % (k, v) for k, v in vifs.items()))

nested = {}
for tag, cols in [('LabuteASA only', ['LabuteASA']),
                  ('MW only', ['MW']),
                  ('LabuteASA + MW', ['LabuteASA', 'MW']),
                  ('15 descriptors', FEATS),
                  ('15 minus MW', [c for c in FEATS if c != 'MW']),
                  ('15 minus LabuteASA', [c for c in FEATS if c != 'LabuteASA'])]:
    r2, sd, rm = grouped_cv(cmp_i, cols, y, g, RF)
    nested[tag] = dict(cv_R2=r2, cv_R2_sd=sd, cv_RMSE=rm, n_features=len(cols))
    print('  %-22s grouped-CV R2 = %.3f +/- %.3f' % (tag, r2, sd))

json.dump({'ionization_models': tbl, 'mw_vs_labute': {
    'spearman_rho': float(rho), 'partial_r_MW_given_LabuteASA': float(pr),
    'partial_p': float(pp), 'simple_r_MW': float(r_simple), 'vif': vifs,
    'nested': nested}}, open(OUT + 'm8_results.json', 'w'), indent=1)
pd.DataFrame(tbl).T.to_csv(OUT + 'm8_ionization_models.csv')
pd.DataFrame(nested).T.to_csv(OUT + 'm8_mw_labute_nested.csv')
print('\nwrote', OUT)
