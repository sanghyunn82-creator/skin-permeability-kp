"""Reviewer 2, Major 2.

"The plateau above log P 2.5 and the break near 220 g/mol are the paper's main
mechanistic claims but derive from partial dependence on one fitted forest. PDPs
are unreliable under collinearity, and Figure S3 documents exactly that (log P
with MR, MW with LabuteASA, and HeavyAtoms at rho >= 0.9). Please supply
accumulated local effects plots and confidence bands across bootstrap or
cross-validation replicates. Note also that the MW effects span only 0.44 log
units, small against the ~1.1 log unit noise floor you establish yourselves."

Accumulated local effects avoid the PDP's failure mode: PDP averages the model
over marginal values of the other features, which fabricates molecules that do
not exist when features are correlated (an MW of 100 with a LabuteASA of 200).
ALE instead accumulates local differences inside narrow windows of the feature,
so it only ever asks the model about combinations the data actually contains.

Bands come from 200 bootstrap refits of the forest; the band is the 2.5-97.5
percentile of the ALE curve across refits. The within-compound noise SD from the
mixed model (0.963 log units) is drawn on the same axis so the size of the
effect can be read against the floor, which is the reviewer's second point.

ALE is implemented here directly (no suitable package is installed) and checked
against a known-additive synthetic case before use.

Writes to revision/m2/. Nothing published is overwritten.
"""
import json, math, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

ROOT = '/workspace/icbmehs_c3_skin/'
D = ROOT + 'data/'
OUT = ROOT + 'revision/m2/'
os.makedirs(OUT, exist_ok=True)
SEED = 20260821
N_BOOT = 200
N_BINS = 20
FEATS = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings', 'Rings',
         'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']
NOISE_SD = 0.963          # within-compound SD from the mixed model (scripts/15)

assert 'Arial' in {f.name for f in font_manager.fontManager.ttflist}, 'Arial not registered'
BLUE, ORANGE, INK, INK2, INK3, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#8a8880', '#dedcd6'
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.7, 'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'text.color': INK,
    'axes.labelcolor': INK, 'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': GRID,
})

def ale_1d(model, X, j, bins=N_BINS):
    """First-order ALE for feature j. Returns (bin edges, centred ALE)."""
    x = X[:, j]
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return None, None
    idx = np.clip(np.searchsorted(qs, x, side='left'), 1, len(qs) - 1)
    eff = np.zeros(len(qs) - 1)
    for k in range(1, len(qs)):
        m = idx == k
        if not m.any():
            continue
        lo = X[m].copy(); lo[:, j] = qs[k - 1]
        hi = X[m].copy(); hi[:, j] = qs[k]
        eff[k - 1] = np.mean(model.predict(hi) - model.predict(lo))
    ale = np.concatenate([[0.0], np.cumsum(eff)])
    # centre on the data distribution, as ALE is defined up to a constant
    counts = np.bincount(idx, minlength=len(qs))[1:]
    mid = (ale[:-1] + ale[1:]) / 2.0
    ale -= np.sum(mid * counts) / max(counts.sum(), 1)
    return qs, ale

# ---- sanity check on a synthetic additive function with correlated inputs
rng = np.random.default_rng(SEED)
n = 1500
z = rng.normal(size=n)
x1 = z + 0.2 * rng.normal(size=n)
x2 = z + 0.2 * rng.normal(size=n)          # rho(x1,x2) ~ 0.96, like MW/LabuteASA
ytrue = 2.0 * x1 + 0.0 * x2
chk = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=8).fit(
    np.c_[x1, x2], ytrue + 0.05 * rng.normal(size=n))
q1, a1 = ale_1d(chk, np.c_[x1, x2], 0)
q2, a2 = ale_1d(chk, np.c_[x1, x2], 1)
slope = (a1[-1] - a1[0]) / (q1[-1] - q1[0])
print('ALE self-check on y = 2*x1 + 0*x2 with rho(x1,x2) = %.2f' % np.corrcoef(x1, x2)[0, 1])
print('  recovered x1 slope %.2f (true 2.00);  x2 total effect %.3f (true 0.00)'
      % (slope, a2[-1] - a2[0]))
if not (1.5 < slope < 2.5 and abs(a2[-1] - a2[0]) < 0.6):
    raise SystemExit('ALE implementation failed its self-check; not proceeding')

# ---- real data
a = pd.read_csv(D + 'compounds_modelling.csv').dropna(subset=FEATS + ['logkp']).reset_index(drop=True)
X = a[FEATS].values
y = a['logkp'].values
itr = np.load(D + 'split_train_idx.npy')
Xtr, ytr = X[itr], y[itr]

TARGETS = [('logP', 'Crippen log $P$'), ('MW', 'Molecular weight (g mol$^{-1}$)')]
res = {}
for feat, label in TARGETS:
    j = FEATS.index(feat)
    base = RandomForestRegressor(n_estimators=800, min_samples_leaf=2,
                                 random_state=SEED, n_jobs=8).fit(Xtr, ytr)
    q, ale = ale_1d(base, Xtr, j)
    boots = []
    rs = np.random.default_rng(SEED)
    for b in range(N_BOOT):
        bi = rs.choice(len(Xtr), len(Xtr), replace=True)
        mb = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   random_state=SEED + b, n_jobs=8).fit(Xtr[bi], ytr[bi])
        qb, ab = ale_1d(mb, Xtr[bi], j)
        boots.append(np.interp(q, qb, ab))
        if (b + 1) % 50 == 0:
            print('  %s bootstrap %d/%d' % (feat, b + 1, N_BOOT))
    B = np.vstack(boots)
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    span = float(ale.max() - ale.min())
    # widest band anywhere on the curve, as a blunt measure of how well pinned down it is
    band = float(np.mean(hi - lo))
    res[feat] = dict(x=q.tolist(), ale=ale.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                     span_log_units=span, mean_band_width=band,
                     span_vs_noise=span / NOISE_SD)
    print('%s: ALE span %.3f log units, mean 95%% band width %.3f, span/noise-SD %.2f'
          % (feat, span, band, span / NOISE_SD))

json.dump({'noise_sd_within_compound': NOISE_SD, 'n_boot': N_BOOT, 'n_bins': N_BINS,
           'results': res}, open(OUT + 'm2_ale.json', 'w'), indent=1)

# ---- figure
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))
for ax, (feat, label) in zip(axs, TARGETS):
    r = res[feat]
    x = np.array(r['x']); ale = np.array(r['ale'])
    ax.fill_between(x, r['lo'], r['hi'], color=BLUE, alpha=0.20, lw=0,
                    label='95% bootstrap band')
    ax.plot(x, ale, color=BLUE, lw=1.6, label='ALE')
    ax.axhline(0, color=INK3, lw=0.6, ls=':')
    # the noise floor, drawn as a band of +/- half the within-compound SD so its
    # height is directly comparable to the vertical travel of the ALE curve
    ax.axhspan(-NOISE_SD / 2, NOISE_SD / 2, color=ORANGE, alpha=0.10, lw=0,
               label='within-compound noise SD (%.2f)' % NOISE_SD)
    ax.plot(x, np.full_like(x, np.nan))
    for xv in x:
        ax.plot([xv, xv], [ax.get_ylim()[0], ax.get_ylim()[0]], color=INK3, lw=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel('ALE of predicted log $K_p$')
    ax.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    ax.text(0.03, 0.96, 'span %.2f log units' % res[feat]['span_log_units'],
            transform=ax.transAxes, va='top', fontsize=6.8, color=INK,
            bbox=dict(fc='white', ec=GRID, lw=0.5, boxstyle='round,pad=0.3'))
axs[0].legend(loc='lower right', fontsize=6.0)
fig.tight_layout(w_pad=2.0)
p = OUT + 'FigR1_ALE.tiff'
fig.savefig(p, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)
im = Image.open(p)
if im.mode != 'RGB':
    bg = Image.new('RGB', im.size, (255, 255, 255)); bg.paste(im); bg.save(p, format='TIFF',
        compression='tiff_lzw', dpi=(300, 300))
pv = Image.open(p).copy(); pv.thumbnail((1000, 1000))
pv.save(OUT + 'prev_FigR1_ALE.png')
print('\nwrote', OUT)
