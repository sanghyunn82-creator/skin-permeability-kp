#!/usr/bin/env /usr/bin/python3
"""Render the four submission figures locally in Arial.
Inputs are pre-aggregated CSV/NPZ produced on the pod (COMPUTE_POLICY: local render only).
Palette: dataviz categorical slots 1-3 (validated all-pairs) + blue sequential ramp.
Marker shape carries identity independently of colour, so the figures survive
greyscale printing."""
import json, math, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'figdata')
OUT = os.path.join(HERE, 'submission', 'figures')
os.makedirs(OUT, exist_ok=True)

# ---------- Arial ----------
ARIAL = '/System/Library/Fonts/Supplemental/'
for f in ['Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf']:
    p = os.path.join(ARIAL, f)
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
assert 'Arial' in {f.name for f in font_manager.fontManager.ttflist}, 'Arial not registered'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8880', '#dedcd6'
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
       '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('seqblue', SEQ)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7.5,
    'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'text.color': INK, 'axes.labelcolor': INK,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'legend.frameon': True, 'legend.framealpha': 1.0,
    'legend.edgecolor': GRID, 'legend.borderpad': 0.4,
})

XL = 'Observed log $K_p$ (cm s$^{-1}$)'
YL = 'Predicted log $K_p$ (cm s$^{-1}$)'

def grid(ax, axis='both'):
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    if axis != 'both':
        ax.grid(axis=('y' if axis == 'x' else 'x'), visible=False)
    ax.set_axisbelow(True)

def panel(ax, letter, dx=-0.20, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left', color=INK)

def box(ax, s, loc='upper left'):
    x, y, va, ha = (0.035, 0.965, 'top', 'left') if loc == 'upper left' else (0.965, 0.045, 'bottom', 'right')
    ax.text(x, y, s, transform=ax.transAxes, va=va, ha=ha, fontsize=6.8, color=INK,
            bbox=dict(fc='white', ec=GRID, lw=0.5, boxstyle='round,pad=0.32'), zorder=6)

# Pass "--only Fig4" to regenerate a single figure and leave the other TIFFs on
# disk untouched (they were rendered earlier and should not be churned).
ONLY = None
if '--only' in sys.argv:
    ONLY = sys.argv[sys.argv.index('--only') + 1]

def save(fig, name):
    if ONLY is not None and name != ONLY:
        plt.close(fig)
        print('%-6s skipped (--only %s)' % (name, ONLY))
        return
    p = os.path.join(OUT, name + '.tiff')
    fig.savefig(p, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    im = Image.open(p)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
        bg.save(p, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
        im = Image.open(p)
    pv = im.copy(); pv.thumbnail((1000, 1000))
    pv.save(os.path.join(OUT, 'prev_' + name + '.png'), optimize=True)
    print('%-6s %s  %.2f MB' % (name, im.size, os.path.getsize(p) / 1e6))

# =====================================================================
# Fig. 1  Model behaviour on the external test set
# =====================================================================
f1 = pd.read_csv(os.path.join(DATA, 'f1_external.csv'))
m1 = json.load(open(os.path.join(DATA, 'f1_meta.json')))
fig, axs = plt.subplots(2, 2, figsize=(6.9, 6.2))
lims = (min(f1.obs.min(), f1.pg.min()) - 0.5, max(f1.obs.max(), f1.pg.max()) + 0.5)

for ax, (nm, col, c, r2, rmse) in zip(
        axs[0], [('Random forest', 'rf', BLUE, m1['rf_r2'], m1['rf_rmse']),
                 ('Potts–Guy (1992)', 'pg', ORANGE, m1['pg_r2'], m1['pg_rmse'])]):
    grid(ax)
    ax.fill_between(lims, [lims[0] - 1, lims[1] - 1], [lims[0] + 1, lims[1] + 1],
                    color=GRID, alpha=0.45, lw=0, zorder=1)
    ax.plot(lims, lims, color=INK2, lw=0.9, ls='--', zorder=2)
    ax.scatter(f1.obs, f1[col], s=24, facecolor=c, edgecolor='white',
               linewidth=0.6, alpha=0.92, zorder=3)
    box(ax, '%s\n$R^2$ = %.3f\nRMSE = %.3f' % (nm, r2, rmse))
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
    ax.set_xlabel(XL); ax.set_ylabel(YL)
panel(axs[0, 0], 'a'); panel(axs[0, 1], 'b')

# (c) residual vs predicted
ax = axs[1, 0]; grid(ax)
ax.axhline(0, color=INK2, lw=0.9, ls='--', zorder=2)
ax.scatter(f1.rf, f1.res_rf, s=22, marker='o', facecolor=BLUE, edgecolor='white',
           linewidth=0.55, label='Random forest', zorder=3)
ax.scatter(f1.pg, f1.res_pg, s=22, marker='s', facecolor=ORANGE, edgecolor='white',
           linewidth=0.55, label='Potts–Guy (1992)', zorder=3)
for col, c, ls in [('res_rf', BLUE, '-'), ('res_pg', ORANGE, '-')]:
    xs = f1.rf if col == 'res_rf' else f1.pg
    o = np.argsort(xs.values); xv = xs.values[o]; yv = f1[col].values[o]
    k = max(9, len(xv) // 6)
    sm = pd.Series(yv).rolling(k, center=True, min_periods=3).mean().values
    ax.plot(xv, sm, color=c, lw=1.4, ls=ls, zorder=4)
ax.text(0.5, 0.06, 'mean bias  RF %+.2f   Potts–Guy %+.2f'
        % (f1.res_rf.mean(), f1.res_pg.mean()), transform=ax.transAxes,
        ha='center', fontsize=6.8, color=INK2)
ax.set_xlabel(YL); ax.set_ylabel('Residual (observed − predicted)')
ax.legend(loc='upper right')
panel(ax, 'c')

# (d) ECDF of absolute error
ax = axs[1, 1]; grid(ax)
for col, c, nm, ls in [('rf', BLUE, 'Random forest', '-'),
                       ('pg', ORANGE, 'Potts–Guy (1992)', '--')]:
    e = np.sort(np.abs(f1.obs - f1[col]).values)
    ax.step(e, np.arange(1, len(e) + 1) / len(e), where='post', color=c, lw=1.6,
            ls=ls, label=nm, zorder=3)
    med = np.median(e)
    ax.plot([med, med], [0, 0.5], color=c, lw=0.8, ls=':', zorder=2)
    ax.plot([med], [0.5], marker='o', ms=4.5, color=c, mec='white', mew=0.8, zorder=4)
ax.axhline(0.5, color=INK3, lw=0.6, ls=':', zorder=1)
ax.set_xlabel('Absolute error (log units)')
ax.set_ylabel('Cumulative proportion of compounds')
ax.set_ylim(0, 1.0); ax.set_xlim(left=0)
box(ax, 'median |error|\nRF 0.516  vs  Potts–Guy 0.745\nWilcoxon signed-rank, p = 0.006', loc='lower right')
ax.legend(loc='upper left')
panel(ax, 'd')
fig.tight_layout(w_pad=2.2, h_pad=2.0)
save(fig, 'Fig1')

# =====================================================================
# Fig. 2  What the model learned
# =====================================================================
imp = pd.read_csv(os.path.join(DATA, 'f2_importance.csv')).head(10).iloc[::-1]
pd1 = pd.read_csv(os.path.join(DATA, 'f2_pd1d.csv'))
pts = pd.read_csv(os.path.join(DATA, 'f2_points.csv'))
z = np.load(os.path.join(DATA, 'f2_pd2d.npz'))
fig, axs = plt.subplots(2, 2, figsize=(6.9, 6.0))

ax = axs[0, 0]; grid(ax, axis='x')
ax.barh(imp.descriptor, imp.imp_mean, xerr=imp.imp_sd, height=0.66, color=BLUE,
        error_kw=dict(ecolor=INK2, lw=0.7, capsize=2), zorder=3)
ax.axvline(0, color=INK2, lw=0.7)
for i, (v, s) in enumerate(zip(imp.imp_mean, imp.imp_sd)):
    ax.text(v + s + 0.008, i, '%.3f' % (0.0 if abs(v) < 5e-4 else v), va='center',
            fontsize=6.2, color=INK2)
ax.set_xlim(right=float(imp.imp_mean.max() + imp.imp_sd.max()) + 0.08)
ax.set_xlabel('Decrease in $R^2$ when permuted')
panel(ax, 'a', dx=-0.34)

for ax, feat, xlab in [(axs[0, 1], 'logP', 'Crippen log $P$'),
                       (axs[1, 0], 'MW', 'Molecular weight (g mol$^{-1}$)')]:
    grid(ax)
    s = pd1[pd1.descriptor == feat]
    ax.plot(s.x, s.yhat, color=BLUE, lw=1.8, zorder=4)
    v = pts[feat].values
    ax.set_xlim(float(s.x.min()), float(s.x.max()))
    yl = ax.get_ylim()
    ax.plot(v, np.full(len(v), yl[0] + 0.02 * (yl[1] - yl[0])), '|', color=INK3,
            ms=4, mew=0.6, alpha=0.55, zorder=2)
    ax.set_ylim(yl)
    ax.set_xlabel(xlab); ax.set_ylabel('Partial dependence of log $K_p$')
panel(axs[0, 1], 'b'); panel(axs[1, 0], 'c')

ax = axs[1, 1]
gx, gy, Zg = z['logP'], z['MW'], z['z']
im = ax.pcolormesh(gx, gy, Zg.T, cmap=CMAP, shading='gouraud', zorder=1)
cs = ax.contour(gx, gy, Zg.T, levels=6, colors=INK2, linewidths=0.55, zorder=3)
ax.clabel(cs, inline=True, fontsize=5.6, fmt='%.1f')
inside = ((pts.logP >= gx.min()) & (pts.logP <= gx.max())
          & (pts.MW >= gy.min()) & (pts.MW <= gy.max()))
ax.scatter(pts.loc[inside, 'logP'], pts.loc[inside, 'MW'], s=5, facecolor='white',
           edgecolor=INK2, linewidth=0.35, alpha=0.75, zorder=4)
ax.set_xlim(gx.min(), gx.max())
ax.set_ylim(gy.min(), gy.max())
ax.set_xlabel('Crippen log $P$'); ax.set_ylabel('Molecular weight (g mol$^{-1}$)')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label('Predicted log $K_p$ (cm s$^{-1}$)', fontsize=6.8)
cb.ax.tick_params(labelsize=6.2, width=0.6, length=2)
cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor(INK2)
panel(ax, 'd')
fig.tight_layout(w_pad=2.4, h_pad=2.2)
save(fig, 'Fig2')

# =====================================================================
# Fig. 3  Analgesic subgroup, leave-analgesics-out
# =====================================================================
an = pd.read_csv(os.path.join(DATA, 'f3_analgesics.csv'))
PCOL = 'pred_PG-form refit (logP, MW)'
STY = {'Opioid': (BLUE, 'o'), 'NSAID / non-opioid': (ORANGE, 's'),
       'Local anaesthetic': (AQUA, '^')}
fig = plt.figure(figsize=(6.9, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.10], height_ratios=[1.55, 1.0],
                      wspace=0.10, hspace=0.42)

ax = fig.add_subplot(gs[0, 0]); grid(ax)
lo = min(an.logkp.min(), an[PCOL].min()) - 0.5
hi = max(an.logkp.max(), an[PCOL].max()) + 0.9
ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color=GRID, alpha=0.45, lw=0, zorder=1)
ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls='--', zorder=2)
rep = an[an.n_rec > 1]
ax.errorbar(rep.logkp, rep[PCOL],
            xerr=[rep.logkp - rep.logkp_min, rep.logkp_max - rep.logkp],
            fmt='none', ecolor=INK3, elinewidth=0.7, capsize=1.6, zorder=3)
for k, (c, m) in STY.items():
    s = an[an.klass == k]
    ax.scatter(s.logkp, s[PCOL], s=40, marker=m, facecolor=c, edgecolor='white',
               linewidth=0.7, label='%s (n = %d)' % (k, len(s)), zorder=4)
# Labels are pulled clear of the cluster and tied to their point by a leader
# line, so none of them sits on top of another or on a marker (Reviewer 2).
for lab, off, ha in [('Morphine',   ( 20, -26), 'left'),
                     ('Ibuprofen',  (-44,  20), 'right'),
                     ('Lidocaine',  ( 10,  34), 'left'),
                     ('Sufentanil', ( 30,  14), 'left'),
                     ('Fentanyl',   ( 34, -20), 'left')]:
    r = an[an.name.str.lower() == lab.lower()]
    if len(r):
        ax.annotate(lab, (r.logkp.iloc[0], r[PCOL].iloc[0]), textcoords='offset points',
                    xytext=off, ha=ha, va='center', fontsize=6.2, color=INK2, zorder=7,
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.8),
                    clip_on=False,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color=INK3,
                                    shrinkA=0.5, shrinkB=3.5))
box(ax, '$R^2$ = 0.361\nRMSE = 0.781\nn = 40, all within\napplicability domain')
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
ax.set_xlabel(XL); ax.set_ylabel(YL)
ax.legend(loc='lower right')
panel(ax, 'a', dx=-0.17)

ax = fig.add_subplot(gs[1, 0]); grid(ax, axis='y')
an['abserr'] = (an.logkp - an[PCOL]).abs()
rng = np.random.default_rng(7)
for i, (k, (c, m)) in enumerate(STY.items()):
    s = an[an.klass == k]
    ax.scatter(np.full(len(s), i) + rng.uniform(-0.14, 0.14, len(s)), s.abserr,
               s=26, marker=m, facecolor=c, edgecolor='white', linewidth=0.6, zorder=3)
    med = float(s.abserr.median())
    ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.4, zorder=4)
    ax.text(i + 0.32, med, '%.2f' % med, ha='left', va='center', fontsize=6.2,
            color=INK, zorder=5)
ax.set_xticks(range(3))
ax.set_xticklabels(['Opioid\n(n = 12)', 'NSAID /\nnon-opioid\n(n = 23)',
                    'Local\nanaesthetic\n(n = 5)'], fontsize=6.4)
ax.set_ylabel('Absolute error (log units)')
ax.set_xlim(-0.55, 2.55); ax.set_ylim(bottom=0)
ax.text(0.98, 0.95, 'horizontal bars: class median', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.0, color=INK2)
panel(ax, 'c', dx=-0.17)

ax = fig.add_subplot(gs[:, 1]); grid(ax, axis='x')
d = an.sort_values('logkp').reset_index(drop=True)
yy = np.arange(len(d))
for i, r in d.iterrows():
    ax.plot([r.logkp, r[PCOL]], [i, i], color=INK3, lw=0.8, zorder=2)
ax.errorbar(d.logkp, yy, xerr=[d.logkp - d.logkp_min, d.logkp_max - d.logkp],
            fmt='none', ecolor=GRID, elinewidth=2.6, zorder=1)
for k, (c, m) in STY.items():
    s = d[d.klass == k]
    ax.scatter(s.logkp, s.index, s=30, marker=m, facecolor=c, edgecolor='white',
               linewidth=0.6, zorder=4)
ax.scatter(d[PCOL], yy, s=22, marker='D', facecolor='white', edgecolor=INK2,
           linewidth=0.8, zorder=3)
ax.yaxis.tick_right()
ax.set_yticks(yy)
ax.set_yticklabels([n if len(n) < 30 else n[:28] + '…' for n in d.name], fontsize=5.6)
ax.tick_params(axis='y', length=0, pad=2)
ax.set_ylim(-0.8, len(d) - 0.2)
ax.set_xlabel('log $K_p$ (cm s$^{-1}$)')
hand = [Line2D([], [], ls='', marker=m, mfc=c, mec='white', mew=0.6, ms=5.5, label=k)
        for k, (c, m) in STY.items()]
hand += [Line2D([], [], ls='', marker='D', mfc='white', mec=INK2, mew=0.8, ms=4.6,
                label='Predicted'),
         Line2D([], [], color=GRID, lw=2.6, label='Range across studies')]
ax.legend(handles=hand, loc='upper left', fontsize=6.0)
panel(ax, 'b', dx=-0.06, dy=1.03)
save(fig, 'Fig3')

# =====================================================================
# Fig. 4  Where the limit lies
# =====================================================================
vd = pd.read_csv(os.path.join(DATA, 'f4_variance.csv'))
ly = pd.read_csv(os.path.join(DATA, 'f4_layers.csv'))
rp = pd.read_csv(os.path.join(DATA, 'f4_repro.csv'))
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.9))

# Per-fold values recovered by scripts/13_foldlevel.py, which re-runs the
# computation behind 08_stats.py under the same seed and aborts unless every
# published aggregate reproduces exactly. Showing them turns the bars into a
# visible distribution rather than a mean that hides its own spread.
fa = pd.read_csv(os.path.join(DATA, 'f4a_folds.csv'))
fb = pd.read_csv(os.path.join(DATA, 'f4b_folds.csv'))
# (a) and (b) plot the same quantity, so they share one y-scale; otherwise the
# bar heights invite a comparison the differing axes would distort.
R2LO = math.floor(min(fa[['chemistry', 'protocol', 'both']].values.min(),
                      fb[['epidermis', 'other_mixed']].values.min()) * 10) / 10
R2HI = 0.95

ax = axs[0]; grid(ax, axis='y')
lab = ['Molecular\ndescriptors', 'Experimental\nprotocol', 'Both']
cols3 = [BLUE, ORANGE, AQUA]
ax.bar(range(3), vd.R2, width=0.62, color=cols3, alpha=0.30, zorder=2)
ax.errorbar(range(3), vd.R2, yerr=vd.R2_sd, fmt='none', ecolor=INK2, lw=0.8, capsize=3, zorder=5)
# the three models were scored on the same ten folds, so join each fold across
# conditions: the paired Wilcoxon test is a statement about these lines
paired = fa[['chemistry', 'protocol', 'both']].values
for row in paired:
    ax.plot(range(3), row, color=INK3, lw=0.4, alpha=0.55, zorder=3)
rng = np.random.default_rng(20260821)
for i, c in enumerate(['chemistry', 'protocol', 'both']):
    xj = i + rng.uniform(-0.10, 0.10, len(fa))
    ax.scatter(xj, fa[c], s=9, color=cols3[i], edgecolor='white', linewidth=0.25,
               alpha=0.95, zorder=4)
for i, v in enumerate(vd.R2):
    ax.plot([i - 0.31, i + 0.31], [v, v], color=INK, lw=1.3, zorder=6)
    ax.text(i, 0.845, '%.3f' % v, ha='center', fontsize=6.6, color=INK, zorder=7)
ax.set_xticks(range(3)); ax.set_xticklabels(lab)
ax.set_ylabel('Cross-validated $R^2$')
ax.set_ylim(R2LO, R2HI)
ax.annotate('', xy=(2, 0.78), xytext=(0, 0.78),
            arrowprops=dict(arrowstyle='-', lw=0.7, color=INK2))
ax.text(1, 0.792, '$p$ = 0.006', ha='center', fontsize=6.4, color=INK)
ax.text(0.985, 0.02, 'lines join the same fold', transform=ax.transAxes, ha='right',
        va='bottom', fontsize=5.8, color=INK3)
panel(ax, 'a', dx=-0.30)

ax = axs[1]; grid(ax, axis='y')
nm = {'epidermis': 'Epidermis\nonly', 'other/mixed': 'Mixed /\nunspecified'}
cols2 = [BLUE, ORANGE]
fbcols = ['epidermis', 'other_mixed']
ax.bar(range(len(ly)), ly.R2, width=0.52, color=cols2, alpha=0.30, zorder=2)
ax.errorbar(range(len(ly)), ly.R2, yerr=ly.R2_sd, fmt='none', ecolor=INK2, lw=0.8,
            capsize=3, zorder=5)
for i, c in enumerate(fbcols):
    xj = i + rng.uniform(-0.13, 0.13, len(fb))
    ax.scatter(xj, fb[c], s=7, color=cols2[i], edgecolor='none', alpha=0.55, zorder=3)
for i, v in enumerate(ly.R2):
    ax.plot([i - 0.26, i + 0.26], [v, v], color=INK, lw=1.3, zorder=6)
    ax.text(i, 0.845, '%.3f' % v, ha='center', fontsize=6.6, color=INK, zorder=7)
ax.set_xticks(range(len(ly)))
ax.set_xticklabels(['%s\n(n = %d)' % (nm.get(x, x), n) for x, n in zip(ly.layer, ly.n_compounds)])
ax.set_ylabel('Cross-validated $R^2$')
ax.set_ylim(R2LO, R2HI)
ax.annotate('', xy=(1, 0.78), xytext=(0, 0.78), arrowprops=dict(arrowstyle='-', lw=0.7, color=INK2))
ax.text(0.5, 0.792, '$p$ = 0.000004', ha='center', fontsize=6.4, color=INK)
ax.text(0.985, 0.02, '50 folds per stratum', transform=ax.transAxes, ha='right',
        va='bottom', fontsize=5.8, color=INK3)
panel(ax, 'b', dx=-0.30)

ax = axs[2]; grid(ax, axis='y')
sp = rp.spread.dropna().values
ax.hist(sp, bins=np.arange(0, max(sp) + 0.5, 0.4), color=BLUE, edgecolor='white',
        linewidth=0.6, zorder=3)
med = float(np.median(sp))
ax.axvline(med, color=INK, lw=1.1, zorder=4)
ax.axvline(0.981, color=ORANGE, lw=1.1, ls='--', zorder=4)
ax.set_xlabel('Between-study spread in log $K_p$\n(log units, n = %d compounds)' % len(sp))
ax.set_ylabel('Number of compounds')
hand = [Line2D([], [], color=INK, lw=1.1, label='Median spread %.2f' % med),
        Line2D([], [], color=ORANGE, lw=1.1, ls='--', label='Best model RMSE 0.981')]
ax.legend(handles=hand, loc='upper right', fontsize=6.0)
panel(ax, 'c', dx=-0.30)
fig.tight_layout(w_pad=2.0)
save(fig, 'Fig4')
print('\nfigures written to', OUT)
