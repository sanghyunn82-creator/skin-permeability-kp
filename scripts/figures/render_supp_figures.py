#!/usr/bin/env python3
"""Render five supplementary figures from already-computed results/ and data/
CSVs (no new modelling, no re-fitting). Reuses the exact palette, font and
panel-lettering conventions of render_figures.py (Fig. 1-4) for visual
consistency across the whole figure set.

Run on the pod (real Arial.ttf is installed under
/usr/share/fonts/truetype/arial/ here; render_figures.py's Mac font path does
not apply on this machine).
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'submission', 'figures')
os.makedirs(OUT, exist_ok=True)

assert 'Arial' in {f.name for f in matplotlib.font_manager.fontManager.ttflist}, 'Arial not found'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8880', '#dedcd6'
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
       '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('seqblue', SEQ)
# Diverging map built from the same two validated categorical hues (BLUE/ORANGE)
# rather than an unvalidated third colormap, for the descriptor-correlation panel.
DIV = LinearSegmentedColormap.from_list('orangewhiteblue', [ORANGE, '#ffffff', BLUE])

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

def save(fig, name):
    p = os.path.join(OUT, name + '.tiff')
    fig.savefig(p, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    im = Image.open(p)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
        bg.save(p, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
    pv = Image.open(p).copy(); pv.thumbnail((1000, 1000))
    pv.save(os.path.join(ROOT, 'figs', 'prev_' + name + '.png'))
    print('wrote', p)

XL = 'Observed log $K_p$ (cm s$^{-1}$)'
YL = 'Predicted log $K_p$ (cm s$^{-1}$)'

# =========================================================
# Fig. S1 -- diversity of experimental protocol across the 732 records
# =========================================================
rec = pd.read_csv(os.path.join(ROOT, 'data', 'records_with_descriptors.csv'))

fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.4))

ax = axs[0, 0]
layer_counts = rec['layer'].fillna('not reported').value_counts().head(7)
ax.barh(range(len(layer_counts)), layer_counts.values[::-1], color=BLUE, height=0.65, zorder=3)
ax.set_yticks(range(len(layer_counts)))
ax.set_yticklabels(layer_counts.index[::-1], fontsize=6.6)
ax.set_xlabel('Records (n)')
grid(ax, axis='y'); panel(ax, 'a')

ax = axs[0, 1]
cell_counts = rec['cell_type'].fillna('not reported').value_counts().head(8)
ax.barh(range(len(cell_counts)), cell_counts.values[::-1], color=AQUA, height=0.65, zorder=3)
ax.set_yticks(range(len(cell_counts)))
ax.set_yticklabels(cell_counts.index[::-1], fontsize=6.4)
ax.set_xlabel('Records (n)')
grid(ax, axis='y'); panel(ax, 'b')

ax = axs[1, 0]
t = rec['donor_temp'].dropna()
ax.hist(t, bins=14, color=ORANGE, edgecolor='white', linewidth=0.4, zorder=3)
ax.set_xlabel('Donor temperature (°C)'); ax.set_ylabel('Records (n)')
box(ax, 'n = %d\nmissing in %d records' % (len(t), rec['donor_temp'].isna().sum()))
grid(ax); panel(ax, 'c')

ax = axs[1, 1]
p = rec['donor_ph'].dropna()
ax.hist(p, bins=14, color=BLUE, edgecolor='white', linewidth=0.4, zorder=3)
ax.set_xlabel('Donor pH'); ax.set_ylabel('Records (n)')
box(ax, 'n = %d\nmissing in %d records' % (len(p), rec['donor_ph'].isna().sum()))
grid(ax); panel(ax, 'd')

fig.tight_layout(w_pad=2.2, h_pad=1.8)
save(fig, 'FigS1')

# =========================================================
# Fig. S2 -- computed versus experimental octanol-water partition coefficient
# =========================================================
lc = pd.read_csv(os.path.join(ROOT, 'results', 'logp_comparison.csv'))
r = lc['logP'].corr(lc['logKow_exp'])
mad = (lc['logP'] - lc['logKow_exp']).abs().mean()

fig, ax = plt.subplots(figsize=(3.4, 3.3))
ax.scatter(lc['logKow_exp'], lc['logP'], s=16, c=BLUE, alpha=0.75,
           edgecolor='white', linewidth=0.3, zorder=3)
lo, hi = -8, 8
ax.plot([lo, hi], [lo, hi], color=INK3, linewidth=0.9, linestyle='--', zorder=2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel('Experimental log $K_{ow}$ (Cheruvu et al.)')
ax.set_ylabel('Computed Crippen log $P$')
box(ax, 'n = %d\nPearson r = %.3f\nmean |difference| = %.2f' % (len(lc), r, mad))
grid(ax)
fig.tight_layout()
save(fig, 'FigS2')

# =========================================================
# Fig. S3 -- correlation structure among the fifteen 2D descriptors
# =========================================================
cm = pd.read_csv(os.path.join(ROOT, 'results', 'compounds_modelling.csv'))
DESC = ['MW', 'logP', 'MR', 'TPSA', 'HBD', 'HBA', 'RotB', 'AromRings',
        'Rings', 'HeavyAtoms', 'FracCSP3', 'LabuteASA', 'BalabanJ', 'BertzCT', 'HeteroAtoms']
corr = cm[DESC].corr(method='spearman')

fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
ax.set_xticks(range(len(DESC))); ax.set_xticklabels(DESC, rotation=90, fontsize=6.4)
ax.set_yticks(range(len(DESC))); ax.set_yticklabels(DESC, fontsize=6.4)
for i in range(len(DESC)):
    for j in range(len(DESC)):
        v = corr.values[i, j]
        if abs(v) >= 0.6 and i != j:
            ax.text(j, i, '%.2f' % v, ha='center', va='center', fontsize=5.2,
                     color='white' if abs(v) > 0.75 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Spearman's ρ", fontsize=7)
fig.tight_layout()
save(fig, 'FigS3')

# =========================================================
# Fig. S4 -- external-test observed-vs-predicted for the four models not
# shown in Fig. 1 (which already shows the random forest and the unmodified
# Potts-Guy equation)
# =========================================================
et = pd.read_csv(os.path.join(ROOT, 'results', 'external_test_predictions.csv'))
MODELS = [
    ('pred_PG-form refit (logP, MW)', 'Potts–Guy form, refitted'),
    ('pred_MLR (all descriptors)', 'Multiple linear regression'),
    ('pred_Ridge (all descriptors)', 'Ridge regression'),
    ('pred_Gradient boosting (XGB)', 'Gradient boosting'),
]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.3))
lims = (-11, -2.5)
for ax, (col, label), letter in zip(axs.flat, MODELS, 'abcd'):
    r2 = 1 - ((et['obs'] - et[col]) ** 2).sum() / ((et['obs'] - et['obs'].mean()) ** 2).sum()
    ax.plot(lims, lims, color=INK3, linewidth=0.9, linestyle='--', zorder=2)
    ax.fill_between(lims, [l - 1 for l in lims], [l + 1 for l in lims],
                     color=BLUE, alpha=0.08, zorder=1)
    colors = np.where(et['analgesic'], ORANGE, BLUE)
    ax.scatter(et['obs'], et[col], s=14, c=colors, alpha=0.75,
               edgecolor='white', linewidth=0.3, zorder=3)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(XL); ax.set_ylabel(YL)
    box(ax, '%s\nR$^2$ = %.3f' % (label, r2))
    grid(ax); panel(ax, letter)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=BLUE, markersize=6, label='non-analgesic'),
                     Line2D([0], [0], marker='o', color='none', markerfacecolor=ORANGE, markersize=6, label='analgesic')],
           loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.03, 1, 1], w_pad=2.2, h_pad=2.4)
save(fig, 'FigS4')

# =========================================================
# Fig. S5 -- applicability-domain distance of the 40 held-out analgesics
# =========================================================
an = pd.read_csv(os.path.join(ROOT, 'results', 'analgesic_predictions_classed.csv'))
meta = json.load(open(os.path.join(ROOT, 'figdata', 'f1_meta.json')))
thr = meta['ad_threshold']
CLASS_COLOR = {'Opioid': BLUE, 'NSAID / non-opioid': AQUA, 'Local anaesthetic': ORANGE}

fig, ax = plt.subplots(figsize=(4.6, 3.3))
order = ['Opioid', 'NSAID / non-opioid', 'Local anaesthetic']
for i, k in enumerate(order):
    sub = an[an['klass'] == k]
    y = np.full(len(sub), i) + np.random.default_rng(0).uniform(-0.14, 0.14, len(sub))
    ax.scatter(sub['ad_dist'], y, s=22, color=CLASS_COLOR[k], alpha=0.85,
               edgecolor='white', linewidth=0.3, zorder=3, label='%s (n = %d)' % (k, len(sub)))
ax.axvline(thr, color=INK, linewidth=0.9, linestyle='--', zorder=2)
ax.set_yticks(range(3)); ax.set_yticklabels(order, fontsize=7)
ax.set_xlabel('Mean distance to five nearest training compounds\n(standardised descriptor space)')
ax.set_xlim(0, thr * 1.6)
ax.set_ylim(-0.6, 2.6)
ax.text(thr / (thr * 1.6), 0.99, ' 95th-percentile threshold ', fontsize=6.4, color=INK,
        va='bottom', ha='center', transform=ax.transAxes,
        bbox=dict(fc='white', ec='none', pad=0.5))
grid(ax, axis='x')
box(ax, 'all 40/40 analgesics\nfall inside the domain', loc='lower right')
fig.tight_layout()
save(fig, 'FigS5')

print('done: FigS1-FigS5 written to', OUT)
