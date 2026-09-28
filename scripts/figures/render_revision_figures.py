#!/usr/bin/env python3
"""Render the three supplementary figures for the analyses added during revision.

Reviewer 2 asked for four checks and a confounding test; the results were
reported as numbers only, which makes them hard to weigh. These show them.

  Fig. S6  Major 1 - the four treatments of repeated measurements side by side,
           with the mixed-effects variance components that the ceiling argument
           rests on
  Fig. S7  Major 3 - the distribution of held-out performance over 200 splits,
           with the single split reported in Table 2 located inside it
  Fig. S8  Major 5 - the permutation null for the skin-layer gap, and the
           covariate balance achieved by nearest-neighbour matching

Nothing is refitted. Every value is read from the result files written by
scripts 15, 16 and 18. Palette, font, panel lettering and 600 dpi output follow
render_supp_figures.py so the supplement stays visually of a piece.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REV = os.path.join(ROOT, 'revision')
OUT = os.path.join(HERE, 'submission', 'figures')
DPI = 600
os.makedirs(OUT, exist_ok=True)

assert 'Arial' in {f.name for f in matplotlib.font_manager.fontManager.ttflist}, 'Arial not found'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, INK3, GRID = '#0b0b0b', '#52514e', '#8a8880', '#dedcd6'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 7.5,
    'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.edgecolor': INK2, 'axes.linewidth': 0.7,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'text.color': INK, 'axes.labelcolor': INK,
    'figure.dpi': DPI, 'savefig.dpi': DPI,
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
    x, y, va, ha = {'upper left': (0.035, 0.965, 'top', 'left'),
                    'upper right': (0.965, 0.965, 'top', 'right'),
                    'lower right': (0.965, 0.045, 'bottom', 'right'),
                    'lower left': (0.035, 0.045, 'bottom', 'left')}[loc]
    ax.text(x, y, s, transform=ax.transAxes, va=va, ha=ha, fontsize=6.6, color=INK,
            bbox=dict(fc='white', ec=GRID, lw=0.5, boxstyle='round,pad=0.32'), zorder=6)


def save(fig, name):
    p = os.path.join(OUT, name + '.tiff')
    fig.savefig(p, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    im = Image.open(p)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
        bg.save(p, format='TIFF', compression='tiff_lzw', dpi=(DPI, DPI))
    else:
        im.save(p, format='TIFF', compression='tiff_lzw', dpi=(DPI, DPI))
    pv = Image.open(p).copy()
    pv.thumbnail((1000, 1000))
    pv.save(os.path.join(OUT, 'prev_' + name + '.png'), optimize=True)
    with Image.open(p) as chk:
        assert round(chk.info.get('dpi', (0, 0))[0]) == DPI, '%s not %d dpi' % (name, DPI)
    print('wrote %s  %dx%d' % (p, pv.width * 0 + Image.open(p).width, Image.open(p).height))


SHORT = {'Potts-Guy (literature)': 'Potts–Guy\npublished',
         'PG-form refit (logP, MW)': 'Potts–Guy\nrefitted',
         'MLR (all descriptors)': 'MLR',
         'Ridge (all descriptors)': 'Ridge',
         'Random forest': 'Random\nforest',
         'Gradient boosting (XGB)': 'Gradient\nboosting'}
ORDER = ['Potts-Guy (literature)', 'PG-form refit (logP, MW)', 'MLR (all descriptors)',
         'Ridge (all descriptors)', 'Random forest', 'Gradient boosting (XGB)']

# =========================================================
# Fig. S6 -- Major 1: how the treatment of repeated measurements moves the results
# =========================================================
sc = pd.read_csv(os.path.join(REV, 'major1', 'scheme_comparison.csv'))
me = pd.read_csv(os.path.join(REV, 'major1', 'mixed_effects.csv'), index_col=0)

SCHEMES = [('A_median', 'Median per compound', BLUE),
           ('B_mean', 'Mean per compound', AQUA),
           ('C_record', 'Record level, grouped CV', ORANGE)]

fig, axs = plt.subplots(2, 2, figsize=(6.9, 5.6))
x = np.arange(len(ORDER))
w = 0.26

for ax, col, lab, ltr in [(axs[0, 0], 'cv_R2', 'Cross-validated $R^2$', 'a'),
                          (axs[0, 1], 'ho_R2', 'Held-out $R^2$', 'b'),
                          (axs[1, 0], 'cv_RMSE', 'Cross-validated RMSE (log units)', 'c')]:
    for i, (key, name, colour) in enumerate(SCHEMES):
        sub = sc[sc.scheme == key].set_index('model')
        vals = [sub.loc[m, col] for m in ORDER]
        ax.bar(x + (i - 1) * w, vals, width=w, color=colour, zorder=3,
               label=name if ltr == 'a' else None)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=6.2)
    ax.set_ylabel(lab)
    if col != 'cv_RMSE':
        ax.axhline(0, color=INK3, lw=0.7)
    grid(ax, axis='y')
    panel(ax, ltr)
axs[0, 0].legend(loc='upper left', fontsize=6.2)

# (d) variance components: what a structure-based description can and cannot reach
ax = axs[1, 1]
rows = ['null (intercept only)', 'PG-form (logP, MW)', 'MLR (all descriptors)']
labels = ['Intercept\nonly', 'Two\ndescriptors', 'All fifteen\ndescriptors']
bet = [me.loc[r, 'between'] for r in rows]
wit = [me.loc[r, 'within'] for r in rows]
xx = np.arange(len(rows))
ax.bar(xx, bet, width=0.55, color=BLUE, zorder=3, label='Between compounds')
ax.bar(xx, wit, width=0.55, bottom=bet, color=ORANGE, zorder=3,
       label='Within compounds (repeat measurements)')
for i, r in enumerate(rows):
    ax.text(i, bet[i] / 2, '%.2f' % bet[i], ha='center', va='center',
            fontsize=6.4, color='white', zorder=5)
    ax.text(i, bet[i] + wit[i] / 2, '%.2f' % wit[i], ha='center', va='center',
            fontsize=6.4, color='white', zorder=5)
ax.set_xticks(xx)
ax.set_xticklabels(labels, fontsize=6.2)
ax.set_ylabel('Variance of log $K_p$')
ax.set_ylim(0, max(np.array(bet) + np.array(wit)) * 1.42)
ax.legend(loc='upper right', fontsize=6.0)
box(ax, 'ICC %.3f: %.0f%% of record-level\nvariance is within compounds'
    % (me.loc['null (intercept only)', 'icc'],
       100 * (1 - me.loc['null (intercept only)', 'icc'])), loc='upper left')
grid(ax, axis='y')
panel(ax, 'd')
fig.tight_layout(w_pad=2.2, h_pad=2.2)
save(fig, 'FigS6')

# =========================================================
# Fig. S7 -- Major 3: performance on one split is a draw from a distribution
# =========================================================
r2 = pd.read_csv(os.path.join(REV, 'm3_m4', 'm3_heldout_R2_by_split.csv'))
summ = pd.read_csv(os.path.join(REV, 'm3_m4', 'm3_summary.csv'), index_col=0)
REPORTED = 0.312                      # the single split reported in Table 2

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))

ax = axs[0]
v = r2['Random forest'].values
ax.hist(v, bins=26, color=BLUE, alpha=0.85, zorder=3)
med = float(np.median(v))
lo, hi = np.percentile(v, [2.5, 97.5])
pct = 100.0 * (v < REPORTED).mean()
for xv, c, ls, lab in [(med, INK, '-', 'median %.3f' % med),
                       (REPORTED, ORANGE, '--', 'reported split %.3f' % REPORTED)]:
    ax.axvline(xv, color=c, lw=1.3, ls=ls, zorder=5, label=lab)
ax.axvspan(lo, hi, color=BLUE, alpha=0.10, lw=0, zorder=1,
           label='2.5–97.5%% (%.3f to %.3f)' % (lo, hi))
ax.set_xlabel('Held-out $R^2$, random forest')
ax.set_ylabel('Splits (of 200)')
ax.legend(loc='upper left', fontsize=6.0)
box(ax, 'the reported split sits at\nthe %.0fth percentile' % pct, loc='lower right')
grid(ax, axis='y')
panel(ax, 'a')

ax = axs[1]
order = list(summ.sort_values('mean_rank').index)
data = [r2[m].values for m in order]
bp = ax.boxplot(data, orientation='horizontal', widths=0.6, patch_artist=True, showfliers=False)
for i, b in enumerate(bp['boxes']):
    b.set(facecolor=(BLUE if order[i] == 'Random forest' else '#c9d9ef'),
          edgecolor=INK2, linewidth=0.7)
for k in ('whiskers', 'caps', 'medians'):
    for e in bp[k]:
        e.set(color=INK2, linewidth=0.8)
ax.axvline(0, color=INK3, lw=0.7, zorder=1)
ax.set_yticklabels([SHORT[m].replace('\n', ' ') for m in order], fontsize=6.4)
ax.set_xlabel('Held-out $R^2$ across 200 splits')
for i, m in enumerate(order):
    ax.text(0.995, i + 1, 'first in %.0f%%' % summ.loc[m, 'rank1_pct'],
            transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontsize=6.0, color=INK2)
ax.set_xlim(right=1.08)
grid(ax, axis='y')
panel(ax, 'b', dx=-0.34)
fig.tight_layout(w_pad=2.4)
save(fig, 'FigS7')

# =========================================================
# Fig. S8 -- Major 5: is the skin-layer gap a chemical-space artefact?
# =========================================================
null = pd.read_csv(os.path.join(REV, 'm5', 'm5_permutation_null.csv'))['null_gap'].values
m5 = json.load(open(os.path.join(REV, 'm5', 'm5_results.json')))
bal = pd.read_csv(os.path.join(REV, 'm5', 'm5_covariate_balance.csv'))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.9))

ax = axs[0]
ax.hist(null, bins=24, color='#c9d9ef', edgecolor=INK2, linewidth=0.4, zorder=3,
        label='null: labels reassigned (n = %d)' % m5['permutation']['n'])
obs = m5['observed']['gap']
ax.axvline(obs, color=ORANGE, lw=1.6, zorder=5, label='observed gap %.3f' % obs)
ax.axvline(m5['permutation']['null_lo'], color=INK3, lw=0.8, ls=':', zorder=4)
ax.axvline(m5['permutation']['null_hi'], color=INK3, lw=0.8, ls=':', zorder=4,
           label='null 2.5–97.5%')
ax.set_xlabel('Epidermis − mixed gap in cross-validated $R^2$')
ax.set_ylabel('Permutations')
ax.legend(loc='upper left', fontsize=6.0)
box(ax, 'two-sided p = %.3f\nthe gap lies inside the null'
    % m5['permutation']['p'], loc='upper right')
grid(ax, axis='y')
panel(ax, 'a')

ax = axs[1]
b = bal.copy()
b['abs_before'] = b.smd_before.abs()
b = b.sort_values('abs_before', ascending=True)
y = np.arange(len(b))
for yi, (bef, aft) in enumerate(zip(b.smd_before.values, b.smd_after.values)):
    ax.plot([bef, aft], [yi, yi], color=GRID, lw=1.2, zorder=2)
ax.scatter(b.smd_before, y, s=18, color=INK3, zorder=3, label='before matching')
ax.scatter(b.smd_after, y, s=18, color=BLUE, zorder=4, label='after matching')
ax.axvline(0, color=INK3, lw=0.7, zorder=1)
for t in (-0.1, 0.1):
    ax.axvline(t, color=ORANGE, lw=0.7, ls=':', zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(b.descriptor.values, fontsize=6.0)
ax.set_ylim(-0.8, len(b) - 1 + 2.1)
ax.set_xlabel('Standardised mean difference between strata')
ax.legend(loc='lower right', fontsize=6.0)
box(ax, '%d matched pairs; gap %.3f after matching'
    % (m5['matched']['n_pairs'], m5['matched']['gap']), loc='upper left')
grid(ax, axis='y')
panel(ax, 'b', dx=-0.30)
fig.tight_layout(w_pad=2.4)
save(fig, 'FigS8')

print('done')
