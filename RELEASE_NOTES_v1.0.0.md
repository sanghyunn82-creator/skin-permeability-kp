# v1.0.0 — deposit accompanying the revised manuscript

Archived release for the revision of *Human Skin Permeability Prediction by Machine Learning for
Transdermal Drug Delivery* (Medicina, MDPI).

## What this release adds over the original submission

**Split and fold assignments** (`deposit/`) so that the held-out split and every cross-validation
fold can be reproduced exactly rather than re-estimated, together with the seeds and package
versions actually used.

**The analyses added at revision** (`scripts/15`–`19`, `22`, results in `derived/revision/`):

| Analysis | Answers |
|---|---|
| Aggregation scheme comparison and mixed-effects variance decomposition | median vs mean vs record level |
| Accumulated local effects with 200 bootstrap refits | replaces partial dependence under collinearity |
| 200 repeated stratified splits | the distribution behind a single reported split |
| Nadeau–Bengio corrected fold-wise tests | folds are not independent samples |
| Permutation null and nearest-neighbour matching on skin layer | chemical-space control |
| Ionization at pH 7.4 (Dimorphite-DL) | whether a missing account of ionization is the limit |
| ATC verification via RxClass | replaces name-matching against a term list |

**The per-panel inputs** to every main-text figure (`derived/figdata/`), and the generators that
actually produced the submitted figures (`scripts/figures/`).

**A statement of conformity to the five OECD principles for QSAR validation**
(`deposit/OECD_QSAR_validation_statement.md`).

## Two claims withdrawn or weakened at revision

1. **The molecular-weight threshold near 220 g mol⁻¹ is withdrawn as a data-supported finding.**
   Its accumulated-local-effects span is 0.44 log units against a mean bootstrap band width of
   0.54; it carries no information independent of the Labute surface area (partial r = 0.008,
   p = 0.88); removing it changes cross-validated R² from 0.364 to 0.366. It is now reported as
   model-derived only.
2. **The epidermis-only performance difference loses significance** once fold dependence is
   corrected (p 0.000004 → 0.059) and lies inside a permutation null (p = 0.184), though it
   survives nearest-neighbour matching on descriptors (gap 0.203). It is reported as large but
   statistically unestablished.

## Figures

Figures are now 1–5 and S1–S8 at 600 dpi, matching the revised manuscript. The accumulated local
effects figure enters the main text as Figure 3, so the previous Figures 3 and 4 become 4 and 5.
