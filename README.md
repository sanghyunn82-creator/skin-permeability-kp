# Human Skin Permeability Prediction by Machine Learning for Transdermal Drug Delivery

Analysis code and derived data for the manuscript under revision at *Medicina* (MDPI).

The study pools two open human skin-permeation databases without protocol-based exclusion and
asks whether log Kp really is the linear function of lipophilicity and molecular weight that the
Potts–Guy equation assumes. It is entirely computational; **no new experimental data were
generated.**

## Source data

**The raw source databases are not redistributed here.** Both are one click from their own
publishers, under their own licences:

| Database | Licence | Where |
|---|---|---|
| HuskinDB | CC BY 4.0 | its public web service |
| SkinPiX v1.1 | Etalab 2.0 | Recherche Data Gouv, [10.57745/7FHQOY](https://doi.org/10.57745/7FHQOY) |

A third published table was used only to compare computed against measured logP; it belongs to
its authors and is not reproduced here either.

After merging, unit harmonisation and structural standardisation the pooled set is **732
permeability records for 327 compounds, of which 40 are analgesics** — that pooled table is in
`derived/`.

## Contents

| Path | What it holds |
|---|---|
| `scripts/` | The analysis pipeline, as run |
| `scripts/figures/` | The generators that actually produced the submitted figures |
| `derived/` | The pooled dataset and every table behind the manuscript |
| `derived/figdata/` | The exact inputs to each main-text figure panel |
| `derived/revision/` | Result tables for the analyses added at revision |
| `deposit/` | Split and fold assignments, reproducibility metadata, OECD QSAR statement |
| `figures/` | Figures 1–5 and Supplementary Figures S1–S8 (600 dpi TIFF, plus PNG previews) |

### `derived/`

| File | Contents |
|---|---|
| `compounds_modelling.csv` | Analysis-ready compound table — 327 compounds with descriptors |
| `records_with_descriptors.csv` | All 732 pooled permeability records with computed descriptors |
| `merged_inventory.csv` | Record-level inventory of the two source databases after harmonisation |
| `union_smiles_verified.csv` | Structure standardisation, verified SMILES per compound |
| `model_results.csv` | Cross-validated performance of every model family |
| `external_test_predictions.csv` | Observed against predicted log Kp on the held-out test set (66 compounds) |
| `analgesic_predictions.csv` | Leave-analgesics-out extrapolation, 40 compounds |
| `analgesic_predictions_classed.csv` | The same, annotated by pharmacological class |
| `descriptor_importance.csv` | Permutation importances |
| `layer_stratified.csv`, `layer_stratified_rkf.csv` | Performance stratified by skin-layer preparation |
| `logp_comparison.csv` | Computed against measured logP for the 98 compounds with a measured value |
| `variance_decomposition.csv` | Variance attributable to experimental protocol |
| `stats_tests.json` | The statistical tests reported in the text |
| `split_train_idx.npy`, `split_test_idx.npy` | Held-out split indices |

Note on terminology: the file name `external_test_predictions.csv` is retained from the original
submission for continuity. The 66-compound set is an **independent held-out test subset** of the
same pooled data, not an external dataset in the strict sense; the manuscript uses the latter
wording throughout.

### `derived/revision/`

| Directory | Analysis | Answers |
|---|---|---|
| `major1/` | Aggregation scheme comparison and mixed-effects variance decomposition | median vs mean vs record level; 52% of record-level variance is within compound |
| `m2/` | Accumulated local effects with 200 bootstrap refits | the molecular-weight threshold does not survive; lipophilicity saturation does |
| `m3_m4/` | 200 repeated stratified splits; Nadeau–Bengio corrected fold-wise tests | held-out R² distribution; corrected p values |
| `m5/` | Permutation null and nearest-neighbour matching on skin layer | the layer gap lies inside the null (p = 0.184) but survives matching |
| `m8/` | Ionization at pH 7.4 (Dimorphite-DL) and MW-vs-LabuteASA nesting | adding ionization changes R² by ≤ 0.005 |
| `minor1/` | ATC code verification of all 40 analgesics via RxClass | 33 coded, 32 agree with the published assignment |

### `deposit/`

| File | Contents |
|---|---|
| `split_assignment.csv` | The held-out split, per compound |
| `record_fold_assignment.csv` | Record-level grouped cross-validation folds |
| `reproducibility_metadata.json` | Package versions and seeds |
| `OECD_QSAR_validation_statement.md` | Conformity to the five OECD principles for QSAR validation |

## Reproducing the analysis

Run the numbered scripts in order. `00_inventory.py` and `01_union_strict.py` retrieve and merge
the two databases; `02_build_dataset.py` standardises structures and computes the fifteen
two-dimensional descriptors with RDKit; `03_model.py` fits and compares the linear, ridge, random
forest and gradient boosting models by repeated ten-fold cross-validation, a held-out test set
and leave-analgesics-out extrapolation; `04_protocol.py` quantifies protocol effects at record
level with grouped cross-validation; `12_figdata.py` and `13_foldlevel.py` write the per-panel
inputs in `derived/figdata/`.

The analyses added at revision are `15_revision_major1.py` (aggregation and mixed effects),
`16_revision_m3_m4.py` (200 splits and corrected tests), `17_revision_m8.py` (ionization),
`18_revision_m5.py` (permutation and matching), `19_revision_m2_ale.py` (accumulated local
effects) and `22_atc_verification.py` (ATC codes).

Figures are drawn by `scripts/figures/render_figures.py`,
`scripts/figures/render_supp_figures.py` and `scripts/figures/render_revision_figures.py`.
The earlier `05_` scripts are superseded and are kept only for provenance.

## Licence

Code is released under the MIT licence (see `LICENSE`). Derived tables may be reused with
attribution to this repository and to HuskinDB and SkinPiX.
