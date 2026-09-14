# Human Skin Permeability Prediction by Machine Learning for Transdermal Drug Delivery

Analysis code and derived data for the manuscript submitted to *Medicina* (MDPI).

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
| `derived/` | The pooled dataset and every table behind the manuscript and the figures |
| `figures/` | Figures 1–4 and Supplementary Figures S1–S5 (600 dpi TIFF, plus PNG previews) |

### `derived/`

| File | Size | Contents |
|---|---|---|
| `analgesic_predictions.csv` | 14 KB | Leave-analgesics-out extrapolation, 40 compounds |
| `analgesic_predictions_classed.csv` | 15 KB | The same, annotated by pharmacological class |
| `compounds_modelling.csv` | 71 KB | Analysis-ready compound table — 327 compounds with descriptors |
| `descriptor_importance.csv` | 1 KB | Descriptor importances |
| `external_test_predictions.csv` | 8 KB | Observed against predicted log Kp on the external test set (66 compounds) |
| `layer_stratified.csv` | 0 KB | Performance stratified by skin-layer preparation |
| `layer_stratified_rkf.csv` | 0 KB | The same under repeated k-fold |
| `logp_comparison.csv` | 5 KB | Computed against measured logP for the 98 compounds with a measured value |
| `merged_inventory.csv` | 129 KB | Record-level inventory of the two source databases after harmonisation |
| `model_results.csv` | 1 KB | Cross-validated performance of every model family |
| `records_with_descriptors.csv` | 248 KB | All 732 pooled permeability records with computed descriptors |
| `split_test_idx.npy` | 1 KB | Test split indices |
| `split_train_idx.npy` | 2 KB | Train split indices |
| `stats_tests.json` | 1 KB | The statistical tests reported in the text |
| `union_smiles_verified.csv` | 102 KB | Structure standardisation, verified SMILES per compound |
| `variance_decomposition.csv` | 0 KB | Variance attributable to experimental protocol |

## Reproducing the analysis

Run the numbered scripts in order. `00_inventory.py` and `01_union_strict.py` retrieve and merge
the two databases; `02_build_dataset.py` standardises structures and computes the fifteen
two-dimensional descriptors with RDKit; `03_model.py` fits and compares the linear, ridge, random
forest and gradient boosting models by repeated ten-fold cross-validation, an external test set
and leave-analgesics-out extrapolation; `04_protocol.py` quantifies protocol effects at record
level with grouped cross-validation; the `05_` scripts draw the figures.

## Licence

Code is released under the MIT licence (see `LICENSE`). Derived tables may be reused with
attribution to this repository and to HuskinDB and SkinPiX.
