# Conformity with the OECD principles for the validation of (Q)SAR models

Prepared in response to Reviewer 2, Major 10. Every statement below refers to what was actually
done in this study; where a principle is only partly met, that is stated rather than glossed.

---

## 1. A defined endpoint

The endpoint is the steady-state permeability coefficient of human skin, reported as
log *K*~p~ in cm s⁻¹.

Values were taken from two open databases of *in vitro* human skin permeation, HuskinDB and
SkinPiX version 1.1, and harmonised to a common unit; where a source reported cm h⁻¹,
log₁₀(3600) was subtracted. Records measured on non-human skin were excluded by the source
databases' own scope. The pooled analysis set is 732 records for 327 compounds.

**Qualification.** The endpoint is defined at the level of the measured quantity, not at the level
of a single standardised protocol. The records were produced under heterogeneous conditions — skin
layer, anatomical site, diffusion cell, donor and receptor temperature and pH — and this study
treats that heterogeneity as an object of analysis rather than removing it by exclusion. A
mixed-effects decomposition of the record-level variance places 52 per cent of it within compounds,
that is between repeated measurements of the same molecule. Any model of this endpoint inherits
that variability.

---

## 2. An unambiguous algorithm

Six predictors were compared, all fully specified:

| Model | Specification |
|---|---|
| Potts–Guy (literature) | log *K*~p~ = −2.72 + 0.71 log *K*~ow~ − 0.0061 MW, published coefficients applied without fitting, converted to cm s⁻¹ |
| Potts–Guy form, refitted | same two descriptors, coefficients refitted to the present data by ordinary least squares on standardised inputs |
| Multiple linear regression | ordinary least squares on all 15 standardised descriptors |
| Ridge regression | as above with an L2 penalty, the penalty selected by internal cross-validation within the training set only |
| Random forest | 800 trees, minimum leaf size 2, all other scikit-learn defaults |
| Gradient boosting | 700 trees, learning rate 0.03, maximum depth 4, subsample 0.8, column subsample 0.8, L2 regularisation 1.0 |

No hyperparameter search was performed. The ensemble settings were fixed a priori and never varied;
the ridge penalty is the only quantity chosen from data, and it was chosen inside the training set.
The held-out test set took no part in fitting, in penalty selection, or in any choice of setting.

Descriptors are 15 two-dimensional RDKit descriptors (version 2026.03.5): molecular weight, Crippen
log *P*, molar refractivity, topological polar surface area, counts of hydrogen-bond donors and
acceptors, rotatable bonds, aromatic rings, total rings, heavy atoms and heteroatoms, fraction of
sp³-hybridised carbon, Labute approximate surface area, Balaban J index and Bertz complexity index.

Random seed 20260821 throughout. Split assignments, grouped cross-validation fold assignments and
the software versions are deposited alongside the code, so every split and fold can be reproduced
exactly rather than re-estimated.

---

## 3. A defined domain of applicability

The applicability domain is the mean Euclidean distance, in standardised descriptor space, to the
five nearest training compounds, with the threshold set at the 95th percentile of the corresponding
distances within the training set.

All 40 analgesics in the leave-class-out analysis fall inside this domain.

**Qualification, and it matters.** Falling inside the domain did not make the predictions reliable
at the edges. Predictions for the analgesic class were systematically compressed at both extremes,
over-predicting the least permeable opioids such as morphine and under-predicting the most permeable
compounds, fentanyl and sufentanil — which are the clinically important members of the class. A
domain check of this kind reports chemical similarity to the training set; it does not certify
accuracy, and this study is an explicit example of the difference.

---

## 4. Appropriate measures of goodness-of-fit, robustness and predictivity

| Measure | How obtained |
|---|---|
| Cross-validated R² and RMSE | 10-fold cross-validation; folds grouped by compound wherever records rather than compounds are the unit, so no compound is split across folds |
| Held-out R² , RMSE and MAE | one stratified 80:20 split on compounds, evaluated once |
| Distribution of held-out R² | the split repeated 200 times under the same stratification; the random forest gives a median of 0.383 with a 2.5–97.5 percentile interval of 0.144 to 0.559, takes first place in 79 per cent of splits, and the single split reported in the manuscript sits at the 26th percentile of its own distribution |
| Extrapolation to a new class | all 40 analgesics withheld from training, models refitted on the 287 remaining compounds |
| Sensitivity to aggregation | repeated with the mean in place of the median, at record level with grouped cross-validation, and as a mixed-effects model with compound as a random intercept |
| Fold-wise comparisons | Nadeau-Bengio correction for resampled tests, reported with effect sizes and intervals, because cross-validation folds share training data and are not independent |
| Confounding check | permutation test preserving compound identity, and nearest-neighbour matching on standardised descriptors, for the skin-layer comparison |

**Qualification.** Two results weakened under these checks and are reported as weakened: the
skin-layer difference is no longer significant after correction for fold dependence (p = 0.059,
from p = 0.000004), and the molecular-weight threshold does not survive an accumulated-local-effects
analysis, its span of 0.44 log units being smaller than the mean width of its own 95 per cent
bootstrap band.

---

## 5. A mechanistic interpretation, if possible

A mechanistic reading is offered for one relationship and explicitly withheld for the other.

**Offered.** The dependence on lipophilicity is saturating: predicted log *K*~p~ rises steeply and
plateaus above a computed octanol–water partition coefficient of about 2.5. Under accumulated local
effects, which are robust to the collinearity among the descriptors, this spans 1.93 log units
against a within-compound noise standard deviation of 0.963, so it is resolved by the data. It is
consistent with partition-limited transport across the lipid-rich stratum corneum: once a solute
partitions favourably into the intercellular lamellae, further lipophilicity adds little to the rate
at which it crosses the barrier.

**Withheld.** The model also learned a decline in permeability above a molecular weight near
220 g mol⁻¹, and an earlier version of this manuscript presented it as a finding. It does not
survive: the effect spans 0.44 log units against a bootstrap band of 0.54, molecular weight carries
no information independent of the Labute approximate surface area (partial r = 0.008, p = 0.88), and
removing molecular weight from the descriptor set moves cross-validated R² from 0.364 to 0.366. It
is now reported as a feature of the fitted model rather than a property of skin.

---

## Summary

Principles 1, 2 and 4 are met in full. Principle 3 is met as specified, with the limitation that
domain membership did not guarantee accuracy at the extremes of the held-out class. Principle 5 is
met for the lipophilicity relationship and deliberately not claimed for molecular weight.
