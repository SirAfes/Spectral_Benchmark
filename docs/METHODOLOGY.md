# Methodology notes and known limitations

This page explains what the validation does and does not guarantee, how to choose the spatial
options, and which biases remain. Read it before quoting accuracies from a paper or report.

## 1. What the validation protocol does

1. Labelled pixels are sampled (class-capped, spread over spatial groups).
2. **Outer loop** — `StratifiedGroupKFold`: every spatial group is in exactly one test fold; each pixel gets one
   out-of-fold (OOF) prediction per repeat. All headline metrics are computed on OOF predictions.
3. **Inner loop** — hyper-parameters are tuned with grouped folds *inside* each outer training set only, so the
   outer test fold never influences tuning or scaling.
4. **Uncertainty** — 95 % intervals come from a bootstrap that resamples whole *groups*. Pairwise model differences use
   the same resampled groups for both models (paired).

## 2. Residual leakage between neighbouring groups

Grouping prevents a polygon from being split between train and test. It does **not** stop two *adjacent*
polygons (or two grid blocks cut from one homogeneous patch) from ending up on opposite sides. Neighbouring pixels share
illumination, sensor noise, water column / canopy conditions and mixed-pixel effects, so a flexible model (kNN, RF,
SVM, HGB) can recognise the test pixel's neighbour in the training set and look much better than it really is.

Three tools address this (all off by default so results stay comparable with v1.0/v1.1):

| Option | Effect |
|---|---|
| `--group-merge-px N` | Groups with pixels within ~N px of each other are merged (transitively), so touching polygons/blocks are always in the same fold. Merging only coarsens groups; if it leaves a class with < 2 groups the run stops with an explanatory error. |
| `--buffer-px N` | Training pixels within N px of any test pixel are dropped, in the outer **and** inner folds. Needed when a single large patch must be split across folds. |
| *(always on)* `min_train_test_dist_px` | Smallest distance between a training and a test pixel, per fold (`tables/all_fold_metrics.csv`, `model_comparison.csv`). Values of only 1-2 px mean neighbour leakage is possible. |

**Choosing N.** Use roughly the spatial autocorrelation range of your imagery (distance at which a band's
semivariogram levels off; for UAV mosaics often tens of pixels). The buffer must stay small compared with your groups,
otherwise most training pixels are removed — the run warns above 50 % removal and errors if training becomes
unusable. A quick sanity experiment: run with `--buffer-px 0` and with your chosen N. A large drop in flexible models
(kNN/RF/SVM) but not in SAM/LDA indicates that the unbuffered score was inflated by leakage.

## 3. Repeated cross-validation

With only a handful of groups per class, the result depends on which groups land in which fold. `--repeats R` repeats the outer
split with different fold assignments; headline metrics are the mean over repeats, `*_repeat_std` shows the spread,
and the bootstrap resamples groups jointly for all repeats. Cost grows linearly with R. This reduces fold-assignment noise;
it does **not** create new independent groups, so it cannot fix a data set with very few of them.

## 4. Comparing many models

`pairwise_model_comparisons.csv` contains a bootstrap p-value and a **Holm-adjusted** p-value per pair within a
feature set. Comparing 10 models gives 45 pairs; unadjusted intervals will show some "differences" by chance. Use the
`_p_holm` columns for claims of superiority. The bootstrap p-value is approximate and limited by the number of
replicates (resolution about 2/(B+1); use `--pairwise-bootstrap 1000` or more for publication).

## 5. Biases that remain (and are not fixable in software)

- **Winner's curse / selection on the same data.** Picking the best of many model x feature-set combinations on the
  same OOF predictions makes the best score optimistic. Fix the protocol before looking, or confirm on independent areas.
- **Band and feature selection.** `band_diagnostics.csv` (Fisher scores etc.) uses all training pixels. Dropping bands based on it
  and then cross-validating on the same pixels leaks label information. Choose bands from physical reasoning or
  select them inside the CV folds.
- **ROI selection bias.** ROIs are drawn on clear, homogeneous patches; accuracy on them overstates accuracy on mixed or
  transitional pixels and on classes that were not digitised.
- **Class-balanced sampling.** Overall Accuracy is computed on a capped, roughly balanced sample, not on the scene's true class proportions.
  Prefer Macro-F1 / per-class metrics, or area-weight the confusion matrix before quoting an area-based OA.
- **Interval coverage.** Group-bootstrap intervals reflect sampling variability of the *test groups*; they do not capture
  variability from re-training on different groups, and they are unreliable with fewer than ~8-10 groups per class.

## 6. Model-specific notes

- SAM, SID, MIN_DIST and MAHALANOBIS are class-mean classifiers: they cannot represent multi-modal classes. MAHALANOBIS
  uses a pooled **within-class** covariance (shrunk with Ledoit-Wolf); QDA estimates a covariance per class.
- SAM and SID use only the spectral part of a feature set (no indices/derivatives).
- Confidence values from distance-based models are a softmax of distances with a scale fixed at fit time. They rank
  pixels within one map but are **not** probabilities.
- Every pixel in a map is forced into one of the trained classes. Water, shadow, unvisited habitats and mixed pixels are
  therefore misclassified silently. Consider adding an explicit "other" class or masking such areas upstream.
