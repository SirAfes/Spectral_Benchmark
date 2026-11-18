# Changelog

## 1.3.0

### Added
- `--holdout-fraction`: reserves whole spatial groups before any analysis and scores the selected best combination once on them (`tables/holdout_result.csv`, `holdout_per_class_metrics.csv`, `holdout_confusion_counts.csv`); the `split` column in the sample manifest shows the assignment. Final maps are fitted on all labelled pixels.
- `--class-prevalence`: area-weighted OA, macro-F1, kappa and per-class UA/PA (`area_weighted_per_class_metrics.csv`).
- `--min-confidence`: maps can mark low-confidence pixels as `Unclassified` (with a warning for models whose confidence is not a probability).
- Warning when a class has fewer than 8 independent groups; `min_groups_per_class` in `model_comparison.csv`.
- Log note that `band_diagnostics.csv` is screening information only.
- End-to-end test with all spatial options; unit tests for the new functions.

### Fixed
- The model-comparison figures crashed the whole run when a bootstrap interval did not contain the point estimate (negative error bars; likely with few bootstrap replicates).


## 1.2.0

### Added
- `--buffer-px`: spatial buffer that removes training pixels near test pixels (outer and inner folds), plus an always-on `min_train_test_dist_px` leakage diagnostic.
- `--group-merge-px`: merges polygons/blocks that touch or lie within N pixels, so adjacent groups cannot straddle train and test.
- `--repeats`: repeated grouped CV; headline metrics are averaged over repeats, bootstrap CIs resample groups jointly.
- Bootstrap p-values and Holm-adjusted p-values in `pairwise_model_comparisons.csv`.
- `docs/METHODOLOGY.md`: what the protocol guarantees, remaining biases, how to choose the spatial options.
- Clear error / warning when a buffer removes too many training pixels.

### Fixed
- `group_source` in the manifest said "ENVI polygons" for CSV group columns.

### Notes
- Defaults are unchanged, so v1.1 results are reproduced exactly unless the new options are used (the fold table gains extra columns).

## 1.1.0

### Fixed
- **MAHALANOBIS** now uses the *pooled within-class* covariance, as documented. Previously it used the total covariance of the training set, which includes between-class scatter and down-weights exactly the directions that separate classes. **Mahalanobis results will differ from v1.0.**
- Confidence from SAM / SID / MIN_DIST / MAHALANOBIS (`predict_proba`) no longer depends on which tile a pixel falls in; the softmax scale is fixed at fit time instead of recomputed per call.
- **SID** is computed with matrix products instead of an `n x classes x bands` tensor, so large tiles of hyperspectral data no longer need gigabytes of RAM. Results are numerically identical.
- GeoTIFF block size for maps is rounded to a multiple of 16, so `--tile-size` values such as 500 no longer produce an invalid file. BigTIFF is only requested when GDAL deems it necessary.
- ENVI ROI XML: regions that share a name are merged into one class (previously duplicate class names); coordinate text is parsed strictly instead of with the deprecated `np.fromstring`.
- Pixels labelled with more than one class (overlapping ROIs) are dropped with a warning; exact duplicate pixels are kept once.
- `np.unique(..., axis=0, return_inverse=True)` result is flattened (shape differs across NumPy 2.0.x).
- `--keep-bands` together with `--keep-wavelengths` is now an error instead of one silently overriding the other.

### Added
- Warning for unknown keys in the JSON config file.
- Warning when an outer training fold contains no pixels of some class.
- `tables/failed_runs.csv` and a `failed_runs` entry in the manifest; `SPECTRAL_BENCHMARK_DEBUG=1` now also re-raises model errors.
- Seed, sampling caps, tile size and other settings recorded in `run_manifest.json`.
- Bootstrap warnings from scikit-learn (missing classes in a resample) are silenced.
- pytest unit tests (including regression tests for the fixes above), CI workflow, `pyproject.toml` with a `spectral-benchmark` entry point, `.gitignore`, MIT license, generic example config.

### Removed
- Machine-specific example config / launcher script and the one-off test report (replaced by automated tests).

## 1.0.0
- Initial release.
