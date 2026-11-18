# Changelog

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
