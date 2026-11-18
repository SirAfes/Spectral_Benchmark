# Spectral Benchmark

A reusable supervised-classification benchmark for multispectral and hyperspectral imagery. It was designed to reproduce the useful parts of an ENVI-style workflow while producing stronger validation data, comparison tables and publication-ready figures.

## What it does

The script can evaluate the same labelled pixels with multiple classifiers:

- `SVM` — RBF support-vector machine with grouped inner hyperparameter search.
- `RF` — Random Forest.
- `HGB` — Histogram Gradient Boosting.
- `LDA` — shrinkage Linear Discriminant Analysis.
- `KNN` — distance-weighted k-nearest neighbours.
- `SAM` — Spectral Angle Mapper using class-mean spectra.
- `SID` — symmetric Spectral Information Divergence using class-mean spectral distributions.
- `MIN_DIST` — Euclidean Minimum Distance to class means.
- `MAHALANOBIS` — nearest class mean under a Ledoit-Wolf-shrunk **pooled within-class** covariance (estimated from class-centred residuals).
- `QDA` — regularized quadratic Gaussian classifier; useful as a stable maximum-likelihood-like baseline.

The framework uses **spatially grouped nested cross-validation**. Pixels belonging to the same spatial group never appear simultaneously in the train and test partitions of an outer fold. This reduces the optimistic bias that occurs when neighbouring pixels from one ROI are randomly split between training and validation.

## Supported label inputs

Use exactly one of:

1. `--roi-xml` — ENVI `RegionsOfInterest` XML with `PixelDef` coordinates.
2. `--samples-csv` — columns `row,col,class` and optional `group,r,g,b`.
3. `--label-raster` — one-band integer raster where `0` is unlabelled. Use `--class-map` optionally with columns `code,name[,r,g,b]`.

For ENVI XML, coordinates are interpreted as `column row`, which matches ENVI PixelDef.

## Installation

Python 3.10 or newer. `rasterio` ships binary wheels for Windows, macOS and Linux; with Conda, `conda install -c conda-forge rasterio` is the most reliable route.

```bash
git clone <this-repository-url>
cd spectral-benchmark
pip install -r requirements.txt        # run the script directly
# or, to get a `spectral-benchmark` command on your PATH:
pip install -e .
```

## First run: validation and reporting only

This is the safest first run because it does **not** classify the full raster. Copy `examples/example_config.json`, edit the paths, and keep personal copies named `*.local.json` (ignored by git):

```bash
python spectral_benchmark.py --config examples/example_config.json
```

Without a config file:

```bash
python spectral_benchmark.py --image data/image.tif --roi-xml data/rois.xml --output-dir results/benchmark --models "SVM,RF,HGB,LDA,KNN,SAM,SID,MIN_DIST,MAHALANOBIS,QDA" --feature-sets raw
```

Command-line options override the config file. All options are listed in [`docs/CLI_HELP.txt`](docs/CLI_HELP.txt) (or run with `--help`). Unknown keys in the config file produce a warning, so typos are not silently ignored; note that the config key is `group_block_size_px` while the CLI flag is `--group-block-size`.

## Feature-set experiments

The important presets are:

- `raw` — selected original spectral bands.
- `mean_norm` — each pixel spectrum is divided by its own mean. This can reduce multiplicative brightness variation, but it does not convert DN values to reflectance.
- `raw_indices` — raw bands plus requested derived indices.
- `mean_norm_indices` — mean-normalized spectrum plus indices.
- `raw_derivative` — raw bands plus wavelength-normalized first derivative.
- `mean_norm_derivative` — normalized bands plus wavelength-normalized first derivative.

Example:

```bash
python spectral_benchmark.py --config examples/example_config.json --feature-sets "raw,mean_norm,raw_indices,mean_norm_indices" --indices "GLI,TGI,RGRI,VARI,NGRDI,BG_RATIO,GR_RATIO"
```

The built-in indices use the nearest available wavelength within the configured tolerance. The actual wavelengths used are written to `tables/index_definitions_<feature_set>.txt`.

**SAM and SID intentionally use only the spectral part of a feature set.** They do not use derived indices or derivative columns because their physical interpretation is based on a spectrum. Other machine-learning classifiers use all features in the selected feature set.

## Band selection

Do not remove a band only because it looks dark on screen. The script writes:

- `tables/band_diagnostics.csv`
- `figures/band_diagnostics.png`
- `tables/class_spectral_statistics.csv`
- `figures/class_spectral_signatures.png`

`band_diagnostics.csv` includes approximate global mean/std, robust range, zero percentage, saturation percentage, entropy and a training-sample Fisher separability score. The `recommendation` field is deliberately conservative:

- `DROP_CONSTANT` — effectively no variation.
- `REVIEW_NEAR_BLANK` — almost entirely zero with negligible range.
- `REVIEW_SATURATED` — heavily clipped.
- `KEEP_OR_VALIDATE` — do not remove automatically; test it by cross-validation.

A scientifically defensible workflow is to compare `raw` against a manually selected band set and retain the change only when grouped validation supports it.

Band-selection examples:

```bash
python spectral_benchmark.py --config examples/example_config.json --drop-bands "12,13,14"
```

or wavelength-based:

```bash
python spectral_benchmark.py --config examples/example_config.json --drop-wavelengths "710,735,850"
```

These are examples only; inspect diagnostics before removing real bands.

## Full-raster classification

Full raster mapping is **off by default** because large mosaics can contain billions of pixels.

Map the single best validation combination:

```bash
python spectral_benchmark.py --config examples/example_config.json --map-top 1
```

Map specific combinations:

```bash
python spectral_benchmark.py --config examples/example_config.json --map "raw:SVM,raw:SAM,raw:SID"
```

The map is processed tile-by-tile, so the entire raster is never loaded into RAM. Classification outputs use `0 = NoData`, then class codes `1..N`. A `legend.csv` is written beside the maps.

Confidence rasters are optional because they can be extremely large:

```bash
python spectral_benchmark.py --config examples/example_config.json --map-top 1 --write-confidence
```

For distance methods, SAM and SID, the confidence value is a normalized relative score and is **not a calibrated probability**. SVM decision-function confidence is also uncalibrated. The map metadata records the confidence type.

## Main outputs

### Tables

- `model_comparison.csv` — primary model/feature-set comparison with OA, balanced accuracy, Macro-F1, weighted F1, Kappa, MCC, fold statistics and grouped-bootstrap 95% confidence intervals.
- `all_per_class_metrics.csv` — class-wise precision/User's Accuracy, recall/Producer's Accuracy, F1 and support.
- `all_fold_metrics.csv` — metrics from each outer spatial fold.
- `pairwise_model_comparisons.csv` — paired A-minus-B differences in OA and Macro-F1 with grouped-bootstrap confidence intervals.
- `failed_runs.csv` — only written when a model/feature-set combination raised an error (the run continues with the others; set `SPECTRAL_BENCHMARK_DEBUG=1` to stop and see the traceback).
- `confusion_counts_*.csv` — confusion matrices in pixel counts.
- `confusion_row_normalized_*.csv` — confusion matrices normalized by reference class.
- `oof_predictions_*.csv` — every out-of-fold validation prediction, with row/column/group.
- `best_params_*.csv` — chosen inner-CV hyperparameters for every outer fold.
- `band_diagnostics.csv` — band quality and separability screening.
- `class_spectral_statistics.csv` — mean/std/median/P05/P95 per class and band.
- `class_mean_spectral_separability.csv` — pairwise class-mean spectral angle and standardized distance.
- `training_sample_manifest.csv` — exact pixels used by the benchmark.

### Figures

- `model_comparison_macro_F1.png`
- `model_comparison_OA.png`
- `per_class_F1_heatmap.png`
- `class_spectral_signatures.png`
- `band_diagnostics.png`
- one normalized confusion-matrix PNG for every model/feature-set combination
- preview PNGs for any full-raster maps produced

### Reproducibility

`run_manifest.json` records raster metadata, wavelengths, selected bands, models, feature sets, label source, grouping configuration, seed, sampling caps, failed runs and library versions. Runs are deterministic for a fixed `--seed`.

## Metrics to report

For a compact scientific comparison, use:

1. Macro-F1 as the main balanced multiclass score.
2. Overall Accuracy as an intuitive global score.
3. Per-class F1 or PA/UA to identify difficult taxa/classes.
4. Row-normalized confusion matrices for the best few models.
5. Grouped-bootstrap 95% confidence intervals when possible.

Kappa and MCC are also generated but do not need to be the main story unless required by the venue.

## Spatial groups

For CSV labels, provide a `group` column when you already know the independent polygons/sites/sampling units. This is the preferred option.

For ENVI PixelDef XML, the script first uses `GeometryDef` polygons as independent groups when those polygons are present and can be matched to the raster. Pixels without usable polygon geometry fall back to fixed spatial blocks. Use `--group-block-size` to control that fallback scale. The correct size is dataset-specific; it should be large enough that strongly correlated neighbouring training pixels are not split across validation folds.

If a class occurs in too few independent groups, the script reduces the number of folds. If fewer than two independent groups exist for a class, grouped validation stops and explains the problem rather than silently falling back to random pixel splitting.

## SVM performance note

RBF SVM fit time grows rapidly with sample count. The framework therefore supports `--svm-train-cap` while keeping the full sampled test folds for evaluation. The default is 12,000 fit samples. Increase it only when runtime and memory allow it.

## Limitations to keep in mind

- **Optimistic by construction.** Training ROIs are usually drawn on clearly identifiable, homogeneous patches. Grouped validation removes neighbour leakage *between* ROIs, but accuracy on ROI pixels will still overstate accuracy over the whole scene, especially for mixed or transitional pixels. Validate a final map against independently collected reference points when you can.
- **Class-balanced sampling.** `--max-samples-per-class` caps every class, so Overall Accuracy is computed on a roughly class-balanced sample, not on the true area proportions of the scene. Prefer Macro-F1 / per-class metrics, or area-weight the confusion matrix before quoting an area-based OA.
- **Few independent groups.** With only a handful of polygons per class, fold-to-fold variance is large and bootstrap confidence intervals are wide. That is a property of the data, not a bug; report the intervals.
- **Pixel-level only.** No spatial context, texture or post-classification smoothing is used.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v          # unit tests + a synthetic end-to-end run (needs rasterio)
ruff check .
```

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT — see [`LICENSE`](LICENSE).

## Important interpretation notes

- Mean spectral normalization is a feature transformation, not radiometric calibration.
- Derived indices do not create new physical measurements; they may expose useful nonlinear relationships to a classifier.
- Dark-looking NIR/red-edge bands should not be discarded without checking class separation and validation performance.
- Raw pixel-level classification and spatial post-processing should be compared separately. Do not apply a sieve/majority filter to only one model and then call that a fair raw-classifier comparison.
- Validation pixels should be spatially independent from training pixels. Random neighbouring-pixel splits can produce unrealistically high accuracy.
