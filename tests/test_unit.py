"""Unit tests for the pure (non-raster) logic of spectral_benchmark."""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spectral_benchmark as sb  # noqa: E402


# --------------------------------------------------------------------------- helpers


def two_class_data(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Classes separated along a *low-variance* axis; axis 0 is large shared noise."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal([0.0, 0.0], [3.0, 0.3], (n, 2))
    X1 = rng.normal([0.0, 1.5], [3.0, 0.3], (n, 2))
    return np.vstack([X0, X1]), np.r_[np.zeros(n, int), np.ones(n, int)]


def positive_spectra(n_classes: int = 3, n: int = 200, bands: int = 6, seed: int = 1):
    rng = np.random.default_rng(seed)
    means = rng.uniform(20, 120, (n_classes, bands))
    X = np.vstack([rng.normal(m, 4.0, (n, bands)) for m in means]).clip(1)
    y = np.repeat(np.arange(n_classes), n)
    return X, y


# --------------------------------------------------------------------------- parsing


def test_parse_helpers():
    assert sb.parse_csv_list(" a, b ,,c ") == ["a", "b", "c"]
    assert sb.parse_csv_list(None) == []
    assert sb.parse_int_list("1, 2,3") == [1, 2, 3]
    assert sb.parse_float_list("450,550.5") == [450.0, 550.5]


def test_parse_number_text_handles_whitespace_and_floats():
    out = sb.parse_number_text("1 2\n3\t4 ", np.int64)
    assert out.tolist() == [1, 2, 3, 4]
    assert sb.parse_number_text("12.0 7.9", np.int64).tolist() == [12, 8]
    assert sb.parse_number_text(None).size == 0
    with pytest.raises(ValueError):  # unlike np.fromstring, bad tokens are not silently truncated
        sb.parse_number_text("1 2 x 4")


def test_sanitize_name():
    assert sb.sanitize_name(" a b/c ") == "a_b_c"
    assert sb.sanitize_name("///") == "item"


# --------------------------------------------------------------------------- bands


WL = np.array([450.0, 550.0, 650.0, 850.0])


def test_band_selection_default_keeps_all():
    assert sb.resolve_band_selection(4, WL, [], [], [], [], 20).tolist() == [0, 1, 2, 3]


def test_band_selection_drop_and_keep():
    assert sb.resolve_band_selection(4, WL, [], [2], [], [], 20).tolist() == [0, 2, 3]
    assert sb.resolve_band_selection(4, WL, [1, 4], [], [], [], 20).tolist() == [0, 3]
    assert sb.resolve_band_selection(4, WL, [], [], [], [850.0], 20).tolist() == [0, 1, 2]


def test_band_selection_errors():
    with pytest.raises(ValueError):
        sb.resolve_band_selection(4, WL, [9], [], [], [], 20)
    with pytest.raises(ValueError):
        sb.resolve_band_selection(4, WL, [], [], [], [700.0], 20)  # nothing within 20 nm
    with pytest.raises(ValueError):
        sb.resolve_band_selection(4, None, [], [], [], [550.0], 20)  # no wavelengths
    with pytest.raises(ValueError):
        sb.resolve_band_selection(4, WL, [], [1, 2, 3, 4], [], [], 20)  # removes everything
    with pytest.raises(ValueError):
        sb.resolve_band_selection(4, WL, [1], [], [550.0], [], 20)  # both keep styles


# --------------------------------------------------------------------------- features


def test_gli_value_and_definition():
    spec = np.array([[10.0, 40.0, 20.0, 5.0]])  # B=450, G=550, R=650
    arrays, names, defs = sb.compute_indices(spec, WL, ["GLI", "NDVI"], 20)
    assert names == ["GLI", "NDVI"]
    assert arrays[0][0] == pytest.approx((2 * 40 - 20 - 10) / (2 * 40 + 20 + 10))
    assert arrays[1][0] == pytest.approx((5 - 20) / (5 + 20))
    assert "G=550nm" in defs[0]


def test_indices_skipped_without_wavelengths_or_in_tolerance():
    spec = np.ones((2, 4))
    with pytest.warns(UserWarning):
        assert sb.compute_indices(spec, None, ["GLI"], 20)[0] == []
    with pytest.warns(UserWarning):
        assert sb.compute_indices(spec, np.array([400.0, 410, 420, 430]), ["NDVI"], 20)[0] == []


def test_mean_normalisation_and_derivative_shapes():
    raw = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    names = [f"b{i}" for i in range(4)]
    idx = np.arange(4)
    X, ctx = sb.build_features(raw, idx, names, WL, sb.feature_preset("mean_norm", []), 20)
    assert np.allclose(X.mean(axis=1), 1.0)
    X, ctx = sb.build_features(raw, idx, names, WL, sb.feature_preset("raw_derivative", []), 20)
    assert X.shape == (2, 4 + 3) and ctx.base_feature_count == 4
    assert X[0, 4] == pytest.approx((20 - 10) / (550 - 450))
    with pytest.raises(ValueError):
        sb.feature_preset("nope", [])


# --------------------------------------------------------------------------- classifiers


def test_sid_matches_direct_formula():
    """The matrix-product SID must equal the textbook broadcast formulation."""
    X, y = positive_spectra()
    sid = sb.SpectralInformationDivergence(X.shape[1]).fit(X, y)
    P, Q = sid._prob(X[:50]), sid.means_
    direct = np.sum(P[:, None, :] * (np.log(P)[:, None, :] - np.log(Q)[None]), axis=2) + np.sum(
        Q[None] * (np.log(Q)[None] - np.log(P)[:, None, :]), axis=2
    )
    assert np.allclose(sid.distance_matrix(X[:50]), direct, atol=1e-9)


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: sb.SpectralAngleMapper(p),
        lambda p: sb.SpectralInformationDivergence(p),
        lambda p: sb.MinimumDistanceClassifier(),
        lambda p: sb.PooledMahalanobisClassifier(),
    ],
)
def test_reference_classifiers_separate_easy_classes(factory):
    X, y = positive_spectra()
    model = factory(X.shape[1]).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3) and np.allclose(proba.sum(axis=1), 1.0)


def test_sam_and_sid_use_only_leading_spectral_columns():
    X, y = positive_spectra(bands=4)
    junk = np.random.default_rng(3).normal(size=(len(X), 3)) * 1e6
    Xj = np.column_stack([X, junk])
    for cls in (sb.SpectralAngleMapper, sb.SpectralInformationDivergence):
        a = cls(4).fit(X, y).predict(X)
        b = cls(4).fit(Xj, y).predict(Xj)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("cls", [sb.MinimumDistanceClassifier, sb.PooledMahalanobisClassifier])
def test_confidence_does_not_depend_on_batch(cls):
    """Regression: scale used to be recomputed per call, so tiles disagreed."""
    X, y = positive_spectra()
    model = cls().fit(X, y)
    alone = model.predict_proba(X[:20])
    in_batch = model.predict_proba(np.vstack([X[:20], X[-300:] * 3.0]))[:20]
    assert np.allclose(alone, in_batch)


def test_mahalanobis_uses_within_class_covariance():
    """Regression: total covariance hides the discriminative low-variance axis."""
    X, y = two_class_data()
    model = sb.PooledMahalanobisClassifier().fit(X, y)
    # Within-class std is 0.3 on axis 1 -> precision should be ~11, not ~1.5 (total cov).
    assert model.precision_[1, 1] > 6.0
    assert (model.predict(X) == y).mean() > 0.97


def test_build_model_covers_all_default_models():
    for name in sb.DEFAULT_MODELS:
        est, grid = sb.build_model(name, n_features=6, spectral_cols=6, seed=0)
        assert est is not None and isinstance(grid, dict)
    with pytest.raises(ValueError):
        sb.build_model("NOPE", 6, 6, 0)


# --------------------------------------------------------------------------- groups & CV


def test_assign_spatial_groups_is_1d_and_blocky():
    rows = np.array([0, 5, 10, 300, 301])
    cols = np.array([0, 5, 10, 300, 10])
    g = sb.assign_spatial_groups(rows, cols, 100)
    assert g.ndim == 1 and g.dtype == np.int64
    assert g[0] == g[1] == g[2] and g[3] != g[0] and g[4] != g[3]
    with pytest.raises(ValueError):
        sb.assign_spatial_groups(rows, cols, 0)


def test_choose_group_folds():
    y = np.array([0, 0, 1, 1])
    groups = np.array([0, 1, 2, 3])
    assert sb.choose_group_folds(y, groups, 5, "outer") == 2
    with pytest.raises(RuntimeError):
        sb.choose_group_folds(y, np.array([0, 0, 1, 1]), 5, "outer")  # 1 group per class


def _grouped_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_groups_per_class, per_group = 6, 25
    Xs, ys, gs = [], [], []
    for c in range(3):
        center = rng.uniform(20, 120, 5)
        for k in range(n_groups_per_class):
            Xs.append(rng.normal(center + rng.normal(0, 2, 5), 3.0, (per_group, 5)))
            ys.append(np.full(per_group, c))
            gs.append(np.full(per_group, c * 100 + k))
    return np.vstack(Xs).clip(1).astype(np.float32), np.concatenate(ys), np.concatenate(gs)


@pytest.mark.parametrize("model", ["SAM", "SID", "MIN_DIST", "MAHALANOBIS", "LDA"])
def test_evaluate_model_is_grouped_and_complete(model):
    X, y, groups = _grouped_problem()
    ctx = sb.FeatureContext(
        selected_band_indices=np.arange(5),
        selected_band_names=[f"b{i}" for i in range(5)],
        selected_wavelengths_nm=None,
        base_feature_count=5,
        feature_names=[f"b{i}" for i in range(5)],
        spec=sb.feature_preset("raw", []),
        index_definitions_used=[],
    )
    res = sb.evaluate_model(
        X, y, groups, ["A", "B", "C"], ctx, model,
        outer_folds=3, inner_folds=2, bootstrap_reps=5, n_jobs=1, seed=0, svm_cap=0, knn_cap=0,
    )
    assert (res.y_pred >= 0).all() and len(res.y_pred) == len(y)
    assert res.summary["OA"] > 0.9
    assert len(res.fold_metrics) == 3
    # Grouped design: every group is predicted in exactly one fold's test set.
    assert sum(f["n_test"] for f in res.fold_metrics) == len(y)
    assert "macro_F1_CI95_low" in res.summary


def test_pairwise_comparison_runs_and_is_antisymmetric_in_sign():
    X, y, groups = _grouped_problem()
    ctx = sb.FeatureContext(np.arange(5), [f"b{i}" for i in range(5)], None, 5,
                            [f"b{i}" for i in range(5)], sb.feature_preset("raw", []), [])
    results = [
        sb.evaluate_model(X, y, groups, ["A", "B", "C"], ctx, m, 3, 2, 0, 1, 0, 0, 0)
        for m in ("SAM", "MIN_DIST")
    ]
    table = sb.pairwise_bootstrap_comparisons(results, 20, 0)
    assert len(table) == 1
    assert table.loc[0, "delta_OA_CI95_low"] <= table.loc[0, "delta_OA_CI95_high"]


def test_bootstrap_is_quiet():
    y = np.array([0, 0, 1, 1, 2, 2])
    groups = np.array([0, 0, 1, 1, 2, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any sklearn warning would raise
        ci = sb.bootstrap_metric_ci(y, y.copy(), groups, np.arange(3), 30, 0)
    assert set(ci) >= {"OA", "macro_F1"}


# --------------------------------------------------------------------------- labels


def test_drop_conflicting_pixels():
    labels = sb.LabelSet(
        rows=np.array([1, 1, 2, 3, 3]),
        cols=np.array([1, 1, 2, 3, 3]),
        class_names=["A", "B"],
        class_ids=np.array([0, 1, 0, 0, 0]),  # (1,1) is in A and B; (3,3) duplicated in A
        colors=[(0, 0, 0), (1, 1, 1)],
        provided_groups=np.array([0, 1, 2, 3, 3]),
    )
    with pytest.warns(UserWarning, match="more than one class"):
        out = sb.drop_conflicting_pixels(labels)
    assert sorted(zip(out.rows.tolist(), out.cols.tolist())) == [(2, 2), (3, 3)]
    assert len(out.provided_groups) == len(out.rows)


def test_load_envi_pixeldef_xml_merges_duplicate_regions(tmp_path):
    xml = tmp_path / "rois.xml"
    xml.write_text(
        """<RegionsOfInterest version="1.1">
          <Region name="Sand" color="255,255,0"><PixelDef><Coordinates>1 2 3 4</Coordinates></PixelDef></Region>
          <Region name="Sand" color="1,2,3"><PixelDef><Coordinates>5 6</Coordinates></PixelDef></Region>
          <Region name="Kelp" color="0,200,0"><PixelDef><Coordinates>7 8 9 10</Coordinates></PixelDef></Region>
          <Region name="BackgroundBlack"><PixelDef><Coordinates>0 0</Coordinates></PixelDef></Region>
        </RegionsOfInterest>""",
        encoding="utf-8",
    )
    labels = sb.load_envi_pixeldef_xml(xml, {"backgroundblack"})
    assert labels.class_names == ["Sand", "Kelp"]
    assert labels.colors[0] == (255, 255, 0)  # first region's colour wins
    assert (labels.class_ids == 0).sum() == 3 and (labels.class_ids == 1).sum() == 2
    # ENVI PixelDef is "col row": "1 2" -> col=1,row=2
    assert (labels.rows[0], labels.cols[0]) == (2, 1)
    assert labels.provided_groups is None


def test_load_envi_xml_rejects_odd_coordinate_count(tmp_path):
    xml = tmp_path / "bad.xml"
    xml.write_text(
        '<RegionsOfInterest><Region name="A"><PixelDef><Coordinates>1 2 3</Coordinates>'
        "</PixelDef></Region></RegionsOfInterest>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Odd number"):
        sb.load_envi_pixeldef_xml(xml, set())


def test_load_samples_csv(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("row,col,class,group\n1,2,A,s1\n3,4,B,s2\n5,6,A,s1\n7,8,Skip,s3\n")
    labels = sb.load_samples_csv(csv_path, {"skip"})
    assert labels.class_names == ["A", "B"]
    assert labels.provided_groups.tolist() == [0, 1, 0]
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError):
        sb.load_samples_csv(bad, set())


def test_balanced_sampling_respects_cap_and_spreads_groups():
    class_ids = np.repeat([0, 1], 500)
    groups = np.tile(np.arange(10), 100)
    idx = sb.spatially_balanced_sample_indices(class_ids, groups, 50, seed=0)
    for c in (0, 1):
        sel = idx[class_ids[idx] == c]
        assert len(sel) == 50 and len(np.unique(groups[sel])) == 10


# --------------------------------------------------------------------------- config


def test_known_config_keys_cover_every_key_the_code_reads():
    src = (Path(sb.__file__)).read_text(encoding="utf-8")
    used = set(re.findall(r'config_value\([^)]*?,\s*config,\s*"([a-z_0-9]+)"', src))
    used |= set(re.findall(r'config\.get\(\s*"([a-z_0-9]+)"', src))
    used |= set(re.findall(r'config\[\s*"([a-z_0-9]+)"\s*\]', src))
    assert used, "regex found nothing; test is broken"
    assert used <= sb.KNOWN_CONFIG_KEYS, sorted(used - sb.KNOWN_CONFIG_KEYS)


def test_example_config_only_uses_known_keys():
    import json

    cfg = json.loads((Path(__file__).resolve().parents[1] / "examples" / "example_config.json").read_text())
    assert set(cfg) <= sb.KNOWN_CONFIG_KEYS
    assert not any("PhD" in str(v) for v in cfg.values()), "private paths must not ship"


def test_feature_specs_from_inputs_cli_overrides_config():
    specs = sb.feature_specs_from_inputs("mean_norm,raw_indices", "GLI", {"feature_sets": ["raw"]})
    assert [s.name for s in specs] == ["mean_norm", "raw_indices"]
    assert specs[1].indices == ["GLI"]
    specs = sb.feature_specs_from_inputs(None, None, {"feature_sets": [{"name": "x", "derivative": True}]})
    assert specs[0].derivative and specs[0].name == "x"
