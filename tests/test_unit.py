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


# --------------------------------------------------------------------------- spatial methods


def _ctx(bands: int = 5) -> "sb.FeatureContext":
    names = [f"b{i}" for i in range(bands)]
    return sb.FeatureContext(np.arange(bands), names, None, bands, names, sb.feature_preset("raw", []), [])


def test_merge_groups_by_proximity():
    # A: rows 0-9, B: rows 10-19 (touching A), D: rows 20-29 (touching B), C: far away.
    rows = np.r_[np.arange(0, 10), np.arange(10, 20), np.arange(20, 30), [500, 501]]
    cols = np.zeros_like(rows)
    groups = np.r_[np.zeros(10), np.ones(10), np.full(10, 2), [3, 3]].astype(np.int64)
    merged = sb.merge_groups_by_proximity(groups, rows, cols, link_px=2)
    assert merged.ndim == 1 and merged.dtype == np.int64
    assert len(np.unique(merged[:30])) == 1, "touching groups must merge transitively"
    assert merged[30] != merged[0] and merged[30] == merged[31]
    # Gap of 5 px: not linked at 2 px, linked at 6 px.
    rows2 = np.r_[np.arange(0, 5), np.arange(10, 15)]
    g2 = np.r_[np.zeros(5), np.ones(5)].astype(np.int64)
    cols2 = np.zeros_like(rows2)
    assert len(np.unique(sb.merge_groups_by_proximity(g2, rows2, cols2, 2))) == 2
    assert len(np.unique(sb.merge_groups_by_proximity(g2, rows2, cols2, 6))) == 1
    # Off by default.
    assert np.array_equal(sb.merge_groups_by_proximity(groups, rows, cols, 0), groups)


def test_merge_groups_diagonal_neighbours_and_never_splits():
    rows = np.array([0, 1, 40, 41])
    cols = np.array([0, 1, 40, 41])
    groups = np.array([0, 1, 2, 3])
    merged = sb.merge_groups_by_proximity(groups, rows, cols, 3)
    assert merged[0] == merged[1] and merged[2] == merged[3] and merged[0] != merged[2]
    # Merging can only coarsen: pixels sharing a group before still share it after.
    rng = np.random.default_rng(0)
    r, c = rng.integers(0, 200, 300), rng.integers(0, 200, 300)
    g = (r // 50) * 4 + c // 50
    m = sb.merge_groups_by_proximity(g, r, c, 5)
    for gid in np.unique(g):
        assert len(np.unique(m[g == gid])) == 1


def test_buffer_train_indices():
    coords = np.column_stack([np.arange(10.0), np.zeros(10)])
    train, test = np.array([0, 1, 2, 3, 4, 5, 6]), np.array([7, 8, 9])
    kept, min_d, dropped = sb.buffer_train_indices(train, test, coords, buffer_px=2.0)
    assert min_d == 1.0 and dropped == 2 and kept.tolist() == [0, 1, 2, 3, 4]  # 5, 6 are within 2 px of 7
    kept, min_d, dropped = sb.buffer_train_indices(train, test, coords, buffer_px=0)
    assert dropped == 0 and len(kept) == 7 and min_d == 1.0
    assert sb.buffer_train_indices(train, test, None, 5.0)[0].tolist() == train.tolist()


def test_holm_adjust_and_bootstrap_p():
    adj = sb.holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert np.allclose(sb.holm_adjust([0.5, 0.9]), [1.0, 1.0])
    assert np.isnan(sb.holm_adjust([np.nan])[0])
    rng = np.random.default_rng(0)
    assert sb.bootstrap_p_value(rng.normal(0, 1, 2000)) > 0.5
    assert sb.bootstrap_p_value(np.full(199, 0.3)) == pytest.approx(2 * 1 / 200)
    assert np.isnan(sb.bootstrap_p_value([]))


def test_evaluate_model_repeats_and_diagnostics():
    X, y, groups = _grouped_problem()
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, (len(y), 2))
    res = sb.evaluate_model(
        X, y, groups, ["A", "B", "C"], _ctx(), "MIN_DIST", 3, 2, 10, 1, 0, 0, 0,
        coords=coords, buffer_px=0.0, repeats=3,
    )
    assert res.n_repeats == 3 and res.y_pred_repeats.shape == (3, len(y))
    assert len(res.fold_metrics) == 9 and {f["repeat"] for f in res.fold_metrics} == {1, 2, 3}
    assert "OA_repeat_std" in res.summary and res.summary["min_train_test_dist_px"] >= 0
    assert np.array_equal(res.y_pred, res.y_pred_repeats[0])
    # Different repeats use different fold assignments.
    assert not np.array_equal(res.y_pred_repeats[0], res.y_pred_repeats[1]) or res.summary["OA"] == 1.0


def test_single_repeat_matches_previous_behaviour():
    """repeats=1 must be the plain nested CV: same folds as repeats=3's first repeat."""
    X, y, groups = _grouped_problem()
    a = sb.evaluate_model(X, y, groups, ["A", "B", "C"], _ctx(), "SAM", 3, 2, 0, 1, 7, 0, 0)
    b = sb.evaluate_model(X, y, groups, ["A", "B", "C"], _ctx(), "SAM", 3, 2, 0, 1, 7, 0, 0, repeats=3)
    assert np.array_equal(a.y_pred, b.y_pred_repeats[0])


def test_buffer_removes_training_pixels_and_is_reported():
    X, y, groups = _grouped_problem()
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 60, (len(y), 2))
    res = sb.evaluate_model(
        X, y, groups, ["A", "B", "C"], _ctx(), "LDA", 3, 2, 0, 1, 0, 0, 0, coords=coords, buffer_px=3.0
    )
    assert all(f["n_train_dropped_by_buffer"] > 0 for f in res.fold_metrics)
    assert all(f["n_train"] == f["n_train_before_buffer"] - f["n_train_dropped_by_buffer"] for f in res.fold_metrics)


def test_oversized_buffer_gives_a_clear_error():
    X, y, groups = _grouped_problem()
    coords = np.random.default_rng(2).uniform(0, 10, (len(y), 2))
    with pytest.raises(RuntimeError, match="buffer_px"):
        sb.evaluate_model(
            X, y, groups, ["A", "B", "C"], _ctx(), "LDA", 3, 2, 0, 1, 0, 0, 0, coords=coords, buffer_px=50.0
        )


def test_pairwise_table_has_holm_adjusted_p_values():
    X, y, groups = _grouped_problem()
    results = [
        sb.evaluate_model(X, y, groups, ["A", "B", "C"], _ctx(), m, 3, 2, 0, 1, 0, 0, 0, repeats=2)
        for m in ("SAM", "MIN_DIST", "LDA")
    ]
    table = sb.pairwise_bootstrap_comparisons(results, 40, 0)
    assert len(table) == 3
    for col in ("delta_OA_p_boot", "delta_OA_p_holm", "delta_macro_F1_p_boot", "delta_macro_F1_p_holm"):
        assert col in table and table[col].between(0, 1).all()
    assert (table["delta_OA_p_holm"] >= table["delta_OA_p_boot"] - 1e-12).all()


def test_new_config_keys_are_known():
    assert {"buffer_px", "repeats", "group_merge_px"} <= sb.KNOWN_CONFIG_KEYS


def _leaky_field(seed: int, size: int = 120, bands: int = 6, delta: float = 0.3, sigma: float = 4.0):
    """Two classes with only a WEAK real signal plus spatially correlated noise.

    Noise is smooth across neighbouring pixels, so neighbours act as near-duplicates:
    a flexible model can 'recognise' a test pixel's neighbour in the training set.
    """
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(seed)
    noise = np.stack([ndimage.gaussian_filter(rng.normal(size=(size, size)), sigma) for _ in range(bands)])
    noise /= noise.reshape(bands, -1).std(axis=1)[:, None, None]
    noise += rng.normal(0, 0.2, noise.shape)
    half = size // 2
    cls = (np.arange(size)[:, None] >= half).astype(int) ^ (np.arange(size)[None, :] >= half).astype(int)
    img = noise + delta * (2 * cls - 1)[None]
    cell, per_cell = 20, 100
    rr, cc = [], []
    for br in range(size // cell):
        for bc in range(size // cell):
            rr.append(rng.integers(br * cell, (br + 1) * cell, per_cell))
            cc.append(rng.integers(bc * cell, (bc + 1) * cell, per_cell))
    rows, cols = np.concatenate(rr), np.concatenate(cc)
    return (
        img[:, rows, cols].T.astype(np.float32),
        cls[rows, cols],
        (rows // cell) * (size // cell) + cols // cell,
        np.column_stack([rows, cols]).astype(float),
    )


def test_spatial_buffer_removes_neighbour_leakage():
    """With grid-block groups, kNN looks better than the data allow; a buffer cures that.

    Bayes-optimal accuracy for this field is about 0.64 (6 bands, delta=0.3 sigma). Without a
    buffer, kNN exploits neighbouring training pixels and scores clearly higher.
    """
    acc = {0: [], 9: []}
    for seed in range(3):
        X, y, groups, coords = _leaky_field(seed)
        for buf in acc:
            res = sb.evaluate_model(
                X, y, groups, ["A", "B"], _ctx(6), "KNN", 4, 2, 0, 1, seed, 0, 0, coords=coords, buffer_px=buf
            )
            acc[buf].append(res.summary["OA"])
    unbuffered, buffered = float(np.mean(acc[0])), float(np.mean(acc[9]))
    assert unbuffered - buffered > 0.04, (unbuffered, buffered)
    assert buffered < 0.70  # close to the true ceiling, not inflated


# --------------------------------------------------------------------------- area weighting


def test_area_weighted_metrics_hand_computed():
    cm = np.array([[90, 10], [20, 80]])
    summary, table = sb.area_weighted_metrics(cm, np.array([0.9, 0.1]), ["A", "B"])
    # p = [[0.81, 0.09], [0.02, 0.08]] -> OA 0.89; PA = [0.9, 0.8]; UA = [0.81/0.83, 0.08/0.17]
    assert summary["OA_area_weighted"] == pytest.approx(0.89)
    assert table["recall_PA_area_weighted"].tolist() == pytest.approx([0.9, 0.8])
    assert table["precision_UA_area_weighted"].tolist() == pytest.approx([0.81 / 0.83, 0.08 / 0.17])
    chance = 0.9 * 0.83 + 0.1 * 0.17
    assert summary["kappa_area_weighted"] == pytest.approx((0.89 - chance) / (1 - chance))


def test_area_weighting_with_sample_proportions_reproduces_plain_metrics():
    cm = np.array([[50, 5, 0], [10, 30, 10], [0, 5, 40]])
    prev = cm.sum(axis=1) / cm.sum()
    summary, _ = sb.area_weighted_metrics(cm, prev, ["A", "B", "C"])
    assert summary["OA_area_weighted"] == pytest.approx(np.trace(cm) / cm.sum())


def test_rare_class_errors_matter_less_when_class_is_rare_in_the_scene():
    cm = np.array([[100, 0], [50, 50]])  # class B is poorly recognised (PA 0.5)
    common_b, _ = sb.area_weighted_metrics(cm, np.array([0.5, 0.5]), ["A", "B"])
    rare_b, _ = sb.area_weighted_metrics(cm, np.array([0.99, 0.01]), ["A", "B"])
    assert rare_b["OA_area_weighted"] > common_b["OA_area_weighted"]


def test_load_class_prevalence(tmp_path):
    f = tmp_path / "prev.csv"
    f.write_text("class,area\nsand,600\nKELP,200\nextra,5\n")
    out = sb.load_class_prevalence(f, ["Sand", "Kelp"])
    assert out.tolist() == pytest.approx([0.75, 0.25])
    with pytest.raises(ValueError, match="no value"):
        sb.load_class_prevalence(f, ["Sand", "Rock"])
    bad = tmp_path / "bad.csv"
    bad.write_text("name,count\nA,1\n")
    with pytest.raises(ValueError):
        sb.load_class_prevalence(bad, ["A"])


# --------------------------------------------------------------------------- hold-out


def test_split_holdout_groups_is_group_disjoint_and_class_complete():
    _, y, groups = _grouped_problem()
    dev, hold = sb.split_holdout_groups(y, groups, 0.34, seed=0)
    assert set(groups[dev]).isdisjoint(set(groups[hold]))
    assert len(dev) + len(hold) == len(y)
    for c in np.unique(y):
        assert np.unique(groups[dev][y[dev] == c]).size >= 2
        assert np.unique(groups[hold][y[hold] == c]).size >= 1
    with pytest.raises(ValueError):
        sb.split_holdout_groups(y, groups, 0.0, 0)
    with pytest.raises(ValueError):
        sb.split_holdout_groups(y, groups, 0.7, 0)


def test_split_holdout_rejects_too_few_groups():
    y = np.repeat([0, 1], 20)
    groups = np.repeat([0, 1, 2, 3], 10)  # two groups per class only
    with pytest.raises(RuntimeError, match="holdout_fraction"):
        sb.split_holdout_groups(y, groups, 0.34, 0)


def test_evaluate_holdout_scores_unseen_groups():
    X, y, groups = _grouped_problem()
    coords = np.random.default_rng(3).uniform(0, 200, (len(y), 2))
    dev, hold = sb.split_holdout_groups(y, groups, 0.34, seed=1)
    record, per_class, cm = sb.evaluate_holdout(
        "LDA", X[dev], y[dev], groups[dev], coords[dev], X[hold], y[hold], groups[hold], coords[hold],
        ["A", "B", "C"], _ctx(), 2, 1, 0, 0, 0, buffer_px=2.0, bootstrap_reps=10,
    )
    assert record["n_holdout"] == len(hold) and record["n_holdout_groups"] == np.unique(groups[hold]).size
    assert record["OA"] > 0.9 and "macro_F1_CI95_low" in record
    assert record["n_train_dropped_by_buffer"] >= 0 and cm.sum() == len(hold)
    assert len(per_class) == 3


# --------------------------------------------------------------------------- rejection


def test_apply_rejection():
    codes = np.array([1, 2, 3, 2])
    conf = np.array([0.9, 0.2, 0.6, 0.49])
    assert sb.apply_rejection(codes, conf, 0.5, 9).tolist() == [1, 9, 3, 9]
    assert sb.apply_rejection(codes, conf, 0.0, 9) is codes


def test_new_config_keys_v13():
    assert {"class_prevalence", "holdout_fraction", "min_confidence"} <= sb.KNOWN_CONFIG_KEYS


def test_model_comparison_reports_group_count():
    X, y, groups = _grouped_problem()
    res = sb.evaluate_model(X, y, groups, ["A", "B", "C"], _ctx(), "SAM", 3, 2, 0, 1, 0, 0, 0)
    assert res.summary["min_groups_per_class"] == 6


def test_comparison_figure_survives_interval_that_excludes_the_estimate(tmp_path):
    """Regression: negative error bars used to crash the whole run at the plotting stage."""
    import pandas as pd

    comparison = pd.DataFrame(
        {
            "feature_set": ["raw", "raw"],
            "model": ["A", "B"],
            "macro_F1": [0.90, 0.80],
            "macro_F1_CI95_low": [0.95, 0.70],  # low > estimate for the first model
            "macro_F1_CI95_high": [0.99, 0.75],  # high < estimate for the second
            "OA": [0.9, 0.8],
            "OA_CI95_low": [0.85, 0.7],
            "OA_CI95_high": [0.95, 0.9],
        }
    )
    sb.save_model_comparison_figures(comparison, tmp_path)
    assert (tmp_path / "model_comparison_macro_F1.png").exists()
