#!/usr/bin/env python3
"""
Spectral Benchmark
==================
A reusable supervised classification and reporting framework for multispectral
and hyperspectral rasters.

Main capabilities
-----------------
- Reads GeoTIFF / ENVI / GDAL-readable multiband rasters with rasterio.
- Reads training labels from:
    * ENVI ROI PixelDef XML
    * CSV pixel samples (row, col, class[, group])
    * Integer label rasters
- Spatially grouped nested cross-validation to reduce pixel-neighbour leakage.
- Classifiers:
    SVM, RF, HGB, LDA, KNN, SAM, SID, Minimum Distance,
    regularized pooled Mahalanobis, and regularized QDA.
- Feature-set experiments:
    raw spectrum, per-pixel mean-normalized spectrum, optional spectral indices,
    and wavelength-normalized first derivatives.
- Conservative band diagnostics and class spectral statistics.
- Publication-oriented CSV tables and PNG figures.
- Optional tile-based full-raster classification maps without loading the
  complete image into memory.

The framework is intentionally dataset-agnostic. Dataset-specific operations
such as CHM fusion, camera co-registration, or underwater radiometric correction
should be performed upstream or added as separate feature sources.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# Limit nested BLAS parallelism before importing NumPy / scikit-learn.
for _env_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_name, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


VERSION = "1.3.0"

# Every key read from a JSON config file. Unknown keys trigger a warning so that
# typos (e.g. "group_block_size" instead of "group_block_size_px") are not
# silently ignored.
KNOWN_CONFIG_KEYS = frozenset(
    {
        "image", "output_dir", "seed", "outer_folds", "inner_folds", "bootstrap",
        "pairwise_bootstrap", "max_samples_per_class", "group_block_size_px", "n_jobs",
        "svm_train_cap", "knn_train_cap", "tile_size", "index_tolerance_nm",
        "band_analysis_samples", "exclude_classes", "models", "feature_sets", "indices",
        "wavelengths_nm", "keep_bands", "drop_bands", "keep_wavelengths_nm",
        "drop_wavelengths_nm", "roi_xml", "samples_csv", "label_raster", "class_map",
        "map", "map_top", "write_confidence", "buffer_px", "repeats", "group_merge_px",
        "class_prevalence", "holdout_fraction", "min_confidence",
    }
)
DEFAULT_MODELS = (
    "SVM",
    "RF",
    "HGB",
    "LDA",
    "KNN",
    "SAM",
    "SID",
    "MIN_DIST",
    "MAHALANOBIS",
    "QDA",
)


# -----------------------------------------------------------------------------
# Configuration data structures
# -----------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    name: str
    normalization: str = "none"  # none | mean
    indices: list[str] = field(default_factory=list)
    derivative: bool = False


@dataclass
class LabelSet:
    rows: np.ndarray
    cols: np.ndarray
    class_names: list[str]
    class_ids: np.ndarray
    colors: list[tuple[int, int, int]]
    provided_groups: np.ndarray | None = None


@dataclass
class FeatureContext:
    selected_band_indices: np.ndarray
    selected_band_names: list[str]
    selected_wavelengths_nm: np.ndarray | None
    base_feature_count: int
    feature_names: list[str]
    spec: FeatureSpec
    index_definitions_used: list[str]


@dataclass
class ValidationResult:
    feature_set: str
    model: str
    y_true: np.ndarray
    y_pred: np.ndarray
    groups: np.ndarray
    fold_metrics: list[dict[str, Any]]
    best_params_per_fold: list[dict[str, Any]]
    summary: dict[str, Any]
    per_class: pd.DataFrame
    confusion: np.ndarray
    confusion_norm: np.ndarray
    # (n_repeats, n_samples) out-of-fold predictions; y_pred is repeat 1.
    y_pred_repeats: np.ndarray | None = None
    n_repeats: int = 1


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "item"


def parse_csv_list(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_int_list(value: str | None) -> list[int]:
    return [int(x) for x in parse_csv_list(value)]


def parse_float_list(value: str | None) -> list[float]:
    return [float(x) for x in parse_csv_list(value)]


def parse_number_text(text: str | None, dtype: Any = np.float64) -> np.ndarray:
    """Parse whitespace-separated numbers from XML text.

    Replaces ``np.fromstring(..., sep=" ")``, which silently stops at the first
    unparsable token and is deprecated in NumPy.
    """
    tokens = (text or "").split()
    if not tokens:
        return np.empty(0, dtype=dtype)
    values = np.asarray(tokens, dtype=np.float64)
    return values if np.dtype(dtype) == np.float64 else np.rint(values).astype(dtype)


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b)), dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    np.divide(aa, bb, out=out, where=np.abs(bb) > eps)
    return out


def default_colors(n: int) -> list[tuple[int, int, int]]:
    cmap = plt.get_cmap("tab20")
    colors: list[tuple[int, int, int]] = []
    for i in range(n):
        r, g, b, _ = cmap(i % cmap.N)
        colors.append((int(round(255 * r)), int(round(255 * g)), int(round(255 * b))))
    return colors


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def library_versions() -> dict[str, str]:
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rasterio": rasterio.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }


# -----------------------------------------------------------------------------
# Raster metadata and wavelength discovery
# -----------------------------------------------------------------------------


def _numbers_from_braced_text(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def infer_wavelengths_nm(ds: rasterio.io.DatasetReader) -> np.ndarray | None:
    """Infer band-centre wavelengths in nm from common GDAL/ENVI metadata."""
    candidates: list[list[float]] = []

    root_tags = ds.tags()
    for key in ("WAVELENGTH", "wavelength"):
        vals = _numbers_from_braced_text(root_tags.get(key))
        if len(vals) == ds.count:
            candidates.append(vals)

    for ns in ("ENVI", "envi"):
        try:
            tags = ds.tags(ns=ns)
        except Exception:
            tags = {}
        for key in ("wavelength", "WAVELENGTH"):
            vals = _numbers_from_braced_text(tags.get(key))
            if len(vals) == ds.count:
                candidates.append(vals)

    if candidates:
        arr = np.asarray(candidates[0], dtype=float)
        # If values look like micrometres, convert to nanometres.
        if np.nanmax(arr) < 20:
            arr *= 1000.0
        return arr

    per_band: list[float] = []
    ok = True
    for band in range(1, ds.count + 1):
        tags = ds.tags(band)
        val: float | None = None
        for key in ("WAVELENGTH_NM", "WAVELENGTH", "wavelength"):
            if key in tags:
                vals = _numbers_from_braced_text(tags[key])
                if vals:
                    val = vals[0]
                    break
        if val is None:
            imagery = ds.tags(band, ns="IMAGERY")
            vals = _numbers_from_braced_text(imagery.get("CENTRAL_WAVELENGTH_UM"))
            if vals:
                val = vals[0] * 1000.0
        if val is None:
            ok = False
            break
        per_band.append(float(val))
    if ok and len(per_band) == ds.count:
        return np.asarray(per_band, dtype=float)

    descriptions = ds.descriptions or ()
    if len(descriptions) == ds.count:
        vals: list[float] = []
        for desc in descriptions:
            if not desc:
                vals = []
                break
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:nm|nanomet)", desc, flags=re.I)
            if not m:
                vals = []
                break
            vals.append(float(m.group(1)))
        if len(vals) == ds.count:
            return np.asarray(vals, dtype=float)

    return None


def infer_band_names(ds: rasterio.io.DatasetReader, wavelengths_nm: np.ndarray | None) -> list[str]:
    names: list[str] = []
    descriptions = ds.descriptions or ()
    for i in range(ds.count):
        desc = descriptions[i] if i < len(descriptions) else None
        if desc:
            names.append(str(desc))
        elif wavelengths_nm is not None:
            names.append(f"{wavelengths_nm[i]:g} nm")
        else:
            names.append(f"Band {i + 1}")
    return names


def resolve_band_selection(
    count: int,
    wavelengths_nm: np.ndarray | None,
    keep_bands_1based: Sequence[int],
    drop_bands_1based: Sequence[int],
    keep_wavelengths_nm: Sequence[float],
    drop_wavelengths_nm: Sequence[float],
    wavelength_tolerance_nm: float,
) -> np.ndarray:
    selected = np.ones(count, dtype=bool)

    if keep_bands_1based and keep_wavelengths_nm:
        raise ValueError("Use either keep-bands or keep-wavelengths, not both.")

    if keep_bands_1based:
        selected[:] = False
        for b in keep_bands_1based:
            if b < 1 or b > count:
                raise ValueError(f"Band index {b} is outside 1..{count}.")
            selected[b - 1] = True

    for b in drop_bands_1based:
        if b < 1 or b > count:
            raise ValueError(f"Band index {b} is outside 1..{count}.")
        selected[b - 1] = False

    if keep_wavelengths_nm or drop_wavelengths_nm:
        if wavelengths_nm is None:
            raise ValueError("Wavelength-based band selection requires wavelength metadata or --wavelengths.")

        def nearest_index(target: float) -> int:
            j = int(np.argmin(np.abs(wavelengths_nm - target)))
            delta = abs(float(wavelengths_nm[j]) - target)
            if delta > wavelength_tolerance_nm:
                raise ValueError(
                    f"No band within {wavelength_tolerance_nm:g} nm of requested {target:g} nm "
                    f"(nearest: {wavelengths_nm[j]:g} nm)."
                )
            return j

        if keep_wavelengths_nm:
            selected[:] = False
            for w in keep_wavelengths_nm:
                selected[nearest_index(float(w))] = True
        for w in drop_wavelengths_nm:
            selected[nearest_index(float(w))] = False

    idx = np.flatnonzero(selected)
    if len(idx) == 0:
        raise ValueError("Band selection removed every band.")
    return idx


# -----------------------------------------------------------------------------
# Training label readers
# -----------------------------------------------------------------------------


def load_envi_pixeldef_xml(
    path: Path,
    exclude_classes: set[str],
    transform: rasterio.Affine | None = None,
) -> LabelSet:
    """Read exact ENVI ROI PixelDef pixels and, when available, polygon groups.

    ENVI PixelDef coordinates are interpreted as x/y = column/row pixel indices.
    When GeometryDef polygons and the raster transform are available, each pixel
    is assigned to its source polygon so grouped cross-validation can keep whole
    ROIs together. Pixels that cannot be matched to a polygon receive group -1
    and are assigned to spatial fallback groups later in the pipeline.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    class_names: list[str] = []
    class_index: dict[str, int] = {}
    colors: list[tuple[int, int, int]] = []
    next_group = 0
    any_geometry_groups = False

    excluded_lower = {x.casefold() for x in exclude_classes}

    for region in root.findall(".//Region"):
        name = (region.attrib.get("name") or "Unnamed").strip()
        if name.casefold() in excluded_lower:
            continue

        coord_nodes = region.findall("./PixelDef/Coordinates")
        if not coord_nodes:
            continue

        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        for node in coord_nodes:
            arr = parse_number_text(node.text, np.int64)
            if arr.size == 0:
                continue
            if arr.size % 2:
                raise ValueError(f"Odd number of PixelDef coordinate values in class {name!r}.")
            pairs = arr.reshape(-1, 2)
            col_parts.append(pairs[:, 0])
            row_parts.append(pairs[:, 1])

        if not row_parts:
            continue

        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)
        packed = (rows.astype(np.int64) << 32) ^ (cols.astype(np.int64) & 0xFFFFFFFF)
        _, unique_idx = np.unique(packed, return_index=True)
        rows = rows[unique_idx]
        cols = cols[unique_idx]

        groups_region = np.full(rows.size, -1, dtype=np.int64)
        if transform is not None:
            inverse = ~transform
            polygon_nodes = region.findall("./GeometryDef/Polygon")
            for polygon_node in polygon_nodes:
                coord_node = polygon_node.find("./Exterior/LinearRing/Coordinates")
                if coord_node is None or not (coord_node.text or "").strip():
                    continue
                world = parse_number_text(coord_node.text, np.float64)
                if world.size < 6 or world.size % 2:
                    continue
                xy = world.reshape(-1, 2)
                pixel_vertices = np.asarray([inverse * (float(x), float(y)) for x, y in xy], dtype=float)
                polygon = MplPath(pixel_vertices)
                min_col, min_row = np.floor(pixel_vertices.min(axis=0) - 1).astype(int)
                max_col, max_row = np.ceil(pixel_vertices.max(axis=0) + 1).astype(int)
                candidate = np.flatnonzero(
                    (groups_region < 0)
                    & (cols >= min_col)
                    & (cols <= max_col)
                    & (rows >= min_row)
                    & (rows <= max_row)
                )
                if candidate.size == 0:
                    next_group += 1
                    continue
                points = np.column_stack([cols[candidate] + 0.5, rows[candidate] + 0.5])
                inside = polygon.contains_points(points, radius=1e-7)
                if np.any(inside):
                    groups_region[candidate[inside]] = next_group
                    any_geometry_groups = True
                next_group += 1

        # ENVI allows several regions with the same name; they are one class.
        key = name.casefold()
        if key in class_index:
            cid = class_index[key]
        else:
            cid = len(class_names)
            class_index[key] = cid
            class_names.append(name)
            color_text = region.attrib.get("color", "")
            try:
                rgb = tuple(int(x.strip()) for x in color_text.split(","))
                if len(rgb) != 3:
                    raise ValueError
                rgb = tuple(int(np.clip(v, 0, 255)) for v in rgb)
            except Exception:
                rgb = default_colors(cid + 1)[-1]
            colors.append(rgb)  # type: ignore[arg-type]
        all_rows.append(rows)
        all_cols.append(cols)
        all_ids.append(np.full(rows.size, cid, dtype=np.int32))
        all_groups.append(groups_region)

    if not class_names:
        raise RuntimeError(f"No usable PixelDef classes found in {path}.")

    return LabelSet(
        rows=np.concatenate(all_rows),
        cols=np.concatenate(all_cols),
        class_names=class_names,
        class_ids=np.concatenate(all_ids),
        colors=colors,
        provided_groups=np.concatenate(all_groups) if any_geometry_groups else None,
    )

def load_samples_csv(path: Path, exclude_classes: set[str]) -> LabelSet:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    required = {"row", "col", "class"}
    missing = [x for x in required if x not in lower]
    if missing:
        raise ValueError(f"Sample CSV must contain columns row, col, class. Missing: {missing}")

    row_col = lower["row"]
    col_col = lower["col"]
    class_col = lower["class"]
    group_col = lower.get("group")

    df = df.copy()
    df[class_col] = df[class_col].astype(str)
    if exclude_classes:
        excluded_lower = {x.casefold() for x in exclude_classes}
        df = df[~df[class_col].str.casefold().isin(excluded_lower)]

    class_names = list(dict.fromkeys(df[class_col].tolist()))
    class_lookup = {name: i for i, name in enumerate(class_names)}
    class_ids = df[class_col].map(class_lookup).to_numpy(np.int32)

    colors = default_colors(len(class_names))
    if all(x in lower for x in ("r", "g", "b")):
        color_by_class: dict[str, tuple[int, int, int]] = {}
        for name in class_names:
            sub = df[df[class_col] == name].iloc[0]
            color_by_class[name] = (
                int(np.clip(sub[lower["r"]], 0, 255)),
                int(np.clip(sub[lower["g"]], 0, 255)),
                int(np.clip(sub[lower["b"]], 0, 255)),
            )
        colors = [color_by_class[x] for x in class_names]

    groups = None
    if group_col is not None:
        groups = pd.factorize(df[group_col].astype(str), sort=False)[0].astype(np.int64)

    return LabelSet(
        rows=df[row_col].to_numpy(np.int64),
        cols=df[col_col].to_numpy(np.int64),
        class_names=class_names,
        class_ids=class_ids,
        colors=colors,
        provided_groups=groups,
    )


def load_class_map(path: Path | None) -> dict[int, tuple[str, tuple[int, int, int] | None]]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if "code" not in lower or "name" not in lower:
        raise ValueError("Class-map CSV must contain code and name columns.")
    result: dict[int, tuple[str, tuple[int, int, int] | None]] = {}
    for _, row in df.iterrows():
        rgb = None
        if all(x in lower for x in ("r", "g", "b")):
            rgb = tuple(int(np.clip(row[lower[x]], 0, 255)) for x in ("r", "g", "b"))  # type: ignore[assignment]
        result[int(row[lower["code"]])] = (str(row[lower["name"]]), rgb)
    return result


def load_label_raster_samples(
    path: Path,
    max_pool_per_class: int,
    seed: int,
    class_map: dict[int, tuple[str, tuple[int, int, int] | None]],
    exclude_classes: set[str],
) -> LabelSet:
    """Uniformly sample labelled pixels from an integer label raster.

    Code 0 is treated as unlabelled. A bounded random-key reservoir prevents
    large dense label rasters from consuming excessive RAM.
    """
    rng = np.random.default_rng(seed)
    reservoirs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise ValueError("Label raster must have exactly one band.")
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window)
            codes = np.unique(arr)
            for code in codes:
                code_i = int(code)
                if code_i == 0:
                    continue
                rr, cc = np.nonzero(arr == code)
                if rr.size == 0:
                    continue
                rr = rr.astype(np.int64) + int(window.row_off)
                cc = cc.astype(np.int64) + int(window.col_off)
                keys = rng.random(rr.size)

                if code_i in reservoirs:
                    old_r, old_c, old_k = reservoirs[code_i]
                    rr = np.concatenate([old_r, rr])
                    cc = np.concatenate([old_c, cc])
                    keys = np.concatenate([old_k, keys])

                if rr.size > max_pool_per_class:
                    keep = np.argpartition(keys, max_pool_per_class - 1)[:max_pool_per_class]
                    rr, cc, keys = rr[keep], cc[keep], keys[keep]
                reservoirs[code_i] = (rr, cc, keys)

    if not reservoirs:
        raise RuntimeError(f"No non-zero labels found in {path}.")

    codes_sorted = sorted(reservoirs)
    class_names: list[str] = []
    colors: list[tuple[int, int, int]] = []
    allowed_codes: list[int] = []
    excluded_lower = {x.casefold() for x in exclude_classes}

    generated_colors = default_colors(len(codes_sorted))
    for i, code in enumerate(codes_sorted):
        mapped = class_map.get(code)
        name = mapped[0] if mapped else f"Class_{code}"
        if name.casefold() in excluded_lower:
            continue
        color = mapped[1] if mapped and mapped[1] is not None else generated_colors[i]
        allowed_codes.append(code)
        class_names.append(name)
        colors.append(color)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    ids_list: list[np.ndarray] = []
    for cid, code in enumerate(allowed_codes):
        rr, cc, _ = reservoirs[code]
        rows_list.append(rr)
        cols_list.append(cc)
        ids_list.append(np.full(rr.size, cid, np.int32))

    return LabelSet(
        rows=np.concatenate(rows_list),
        cols=np.concatenate(cols_list),
        class_names=class_names,
        class_ids=np.concatenate(ids_list),
        colors=colors,
        provided_groups=None,
    )


def check_label_bounds(labels: LabelSet, height: int, width: int) -> LabelSet:
    valid = (
        (labels.rows >= 0)
        & (labels.rows < height)
        & (labels.cols >= 0)
        & (labels.cols < width)
    )
    n_bad = int((~valid).sum())
    if n_bad:
        warnings.warn(f"Ignoring {n_bad} labelled pixels outside raster bounds.")
    return LabelSet(
        rows=labels.rows[valid],
        cols=labels.cols[valid],
        class_names=labels.class_names,
        class_ids=labels.class_ids[valid],
        colors=labels.colors,
        provided_groups=labels.provided_groups[valid] if labels.provided_groups is not None else None,
    )


def drop_conflicting_pixels(labels: LabelSet) -> LabelSet:
    """Remove pixels that appear in more than one class.

    Overlapping ROIs would put the same spectrum into the training and the
    validation partition with different labels, so such pixels are dropped.
    Pixels repeated within the *same* class are kept once.
    """
    packed = (labels.rows.astype(np.int64) << 32) ^ (labels.cols.astype(np.int64) & 0xFFFFFFFF)
    order = np.lexsort((labels.class_ids, packed))
    p, c = packed[order], labels.class_ids[order]
    same_pixel = p[1:] == p[:-1]
    differs = same_pixel & (c[1:] != c[:-1])
    conflict_keys = np.unique(p[1:][differs])
    keep = ~np.isin(packed, conflict_keys)
    # Exact duplicates (same pixel, same class): keep the first occurrence only.
    _, first = np.unique(np.column_stack([packed, labels.class_ids]), axis=0, return_index=True)
    first_mask = np.zeros(len(packed), dtype=bool)
    first_mask[first.reshape(-1)] = True
    keep &= first_mask
    if conflict_keys.size:
        warnings.warn(
            f"Dropped {conflict_keys.size} pixel(s) labelled with more than one class (overlapping ROIs)."
        )
    if keep.all():
        return labels
    return LabelSet(
        rows=labels.rows[keep],
        cols=labels.cols[keep],
        class_names=labels.class_names,
        class_ids=labels.class_ids[keep],
        colors=labels.colors,
        provided_groups=labels.provided_groups[keep] if labels.provided_groups is not None else None,
    )


def assign_spatial_groups(rows: np.ndarray, cols: np.ndarray, block_size_px: int) -> np.ndarray:
    if block_size_px <= 0:
        raise ValueError("group_block_size_px must be > 0.")
    gr = rows // block_size_px
    gc = cols // block_size_px
    pairs = np.column_stack([gr, gc])
    _, group_ids = np.unique(pairs, axis=0, return_inverse=True)
    # NumPy 2.0.0 returned a 2-D inverse for axis=0; ravel() is safe on every version.
    return np.asarray(group_ids).reshape(-1).astype(np.int64)


def merge_groups_by_proximity(
    groups: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    link_px: float,
) -> np.ndarray:
    """Merge groups that have pixels within ~``link_px`` pixels of each other.

    Independent polygons or blocks that touch each other still share spatially
    autocorrelated pixels, so putting them in different folds leaks information.
    Groups are merged transitively (connected components). Pixels are binned
    into cells of ``ceil(link_px)`` pixels; any two pixels closer than that are
    in the same or in neighbouring cells, so the merge is conservative.
    """
    if link_px <= 0 or len(groups) == 0:
        return groups
    cell = max(1, int(math.ceil(link_px)))
    cr = rows.astype(np.int64) // cell + 1
    cc = cols.astype(np.int64) // cell + 1
    stride = int(cc.max()) + 3
    ukeys, cell_of_pixel = np.unique(cr * stride + cc, return_inverse=True)
    cell_of_pixel = np.asarray(cell_of_pixel).reshape(-1)
    n_cells = len(ukeys)

    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):  # forward half of the 8-neighbourhood
        nk = ukeys + dr * stride + dc
        pos = np.minimum(np.searchsorted(ukeys, nk), n_cells - 1)
        hit = ukeys[pos] == nk
        src.append(np.flatnonzero(hit))
        dst.append(pos[hit])

    unique_groups, ginv = np.unique(groups, return_inverse=True)
    ginv = np.asarray(ginv).reshape(-1)
    pairs = np.unique(np.column_stack([cell_of_pixel, ginv]), axis=0)
    src.append(pairs[:, 0])
    dst.append(n_cells + pairs[:, 1])

    s_all, d_all = np.concatenate(src), np.concatenate(dst)
    n_nodes = n_cells + len(unique_groups)
    graph = coo_matrix((np.ones(len(s_all), dtype=np.int32), (s_all, d_all)), shape=(n_nodes, n_nodes))
    _, labels = connected_components(graph, directed=False)
    _, merged = np.unique(labels[n_cells + ginv], return_inverse=True)
    return np.asarray(merged).reshape(-1).astype(np.int64)


def spatially_balanced_sample_indices(
    class_ids: np.ndarray,
    groups: np.ndarray,
    max_per_class: int,
    seed: int,
) -> np.ndarray:
    """Sample each class while spreading selected pixels across spatial groups."""
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []

    for cid in np.unique(class_ids):
        idx_class = np.flatnonzero(class_ids == cid)
        if max_per_class <= 0 or idx_class.size <= max_per_class:
            selected.append(idx_class)
            continue

        groups_class = groups[idx_class]
        unique_groups = np.unique(groups_class)
        quota = max(1, int(math.ceil(max_per_class / len(unique_groups))))
        parts: list[np.ndarray] = []
        used = np.zeros(idx_class.size, dtype=bool)

        for g in rng.permutation(unique_groups):
            local_pos = np.flatnonzero(groups_class == g)
            take_n = min(quota, local_pos.size)
            chosen_pos = rng.choice(local_pos, size=take_n, replace=False)
            used[chosen_pos] = True
            parts.append(idx_class[chosen_pos])

        current = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if current.size > max_per_class:
            current = rng.choice(current, size=max_per_class, replace=False)
        elif current.size < max_per_class:
            remaining = idx_class[~used]
            if remaining.size:
                extra_n = min(max_per_class - current.size, remaining.size)
                current = np.concatenate([current, rng.choice(remaining, extra_n, replace=False)])
        selected.append(current)

    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


# -----------------------------------------------------------------------------
# Efficient raster sampling
# -----------------------------------------------------------------------------


def read_sample_spectra(
    ds: rasterio.io.DatasetReader,
    rows: np.ndarray,
    cols: np.ndarray,
    tile_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Read spectra at sparse pixel coordinates by grouping them into tiles."""
    n = len(rows)
    out = np.full((n, ds.count), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)

    tile_r = rows // tile_size
    tile_c = cols // tile_size
    keys = tile_r.astype(np.int64) * (int(math.ceil(ds.width / tile_size)) + 1) + tile_c
    order = np.argsort(keys)
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], len(order)]

    for s, e in zip(starts, ends):
        idx = order[s:e]
        tr = int(tile_r[idx[0]])
        tc = int(tile_c[idx[0]])
        r0 = tr * tile_size
        c0 = tc * tile_size
        h = min(tile_size, ds.height - r0)
        w = min(tile_size, ds.width - c0)
        window = Window(c0, r0, w, h)
        data = ds.read(window=window).astype(np.float32, copy=False)
        mask = ds.read_masks(1, window=window) > 0

        lr = rows[idx] - r0
        lc = cols[idx] - c0
        vals = data[:, lr, lc].T
        local_valid = mask[lr, lc] & np.isfinite(vals).all(axis=1)
        out[idx] = vals
        valid[idx] = local_valid

    return out, valid


# -----------------------------------------------------------------------------
# Features and spectral indices
# -----------------------------------------------------------------------------


INDEX_TARGETS = {
    "GLI": {"B": 450.0, "G": 550.0, "R": 650.0},
    "TGI": {"B": 480.0, "G": 550.0, "R": 670.0},
    "RGRI": {"G": 550.0, "R": 650.0},
    "VARI": {"B": 450.0, "G": 550.0, "R": 650.0},
    "NGRDI": {"G": 550.0, "R": 650.0},
    "BG_RATIO": {"B": 450.0, "G": 550.0},
    "GR_RATIO": {"G": 550.0, "R": 650.0},
    "NDVI": {"R": 650.0, "NIR": 850.0},
}


def _nearest_wavelength_column(
    wavelengths_nm: np.ndarray,
    target_nm: float,
    tolerance_nm: float,
) -> int | None:
    j = int(np.argmin(np.abs(wavelengths_nm - target_nm)))
    if abs(float(wavelengths_nm[j]) - target_nm) > tolerance_nm:
        return None
    return j


def compute_indices(
    spectrum: np.ndarray,
    wavelengths_nm: np.ndarray | None,
    requested: Sequence[str],
    tolerance_nm: float,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    if not requested:
        return [], [], []
    if wavelengths_nm is None:
        warnings.warn("Spectral indices were requested but wavelengths are unavailable; indices are skipped.")
        return [], [], []

    arrays: list[np.ndarray] = []
    names: list[str] = []
    definitions: list[str] = []

    for raw_name in requested:
        name = raw_name.upper().strip()
        if name not in INDEX_TARGETS:
            warnings.warn(f"Unknown index {raw_name!r}; skipping.")
            continue
        targets = INDEX_TARGETS[name]
        col_by_symbol: dict[str, int] = {}
        missing = False
        for symbol, target in targets.items():
            j = _nearest_wavelength_column(wavelengths_nm, target, tolerance_nm)
            if j is None:
                warnings.warn(
                    f"Index {name} skipped: no wavelength within {tolerance_nm:g} nm of {target:g} nm."
                )
                missing = True
                break
            col_by_symbol[symbol] = j
        if missing:
            continue

        def v(symbol: str) -> np.ndarray:
            return spectrum[:, col_by_symbol[symbol]].astype(np.float64, copy=False)

        if name == "GLI":
            B, G, R = v("B"), v("G"), v("R")
            value = safe_divide(2 * G - R - B, 2 * G + R + B)
        elif name == "TGI":
            B, G, R = v("B"), v("G"), v("R")
            value = -0.5 * (190.0 * (R - G) - 120.0 * (R - B))
        elif name == "RGRI":
            value = safe_divide(v("R"), v("G"))
        elif name == "VARI":
            B, G, R = v("B"), v("G"), v("R")
            value = safe_divide(G - R, G + R - B)
        elif name == "NGRDI":
            G, R = v("G"), v("R")
            value = safe_divide(G - R, G + R)
        elif name == "BG_RATIO":
            value = safe_divide(v("B"), v("G"))
        elif name == "GR_RATIO":
            value = safe_divide(v("G"), v("R"))
        elif name == "NDVI":
            R, NIR = v("R"), v("NIR")
            value = safe_divide(NIR - R, NIR + R)
        else:
            continue

        used_desc = ", ".join(
            f"{symbol}={wavelengths_nm[col_by_symbol[symbol]]:g}nm" for symbol in targets
        )
        arrays.append(np.asarray(value, dtype=np.float32))
        names.append(name)
        definitions.append(f"{name}: {used_desc}")

    return arrays, names, definitions


def build_features(
    raw_spectra: np.ndarray,
    selected_band_indices: np.ndarray,
    band_names: Sequence[str],
    wavelengths_nm: np.ndarray | None,
    spec: FeatureSpec,
    index_tolerance_nm: float,
) -> tuple[np.ndarray, FeatureContext]:
    base = raw_spectra[:, selected_band_indices].astype(np.float32, copy=True)
    selected_names = [band_names[i] for i in selected_band_indices]
    selected_wl = wavelengths_nm[selected_band_indices] if wavelengths_nm is not None else None

    if spec.normalization == "mean":
        denom = np.mean(base, axis=1, keepdims=True, dtype=np.float64)
        denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
        base = (base / denom).astype(np.float32)
    elif spec.normalization != "none":
        raise ValueError(f"Unknown normalization {spec.normalization!r}.")

    parts: list[np.ndarray] = [base]
    feature_names = list(selected_names)

    index_arrays, index_names, index_defs = compute_indices(
        base,
        selected_wl,
        spec.indices,
        index_tolerance_nm,
    )
    if index_arrays:
        parts.append(np.column_stack(index_arrays).astype(np.float32))
        feature_names.extend(index_names)

    if spec.derivative:
        if base.shape[1] < 2:
            warnings.warn(f"Feature set {spec.name!r}: derivative skipped because fewer than 2 bands remain.")
        else:
            diff = np.diff(base.astype(np.float64), axis=1)
            if selected_wl is not None:
                dw = np.diff(selected_wl.astype(np.float64))
                if np.any(np.abs(dw) < 1e-12):
                    raise ValueError("Duplicate wavelengths prevent wavelength-normalized derivatives.")
                diff = diff / dw[None, :]
                deriv_names = [
                    f"d({selected_wl[i]:g}-{selected_wl[i + 1]:g})/d_lambda"
                    for i in range(len(selected_wl) - 1)
                ]
            else:
                deriv_names = [f"dBand{i + 1}_Band{i + 2}" for i in range(base.shape[1] - 1)]
            parts.append(diff.astype(np.float32))
            feature_names.extend(deriv_names)

    X = np.column_stack(parts).astype(np.float32, copy=False)
    X[~np.isfinite(X)] = 0.0

    context = FeatureContext(
        selected_band_indices=selected_band_indices.copy(),
        selected_band_names=selected_names,
        selected_wavelengths_nm=selected_wl.copy() if selected_wl is not None else None,
        base_feature_count=base.shape[1],
        feature_names=feature_names,
        spec=spec,
        index_definitions_used=index_defs,
    )
    return X, context


def feature_preset(name: str, indices: list[str]) -> FeatureSpec:
    key = name.strip().lower()
    presets = {
        "raw": FeatureSpec("raw", "none", [], False),
        "mean_norm": FeatureSpec("mean_norm", "mean", [], False),
        "raw_indices": FeatureSpec("raw_indices", "none", indices, False),
        "mean_norm_indices": FeatureSpec("mean_norm_indices", "mean", indices, False),
        "raw_derivative": FeatureSpec("raw_derivative", "none", [], True),
        "mean_norm_derivative": FeatureSpec("mean_norm_derivative", "mean", [], True),
        "raw_indices_derivative": FeatureSpec("raw_indices_derivative", "none", indices, True),
        "mean_norm_indices_derivative": FeatureSpec(
            "mean_norm_indices_derivative", "mean", indices, True
        ),
    }
    if key not in presets:
        raise ValueError(
            f"Unknown feature-set preset {name!r}. Available: {', '.join(sorted(presets))}"
        )
    return presets[key]


# -----------------------------------------------------------------------------
# Classical spectral classifiers
# -----------------------------------------------------------------------------


class _NearestReferenceClassifier:
    """Shared behaviour for distance-to-class-reference classifiers.

    Subclasses implement ``fit`` (which must call ``_finish_fit``) and
    ``distance_matrix``. ``predict_proba`` converts distances to a *relative*
    score with a softmax whose scale is fixed at fit time from the training
    data. Fixing the scale matters for tiled mapping: a per-call scale would make
    the same pixel receive a different confidence depending on the tile it falls in.
    These scores are not calibrated probabilities.
    """

    _SCALE_SAMPLE = 20000

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def _finish_fit(self, X: np.ndarray) -> None:
        n = X.shape[0]
        if n > self._SCALE_SAMPLE:
            step = int(math.ceil(n / self._SCALE_SAMPLE))
            X = X[::step]
        d = self.distance_matrix(X)
        positive = d[np.isfinite(d) & (d > 0)]
        self.scale_ = float(np.median(positive)) if positive.size else 1.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmin(self.distance_matrix(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d = self.distance_matrix(X)
        return softmax(-d / max(self.scale_, 1e-12), axis=1)


class SpectralAngleMapper(_NearestReferenceClassifier):
    def __init__(self, spectral_cols: int):
        self.spectral_cols = int(spectral_cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralAngleMapper":
        Xs = np.asarray(X[:, : self.spectral_cols], dtype=np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([Xs[y == c].mean(axis=0) for c in self.classes_])
        self._finish_fit(X)
        return self

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        Xs = np.asarray(X[:, : self.spectral_cols], dtype=np.float64)
        num = Xs @ self.means_.T
        den = np.linalg.norm(Xs, axis=1)[:, None] * np.linalg.norm(self.means_, axis=1)[None, :]
        cosang = np.clip(safe_divide(num, np.maximum(den, 1e-15)), -1.0, 1.0)
        return np.arccos(cosang)


class SpectralInformationDivergence(_NearestReferenceClassifier):
    """Symmetric SID between each pixel and each class-mean spectral distribution."""

    def __init__(self, spectral_cols: int, epsilon: float = 1e-12):
        self.spectral_cols = int(spectral_cols)
        self.epsilon = float(epsilon)

    def _prob(self, X: np.ndarray) -> np.ndarray:
        Xs = np.asarray(X[:, : self.spectral_cols], dtype=np.float64)
        # SID requires non-negative spectra. Shift individual spectra only when necessary.
        minv = np.min(Xs, axis=1, keepdims=True)
        Xs = Xs - np.minimum(minv, 0.0)
        Xs = np.clip(Xs, self.epsilon, None)
        return Xs / np.sum(Xs, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralInformationDivergence":
        P = self._prob(X)
        self.classes_ = np.unique(y)
        means = np.vstack([P[y == c].mean(axis=0) for c in self.classes_])
        means = np.clip(means, self.epsilon, None)
        self.means_ = means / means.sum(axis=1, keepdims=True)
        self._finish_fit(X)
        return self

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        # SID(P, Q) = sum (P - Q)(log P - log Q)
        #           = sum P log P + sum Q log Q - P . log Q - Q . log P
        # Evaluated with matrix products: O(n*k) memory instead of the O(n*k*p)
        # tensor that a direct broadcast would need (GBs for hyperspectral tiles).
        P = self._prob(X)
        Q = self.means_
        logP = np.log(P)
        logQ = np.log(Q)
        self_p = np.sum(P * logP, axis=1)[:, None]
        self_q = np.sum(Q * logQ, axis=1)[None, :]
        d = self_p + self_q - P @ logQ.T - logP @ Q.T
        return np.maximum(d, 0.0)


class MinimumDistanceClassifier(_NearestReferenceClassifier):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MinimumDistanceClassifier":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        self._finish_fit(X)
        return self

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        x2 = np.sum(X * X, axis=1)[:, None]
        m2 = np.sum(self.means_ * self.means_, axis=1)[None, :]
        d2 = np.maximum(x2 + m2 - 2.0 * (X @ self.means_.T), 0.0)
        return np.sqrt(d2)


class PooledMahalanobisClassifier(_NearestReferenceClassifier):
    """Nearest class mean under a shrinkage *pooled within-class* covariance metric.

    The covariance is estimated from class-centred residuals, so between-class
    scatter does not inflate it (using the total covariance would down-weight
    exactly the directions that separate the classes).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PooledMahalanobisClassifier":
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        residuals = np.empty_like(X)
        for c, mean in zip(self.classes_, self.means_):
            residuals[y == c] = X[y == c] - mean
        covariance = LedoitWolf(assume_centered=True).fit(residuals).covariance_
        self.precision_ = np.linalg.pinv(covariance, hermitian=True)
        self._finish_fit(X)
        return self

    def distance_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], len(self.classes_)), dtype=np.float64)
        for j, mean in enumerate(self.means_):
            delta = X - mean
            out[:, j] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, self.precision_, delta), 0.0))
        return out


# -----------------------------------------------------------------------------
# Model factory and grouped nested validation
# -----------------------------------------------------------------------------


def build_model(model_name: str, n_features: int, spectral_cols: int, seed: int):
    name = model_name.upper()
    if name == "SVM":
        estimator = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                cache_size=1024,
                probability=False,
                decision_function_shape="ovr",
                random_state=seed,
            ),
        )
        grid = {
            "svc__C": [1.0, 10.0, 100.0],
            "svc__gamma": [0.3 / n_features, 1.0 / n_features, 3.0 / n_features],
        }
        return estimator, grid
    if name == "RF":
        estimator = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        return estimator, {"min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]}
    if name == "HGB":
        estimator = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        )
        return estimator, {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}
    if name == "LDA":
        return make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ), {}
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(weights="distance", n_jobs=1),
        ), {"kneighborsclassifier__n_neighbors": [5, 15, 31]}
    if name == "SAM":
        return SpectralAngleMapper(spectral_cols), {}
    if name == "SID":
        return SpectralInformationDivergence(spectral_cols), {}
    if name == "MIN_DIST":
        return MinimumDistanceClassifier(), {}
    if name == "MAHALANOBIS":
        return PooledMahalanobisClassifier(), {}
    if name == "QDA":
        # reg_param regularizes class covariance and is compatible with older sklearn versions.
        return QuadraticDiscriminantAnalysis(), {"reg_param": [0.01, 0.1, 0.3]}
    raise ValueError(f"Unknown model {model_name!r}.")


def groups_per_class(y: np.ndarray, groups: np.ndarray) -> dict[int, int]:
    return {int(c): int(np.unique(groups[y == c]).size) for c in np.unique(y)}


def choose_group_folds(y: np.ndarray, groups: np.ndarray, desired: int, stage: str) -> int:
    counts = groups_per_class(y, groups)
    max_safe = min(counts.values()) if counts else 0
    n = min(int(desired), int(np.unique(groups).size), max_safe)
    if n < 2:
        raise RuntimeError(
            f"Grouped {stage} cross-validation is impossible: at least one class occurs in fewer than 2 "
            f"spatial groups. Groups per class: {counts}. Reduce --group-block-size or add spatially "
            f"independent training regions."
        )
    if n < desired:
        warnings.warn(f"Reducing {stage} folds from {desired} to {n}; groups per class: {counts}.")
    return n


def balanced_training_cap(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    if cap <= 0 or len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    y_sub = y[indices]
    groups_sub = groups[indices]
    classes = np.unique(y_sub)
    per_class = max(1, cap // len(classes))
    chosen: list[np.ndarray] = []
    for c in classes:
        pos = np.flatnonzero(y_sub == c)
        class_idx = indices[pos]
        class_groups = groups_sub[pos]
        if len(class_idx) <= per_class:
            chosen.append(class_idx)
            continue
        unique_groups = np.unique(class_groups)
        quota = max(1, math.ceil(per_class / len(unique_groups)))
        parts: list[np.ndarray] = []
        for g in rng.permutation(unique_groups):
            gidx = class_idx[class_groups == g]
            take = min(quota, len(gidx))
            parts.append(rng.choice(gidx, take, replace=False))
        merged = np.concatenate(parts)
        if len(merged) > per_class:
            merged = rng.choice(merged, per_class, replace=False)
        chosen.append(merged)
    out = np.concatenate(chosen)
    if len(out) > cap:
        out = rng.choice(out, cap, replace=False)
    return out


def model_training_cap(model_name: str, svm_cap: int, knn_cap: int) -> int:
    if model_name.upper() == "SVM":
        return svm_cap
    if model_name.upper() == "KNN":
        return knn_cap
    return 0


def buffer_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    coords: np.ndarray | None,
    buffer_px: float,
) -> tuple[np.ndarray, float, int]:
    """Drop training pixels closer than ``buffer_px`` to any test pixel.

    Grouping keeps whole polygons/blocks apart, but neighbouring groups are still
    spatially autocorrelated. Excluding a buffer around the test pixels removes
    that residual leakage. Returns ``(kept_train_idx, min_train_test_distance_px,
    n_dropped)``; the minimum distance is a leakage diagnostic reported even when
    no buffer is applied.
    """
    if coords is None or len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx, float("nan"), 0
    dist, _ = cKDTree(coords[test_idx]).query(coords[train_idx], k=1)
    min_dist = float(dist.min())
    if buffer_px <= 0:
        return train_idx, min_dist, 0
    keep = dist > buffer_px
    return train_idx[keep], min_dist, int((~keep).sum())


def fit_with_grouped_tuning(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    n_features: int,
    spectral_cols: int,
    inner_folds: int,
    n_jobs: int,
    seed: int,
    svm_cap: int,
    knn_cap: int,
    coords: np.ndarray | None = None,
    buffer_px: float = 0.0,
):
    estimator, grid = build_model(model_name, n_features, spectral_cols, seed)
    cap = model_training_cap(model_name, svm_cap, knn_cap)
    fit_idx = balanced_training_cap(train_idx, y, groups, cap, seed) if cap else train_idx

    if not grid:
        estimator.fit(X[fit_idx], y[fit_idx])
        return estimator, {}, len(fit_idx)

    try:
        n_inner = choose_group_folds(y[fit_idx], groups[fit_idx], inner_folds, "inner")
    except RuntimeError:
        warnings.warn(
            f"Skipping inner hyperparameter tuning for {model_name}: insufficient independent groups. "
            f"Using the first parameter-grid values."
        )
        params = {k: v[0] for k, v in grid.items()}
        estimator.set_params(**params)
        estimator.fit(X[fit_idx], y[fit_idx])
        return estimator, params, len(fit_idx)

    splitter = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 17)
    X_fit, y_fit, g_fit = X[fit_idx], y[fit_idx], groups[fit_idx]
    if coords is not None and buffer_px > 0:
        # Apply the same spatial buffer inside the tuning loop, otherwise tuning
        # still rewards models that exploit neighbour leakage.
        c_fit = coords[fit_idx]
        cv: Any = []
        for tr, te in splitter.split(X_fit, y_fit, g_fit):
            tr_buf, _, _ = buffer_train_indices(tr, te, c_fit, buffer_px)
            if np.unique(y_fit[tr_buf]).size < 2:  # buffer removed (almost) everything
                tr_buf = tr
            cv.append((tr_buf, te))
        fit_kwargs: dict[str, Any] = {}
    else:
        cv = splitter
        fit_kwargs = {"groups": g_fit}
    search = GridSearchCV(
        estimator,
        grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X_fit, y_fit, **fit_kwargs)
    return search.best_estimator_, search.best_params_, len(fit_idx)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_F1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_id": labels,
            "class_name": list(class_names),
            "precision_UA": precision,
            "recall_PA": recall,
            "F1": f1,
            "support": support,
        }
    )


def mean_metrics(y_true: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Metrics averaged over repeats; ``preds`` has shape (n_repeats, n_samples)."""
    preds = np.atleast_2d(preds)
    per_repeat = [metric_summary(y_true, p, labels) for p in preds]
    return {k: float(np.mean([m[k] for m in per_repeat])) for k in per_repeat[0]}


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if n_boot <= 0:
        return {}
    preds = np.atleast_2d(y_pred)  # (n_repeats, n_samples); 1-D input means one repeat
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    records: dict[str, list[float]] = {
        "OA": [],
        "balanced_accuracy": [],
        "macro_F1": [],
        "weighted_F1": [],
        "kappa": [],
        "MCC": [],
    }
    # A resample can miss a class entirely; scikit-learn then warns on every
    # replicate. That is expected here, so the warnings are silenced locally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([idx_by_group[g] for g in sampled_groups])
            m = mean_metrics(y_true[idx], preds[:, idx], labels)
            for key in records:
                records[key].append(m[key])
    return {
        key: (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))
        for key, vals in records.items()
    }


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_names: Sequence[str],
    feature_context: FeatureContext,
    model_name: str,
    outer_folds: int,
    inner_folds: int,
    bootstrap_reps: int,
    n_jobs: int,
    seed: int,
    svm_cap: int,
    knn_cap: int,
    coords: np.ndarray | None = None,
    buffer_px: float = 0.0,
    repeats: int = 1,
) -> ValidationResult:
    """Spatially grouped nested cross-validation.

    ``coords`` (n, 2) pixel coordinates enable the leakage diagnostic
    (``min_train_test_dist_px``) and, with ``buffer_px > 0``, a spatial buffer:
    training pixels closer than ``buffer_px`` to a test pixel are excluded, in
    both the outer folds and the inner tuning folds. ``repeats > 1`` repeats the
    whole outer split with different random fold assignments; headline metrics
    are the mean over repeats and the bootstrap resamples groups jointly for
    all repeats.
    """
    labels = np.arange(len(class_names), dtype=np.int32)
    n_outer = choose_group_folds(y, groups, outer_folds, "outer")
    repeats = max(1, int(repeats))

    preds = np.full((repeats, len(y)), -1, dtype=y.dtype)
    fold_records: list[dict[str, Any]] = []
    params_records: list[dict[str, Any]] = []

    for rep in range(repeats):
        outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed + 1009 * rep)
        for fold, (train_all, test_idx) in enumerate(outer.split(X, y, groups), start=1):
            t0 = time.time()
            train_idx, min_dist, n_dropped = buffer_train_indices(train_all, test_idx, coords, buffer_px)
            if n_dropped:
                frac = n_dropped / len(train_all)
                if np.unique(y[train_idx]).size < 2 or len(train_idx) < 2 * len(labels):
                    raise RuntimeError(
                        f"buffer_px={buffer_px:g} removes {n_dropped} of {len(train_all)} training pixels in outer "
                        f"fold {fold}, leaving too few to fit a model. Use a smaller buffer (it should be about the "
                        f"spatial autocorrelation range, well below the size of your groups) or more training regions."
                    )
                if frac > 0.5:
                    warnings.warn(
                        f"{model_name}: spatial buffer removed {100 * frac:.0f}% of the training pixels in outer "
                        f"fold {fold}; results may be pessimistic and unstable."
                    )
            absent = sorted(set(labels.tolist()) - set(np.unique(y[train_idx]).tolist()))
            if absent:
                warnings.warn(
                    f"{model_name}: outer fold {fold} (repeat {rep + 1}) has no training pixels for class "
                    f"id(s) {absent}; those classes cannot be predicted in this fold."
                    + (" The spatial buffer may be too large." if n_dropped else "")
                )
            model, best_params, n_fit = fit_with_grouped_tuning(
                model_name=model_name,
                X=X,
                y=y,
                groups=groups,
                train_idx=train_idx,
                n_features=X.shape[1],
                spectral_cols=feature_context.base_feature_count,
                inner_folds=inner_folds,
                n_jobs=n_jobs,
                seed=seed + fold * 101 + rep * 100003,
                svm_cap=svm_cap,
                knn_cap=knn_cap,
                coords=coords,
                buffer_px=buffer_px,
            )
            pred = np.asarray(model.predict(X[test_idx]), dtype=y.dtype)
            preds[rep, test_idx] = pred
            fm = metric_summary(y[test_idx], pred, labels)
            fold_records.append(
                {
                    "repeat": rep + 1,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_train_before_buffer": len(train_all),
                    "n_train_dropped_by_buffer": n_dropped,
                    "min_train_test_dist_px": min_dist,
                    "n_fit_after_cap": n_fit,
                    "n_test": len(test_idx),
                    "n_train_groups": int(np.unique(groups[train_idx]).size),
                    "n_test_groups": int(np.unique(groups[test_idx]).size),
                    "seconds": time.time() - t0,
                    **fm,
                }
            )
            params_records.append(best_params)
            tag = f"repeat {rep + 1}/{repeats}, " if repeats > 1 else ""
            log(
                f"    {tag}fold {fold}/{n_outer}: macro-F1={fm['macro_F1']:.3f}, "
                f"OA={fm['OA']:.3f}, {time.time() - t0:.1f}s"
            )

    if np.any(preds < 0):
        raise RuntimeError("Internal error: some samples received no out-of-fold prediction.")

    summary = mean_metrics(y, preds, labels)
    if repeats > 1:
        per_repeat = [metric_summary(y, preds[r], labels) for r in range(repeats)]
        for key in summary.copy():
            summary[f"{key}_repeat_std"] = float(np.std([m[key] for m in per_repeat], ddof=1))
    ci = bootstrap_metric_ci(y, preds, groups, labels, bootstrap_reps, seed + 9001)
    for key, (lo, hi) in ci.items():
        summary[f"{key}_CI95_low"] = lo
        summary[f"{key}_CI95_high"] = hi

    fold_df = pd.DataFrame(fold_records)
    for key in ("OA", "balanced_accuracy", "macro_F1", "weighted_F1", "kappa", "MCC"):
        summary[f"{key}_fold_mean"] = float(fold_df[key].mean())
        summary[f"{key}_fold_std"] = float(fold_df[key].std(ddof=1)) if len(fold_df) > 1 else 0.0
    summary["min_train_test_dist_px"] = float(fold_df["min_train_test_dist_px"].min())
    summary["n_repeats"] = repeats
    summary["min_groups_per_class"] = min(groups_per_class(y, groups).values())

    tables = [per_class_table(y, preds[r], class_names) for r in range(repeats)]
    per_class = tables[0].copy()
    for col in ("precision_UA", "recall_PA", "F1", "support"):
        per_class[col] = np.mean([t[col].to_numpy() for t in tables], axis=0)
    cms = [confusion_matrix(y, preds[r], labels=labels) for r in range(repeats)]
    cm = cms[0] if repeats == 1 else np.mean(cms, axis=0)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    return ValidationResult(
        feature_set=feature_context.spec.name,
        model=model_name,
        y_true=y.copy(),
        y_pred=preds[0].copy(),
        groups=groups.copy(),
        fold_metrics=fold_records,
        best_params_per_fold=params_records,
        summary=summary,
        per_class=per_class,
        confusion=cm,
        confusion_norm=cm_norm,
        y_pred_repeats=preds,
        n_repeats=repeats,
    )


# -----------------------------------------------------------------------------
# Band diagnostics and spectral statistics
# -----------------------------------------------------------------------------


def fisher_scores(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    overall = np.mean(X, axis=0, dtype=np.float64)
    between = np.zeros(X.shape[1], dtype=np.float64)
    within = np.zeros(X.shape[1], dtype=np.float64)
    for c in range(n_classes):
        xc = X[y == c].astype(np.float64)
        if len(xc) == 0:
            continue
        mean_c = np.mean(xc, axis=0)
        var_c = np.var(xc, axis=0)
        between += len(xc) * (mean_c - overall) ** 2
        within += len(xc) * var_c
    return between / np.maximum(within, 1e-12)


def sample_global_raster_pixels(
    ds: rasterio.io.DatasetReader,
    max_samples: int,
    seed: int,
    window_size: int = 128,
) -> np.ndarray:
    if max_samples <= 0:
        return np.empty((0, ds.count), dtype=np.float32)
    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    n = 0
    max_windows = max(20, int(math.ceil(max_samples / (window_size * window_size))) * 8)

    for _ in range(max_windows):
        if n >= max_samples:
            break
        h = min(window_size, ds.height)
        w = min(window_size, ds.width)
        r0 = int(rng.integers(0, max(1, ds.height - h + 1)))
        c0 = int(rng.integers(0, max(1, ds.width - w + 1)))
        window = Window(c0, r0, w, h)
        data = ds.read(window=window).astype(np.float32, copy=False)
        mask = ds.read_masks(1, window=window) > 0
        flat = np.moveaxis(data, 0, -1).reshape(-1, ds.count)
        valid = mask.ravel() & np.isfinite(flat).all(axis=1)
        vals = flat[valid]
        if len(vals) == 0:
            continue
        remaining = max_samples - n
        if len(vals) > remaining:
            vals = vals[rng.choice(len(vals), remaining, replace=False)]
        collected.append(vals)
        n += len(vals)

    if not collected:
        return np.empty((0, ds.count), dtype=np.float32)
    return np.concatenate(collected, axis=0)[:max_samples]


def shannon_entropy_approx(x: np.ndarray, bins: int = 64) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.nanmin(x) == np.nanmax(x):
        return 0.0
    hist, _ = np.histogram(x, bins=bins)
    p = hist[hist > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def band_diagnostics(
    global_sample: np.ndarray,
    training_spectra: np.ndarray,
    y: np.ndarray,
    band_names: Sequence[str],
    wavelengths_nm: np.ndarray | None,
    dtype_name: str,
) -> pd.DataFrame:
    if len(global_sample) == 0:
        global_sample = training_spectra

    fisher = fisher_scores(training_spectra, y, int(np.max(y)) + 1)
    dtype = np.dtype(dtype_name)
    integer_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else None
    rows: list[dict[str, Any]] = []

    for b in range(training_spectra.shape[1]):
        x = global_sample[:, b].astype(np.float64)
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            stats = dict(mean=np.nan, std=np.nan, min=np.nan, p01=np.nan, p99=np.nan, max=np.nan)
            zero_pct = sat_pct = entropy = np.nan
        else:
            stats = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "p01": float(np.percentile(finite, 1)),
                "p99": float(np.percentile(finite, 99)),
                "max": float(np.max(finite)),
            }
            zero_pct = float(np.mean(finite == 0) * 100.0)
            sat_pct = float(np.mean(finite == integer_max) * 100.0) if integer_max is not None else np.nan
            entropy = shannon_entropy_approx(finite)

        dynamic_range = stats["p99"] - stats["p01"] if np.isfinite(stats["p99"]) else np.nan
        if not np.isfinite(stats["std"]) or stats["std"] <= 1e-12 or dynamic_range <= 1e-12:
            recommendation = "DROP_CONSTANT"
        elif zero_pct >= 99.5 and (not np.isfinite(dynamic_range) or dynamic_range <= 1.0):
            recommendation = "REVIEW_NEAR_BLANK"
        elif np.isfinite(sat_pct) and sat_pct >= 50.0:
            recommendation = "REVIEW_SATURATED"
        else:
            recommendation = "KEEP_OR_VALIDATE"

        rows.append(
            {
                "band": b + 1,
                "band_name": band_names[b],
                "wavelength_nm": float(wavelengths_nm[b]) if wavelengths_nm is not None else np.nan,
                **stats,
                "robust_range_p99_p01": dynamic_range,
                "zero_pct": zero_pct,
                "saturated_pct": sat_pct,
                "entropy_bits_approx": entropy,
                "fisher_class_separability": float(fisher[b]),
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def class_spectral_statistics(
    training_spectra: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    band_names: Sequence[str],
    wavelengths_nm: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, name in enumerate(class_names):
        Xc = training_spectra[y == cid].astype(np.float64)
        for b in range(training_spectra.shape[1]):
            rows.append(
                {
                    "class_id": cid,
                    "class_name": name,
                    "band": b + 1,
                    "band_name": band_names[b],
                    "wavelength_nm": float(wavelengths_nm[b]) if wavelengths_nm is not None else np.nan,
                    "n": len(Xc),
                    "mean": float(np.mean(Xc[:, b])) if len(Xc) else np.nan,
                    "std": float(np.std(Xc[:, b], ddof=1)) if len(Xc) > 1 else 0.0,
                    "median": float(np.median(Xc[:, b])) if len(Xc) else np.nan,
                    "p05": float(np.percentile(Xc[:, b], 5)) if len(Xc) else np.nan,
                    "p95": float(np.percentile(Xc[:, b], 95)) if len(Xc) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def class_mean_spectral_separability(
    training_spectra: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    means = np.vstack([training_spectra[y == c].mean(axis=0) for c in range(len(class_names))]).astype(float)
    pooled_std = np.std(training_spectra.astype(float), axis=0)
    pooled_std = np.where(pooled_std > 1e-12, pooled_std, 1.0)
    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(class_names)), 2):
        a, b = means[i], means[j]
        cosv = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-15))
        angle = float(np.arccos(np.clip(cosv, -1.0, 1.0)))
        std_euclid = float(np.linalg.norm((a - b) / pooled_std))
        rows.append(
            {
                "class_a": class_names[i],
                "class_b": class_names[j],
                "mean_spectral_angle_rad": angle,
                "mean_spectral_angle_deg": math.degrees(angle),
                "standardized_mean_distance": std_euclid,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Reporting and figures
# -----------------------------------------------------------------------------


def save_confusion_figure(
    cm_norm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(class_names)), max(5, 0.68 * len(class_names))))
    im = ax.imshow(cm_norm, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Reference class")
    ax.set_title(title)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{100 * cm_norm[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_spectral_signatures_figure(
    stats_df: pd.DataFrame,
    class_names: Sequence[str],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    use_wavelength = stats_df["wavelength_nm"].notna().all()
    for name in class_names:
        sub = stats_df[stats_df["class_name"] == name].sort_values("band")
        x = sub["wavelength_nm"].to_numpy() if use_wavelength else sub["band"].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].to_numpy()
        line = ax.plot(x, mean, label=name)[0]
        ax.fill_between(x, mean - std, mean + std, alpha=0.14, color=line.get_color())
    ax.set_xlabel("Wavelength (nm)" if use_wavelength else "Band")
    ax.set_ylabel("Mean spectral value")
    ax.set_title("Class spectral signatures (mean ± 1 SD)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_band_diagnostics_figure(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = df["wavelength_nm"].to_numpy()
    if np.isnan(x).any():
        x = df["band"].to_numpy()
        xlabel = "Band"
    else:
        xlabel = "Wavelength (nm)"
    ax.plot(x, df["fisher_class_separability"].to_numpy(), marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fisher class-separability score")
    ax.set_title("Per-band class separability (training samples)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_figures(comparison: pd.DataFrame, figures_dir: Path) -> None:
    if comparison.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.feature_set}\n{r.model}" for r in comparison.itertuples()]

    for metric, title in (("macro_F1", "Macro-F1"), ("OA", "Overall Accuracy")):
        order = np.argsort(comparison[metric].to_numpy())[::-1]
        comp = comparison.iloc[order].reset_index(drop=True)
        plot_labels = [f"{r.feature_set}\n{r.model}" for r in comp.itertuples()]
        values = comp[metric].to_numpy()
        low_col = f"{metric}_CI95_low"
        high_col = f"{metric}_CI95_high"
        yerr = None
        if low_col in comp and high_col in comp:
            low = comp[low_col].to_numpy()
            high = comp[high_col].to_numpy()
            # A percentile interval need not contain the point estimate (few replicates), and matplotlib
            # rejects negative error bars, so clip at zero instead of crashing the whole run.
            yerr = np.clip(np.vstack([values - low, high - values]), 0.0, None)
        fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(comp)), 5.5))
        ax.bar(np.arange(len(comp)), values, yerr=yerr, capsize=3)
        ax.set_xticks(np.arange(len(comp)), plot_labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        ax.set_title(f"Model comparison: {title}")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"model_comparison_{sanitize_name(metric)}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_per_class_f1_heatmap(per_class_all: pd.DataFrame, figures_dir: Path) -> None:
    if per_class_all.empty:
        return
    pivot = per_class_all.pivot_table(
        index=["feature_set", "model"], columns="class_name", values="F1", aggfunc="first"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(
        figsize=(max(8, 0.9 * len(pivot.columns)), max(5, 0.48 * len(pivot.index) + 2))
    )
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    row_labels = [f"{a} | {b}" for a, b in pivot.index]
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Per-class F1 across model / feature-set combinations")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "per_class_F1_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment (controls family-wise error)."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if valid.size == 0:
        return out
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = valid.size
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def bootstrap_p_value(boot_deltas: Sequence[float]) -> float:
    """Two-sided p-value for H0: delta = 0 from a bootstrap distribution of the delta.

    Uses the share of replicates on the opposite side of zero (with a +1
    correction so p is never exactly 0). It is approximate and its resolution is
    limited by the number of replicates (about 2/(B+1)).
    """
    b = np.asarray(boot_deltas, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan")
    tail = min(int(np.sum(b <= 0)), int(np.sum(b >= 0)))
    return float(min(1.0, 2.0 * (tail + 1) / (b.size + 1)))


def pairwise_bootstrap_comparisons(
    results: Sequence[ValidationResult],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Paired grouped-bootstrap differences between models within each feature set.

    Both models are scored on the *same* resampled groups, so the interval
    reflects the paired difference. The Holm-adjusted p-values account for the
    number of model pairs compared within a feature set; unadjusted intervals
    across many pairs will show some spurious "differences" by chance.
    """
    rows: list[dict[str, Any]] = []
    by_feature: dict[str, list[ValidationResult]] = {}
    for r in results:
        by_feature.setdefault(r.feature_set, []).append(r)

    def repeats_of(res: ValidationResult) -> np.ndarray:
        return res.y_pred_repeats if res.y_pred_repeats is not None else np.atleast_2d(res.y_pred)

    rng = np.random.default_rng(seed)
    for feature_set, feature_results in by_feature.items():
        feature_rows: list[dict[str, Any]] = []
        for a, b in itertools.combinations(feature_results, 2):
            pa, pb = repeats_of(a), repeats_of(b)
            if not (
                np.array_equal(a.y_true, b.y_true)
                and np.array_equal(a.groups, b.groups)
                and pa.shape == pb.shape
            ):
                continue
            y = a.y_true
            groups = a.groups
            labels = np.unique(y)
            unique_groups = np.unique(groups)
            idx_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
            actual_a = mean_metrics(y, pa, labels)
            actual_b = mean_metrics(y, pb, labels)

            boot_oa: list[float] = []
            boot_f1: list[float] = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(max(0, n_boot)):
                    sampled = rng.choice(unique_groups, len(unique_groups), replace=True)
                    idx = np.concatenate([idx_by_group[g] for g in sampled])
                    ma = mean_metrics(y[idx], pa[:, idx], labels)
                    mb = mean_metrics(y[idx], pb[:, idx], labels)
                    boot_oa.append(ma["OA"] - mb["OA"])
                    boot_f1.append(ma["macro_F1"] - mb["macro_F1"])

            def ci(vals: list[float]) -> tuple[float, float]:
                if not vals:
                    return np.nan, np.nan
                return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

            oa_lo, oa_hi = ci(boot_oa)
            f1_lo, f1_hi = ci(boot_f1)
            feature_rows.append(
                {
                    "feature_set": feature_set,
                    "model_A": a.model,
                    "model_B": b.model,
                    "delta_OA_A_minus_B": actual_a["OA"] - actual_b["OA"],
                    "delta_OA_CI95_low": oa_lo,
                    "delta_OA_CI95_high": oa_hi,
                    "delta_OA_p_boot": bootstrap_p_value(boot_oa),
                    "delta_macro_F1_A_minus_B": actual_a["macro_F1"] - actual_b["macro_F1"],
                    "delta_macro_F1_CI95_low": f1_lo,
                    "delta_macro_F1_CI95_high": f1_hi,
                    "delta_macro_F1_p_boot": bootstrap_p_value(boot_f1),
                }
            )
        for key in ("delta_OA", "delta_macro_F1"):
            adjusted = holm_adjust([r[f"{key}_p_boot"] for r in feature_rows])
            for r, p_adj in zip(feature_rows, adjusted):
                r[f"{key}_p_holm"] = float(p_adj)
        rows.extend(feature_rows)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Area weighting, independent hold-out, and rejection
# -----------------------------------------------------------------------------


def load_class_prevalence(path: Path, class_names: Sequence[str]) -> np.ndarray:
    """Read ``class,area`` (or ``class,proportion``) and align it with ``class_names``.

    Values only need to be proportional; they describe how much of the *scene* each
    reference class covers, which the class-capped training sample cannot tell.
    """
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    value_col = next((lower[k] for k in ("area", "proportion", "weight", "fraction") if k in lower), None)
    if "class" not in lower or value_col is None:
        raise ValueError("Class-prevalence CSV needs columns: class and one of area/proportion/weight/fraction.")
    table = {str(k).casefold(): float(v) for k, v in zip(df[lower["class"]], df[value_col])}
    missing = [n for n in class_names if n.casefold() not in table]
    if missing:
        raise ValueError(f"Class-prevalence CSV has no value for classes: {missing}")
    values = np.asarray([table[n.casefold()] for n in class_names], dtype=float)
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Class-prevalence values must be non-negative with a positive sum.")
    return values / values.sum()


def area_weighted_metrics(
    confusion: np.ndarray,
    prevalence: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Re-weight a confusion matrix (rows = reference) to scene class proportions.

    The training sample is capped per reference class, so it over-represents rare classes.
    For a sample stratified by reference class, the estimated area proportion in cell (i, j)
    is ``W_i * n_ij / n_i`` (W_i = share of the scene that is class i). Overall accuracy,
    user's/producer's accuracy, F1 and kappa follow from those proportions.
    """
    cm = np.asarray(confusion, dtype=float)
    w = np.asarray(prevalence, dtype=float)
    n_i = cm.sum(axis=1, keepdims=True)
    p = np.divide(w[:, None] * cm, n_i, out=np.zeros_like(cm), where=n_i > 0)
    diag = np.diag(p)
    row, col = p.sum(axis=1), p.sum(axis=0)
    pa = np.divide(diag, row, out=np.zeros_like(diag), where=row > 0)
    ua = np.divide(diag, col, out=np.zeros_like(diag), where=col > 0)
    denom = ua + pa
    f1 = np.divide(2 * ua * pa, denom, out=np.zeros_like(diag), where=denom > 0)
    oa = float(diag.sum())
    chance = float(np.sum(row * col))
    kappa = (oa - chance) / (1.0 - chance) if chance < 1.0 else float("nan")
    summary = {
        "OA_area_weighted": oa,
        "macro_F1_area_weighted": float(f1.mean()),
        "kappa_area_weighted": float(kappa),
    }
    per_class = pd.DataFrame(
        {
            "class_name": list(class_names),
            "scene_proportion": w,
            "precision_UA_area_weighted": ua,
            "recall_PA_area_weighted": pa,
            "F1_area_weighted": f1,
        }
    )
    return summary, per_class


def split_holdout_groups(
    y: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve whole spatial groups, stratified by class, as an independent test set.

    Returns ``(dev_idx, holdout_idx)``. Every class must keep at least two groups for
    development (so grouped CV still works) and have at least one hold-out group.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5 (exclusive).")
    n_splits = max(2, int(round(1.0 / fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 555)
    dev_idx, hold_idx = next(iter(splitter.split(np.zeros(len(y)), y, groups)))
    problems = []
    for c in np.unique(y):
        n_dev = np.unique(groups[dev_idx][y[dev_idx] == c]).size
        n_hold = np.unique(groups[hold_idx][y[hold_idx] == c]).size
        if n_dev < 2 or n_hold < 1:
            problems.append(f"class id {int(c)}: {n_dev} development / {n_hold} hold-out groups")
    if problems:
        raise RuntimeError(
            "holdout_fraction leaves a class without enough independent groups ("
            + "; ".join(problems)
            + "). Use a smaller fraction, more training regions, or disable the hold-out."
        )
    return dev_idx, hold_idx


def evaluate_holdout(
    model_name: str,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: np.ndarray,
    coords_dev: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    groups_hold: np.ndarray,
    coords_hold: np.ndarray,
    class_names: Sequence[str],
    feature_context: FeatureContext,
    inner_folds: int,
    n_jobs: int,
    seed: int,
    svm_cap: int,
    knn_cap: int,
    buffer_px: float,
    bootstrap_reps: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Fit on all development pixels and score once on the reserved hold-out groups.

    Development pixels within ``buffer_px`` of a hold-out pixel are excluded from the fit.
    """
    labels = np.arange(len(class_names), dtype=np.int32)
    n_dev = len(y_dev)
    all_coords = np.vstack([coords_dev, coords_hold])
    train_idx, min_dist, n_dropped = buffer_train_indices(
        np.arange(n_dev), n_dev + np.arange(len(y_hold)), all_coords, buffer_px
    )
    model, params, n_fit = fit_with_grouped_tuning(
        model_name=model_name,
        X=X_dev,
        y=y_dev,
        groups=groups_dev,
        train_idx=train_idx,
        n_features=X_dev.shape[1],
        spectral_cols=feature_context.base_feature_count,
        inner_folds=inner_folds,
        n_jobs=n_jobs,
        seed=seed + 7777,
        svm_cap=svm_cap,
        knn_cap=knn_cap,
        coords=coords_dev,
        buffer_px=buffer_px,
    )
    pred = np.asarray(model.predict(X_hold), dtype=y_hold.dtype)
    record: dict[str, Any] = {
        "model": model_name,
        "feature_set": feature_context.spec.name,
        "n_holdout": int(len(y_hold)),
        "n_holdout_groups": int(np.unique(groups_hold).size),
        "n_fit": int(n_fit),
        "min_train_holdout_dist_px": min_dist,
        "n_train_dropped_by_buffer": n_dropped,
        "best_params": json.dumps(params, default=str),
        **metric_summary(y_hold, pred, labels),
    }
    for key, (lo, hi) in bootstrap_metric_ci(y_hold, pred, groups_hold, labels, bootstrap_reps, seed + 9999).items():
        record[f"{key}_CI95_low"] = lo
        record[f"{key}_CI95_high"] = hi
    cm = confusion_matrix(y_hold, pred, labels=labels)
    return record, per_class_table(y_hold, pred, class_names), cm


def apply_rejection(
    codes: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float,
    reject_code: int,
) -> np.ndarray:
    """Replace class codes by ``reject_code`` where confidence is below the threshold."""
    if min_confidence <= 0:
        return codes
    return np.where(confidence < min_confidence, reject_code, codes)


# -----------------------------------------------------------------------------
# Final fit, confidence, and tile-based mapping
# -----------------------------------------------------------------------------


def fit_final_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_context: FeatureContext,
    inner_folds: int,
    n_jobs: int,
    seed: int,
    svm_cap: int,
    knn_cap: int,
    coords: np.ndarray | None = None,
    buffer_px: float = 0.0,
):
    idx = np.arange(len(y))
    model, params, n_fit = fit_with_grouped_tuning(
        model_name=model_name,
        X=X,
        y=y,
        groups=groups,
        train_idx=idx,
        n_features=X.shape[1],
        spectral_cols=feature_context.base_feature_count,
        inner_folds=inner_folds,
        n_jobs=n_jobs,
        seed=seed + 4444,
        svm_cap=svm_cap,
        knn_cap=knn_cap,
        coords=coords,
        buffer_px=buffer_px,
    )
    return model, params, n_fit


def confidence_from_model(model: Any, X: np.ndarray) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=np.float64)
            return np.max(p, axis=1).astype(np.float32), "max_predict_proba_or_distance_softmax"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        p = softmax(scores, axis=1)
        return np.max(p, axis=1).astype(np.float32), "softmax_of_decision_function_uncalibrated"
    return np.full(len(X), np.nan, dtype=np.float32), "unavailable"


def iter_windows(height: int, width: int, tile_size: int) -> Iterable[Window]:
    for r0 in range(0, height, tile_size):
        h = min(tile_size, height - r0)
        for c0 in range(0, width, tile_size):
            w = min(tile_size, width - c0)
            yield Window(c0, r0, w, h)


def map_full_raster(
    image_path: Path,
    output_dir: Path,
    model: Any,
    model_name: str,
    feature_context: FeatureContext,
    selected_band_indices: np.ndarray,
    band_names: Sequence[str],
    wavelengths_nm: np.ndarray | None,
    class_names: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
    tile_size: int,
    write_confidence: bool,
    index_tolerance_nm: float,
    min_confidence: float = 0.0,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"classification_{sanitize_name(feature_context.spec.name)}_{sanitize_name(model_name)}"
    class_path = output_dir / f"{stem}.tif"
    conf_path = output_dir / f"confidence_{sanitize_name(feature_context.spec.name)}_{sanitize_name(model_name)}.tif"

    with rasterio.open(image_path) as src:
        # One extra code (len+1) is reserved for "Unclassified" when a rejection threshold is used.
        dtype = "uint8" if len(class_names) < 254 else "uint16"
        reject_code = len(class_names) + 1
        need_confidence = write_confidence or min_confidence > 0
        profile = src.profile.copy()
        # GeoTIFF internal tiles must be a multiple of 16 pixels.
        block = max(16, (min(512, tile_size) // 16) * 16)
        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=0,
            tiled=True,
            blockxsize=block,
            blockysize=block,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        profile.pop("photometric", None)
        profile.pop("interleave", None)

        conf_profile = profile.copy()
        conf_profile.update(dtype="float32", nodata=np.nan)

        total = math.ceil(src.height / tile_size) * math.ceil(src.width / tile_size)
        conf_type = "unavailable"
        with rasterio.open(class_path, "w", **profile) as dst_class:
            dst_conf_ctx = rasterio.open(conf_path, "w", **conf_profile) if write_confidence else None
            try:
                for tile_no, window in enumerate(iter_windows(src.height, src.width, tile_size), start=1):
                    data = src.read(window=window).astype(np.float32, copy=False)
                    mask = src.read_masks(1, window=window) > 0
                    h, w = int(window.height), int(window.width)
                    flat_raw = np.moveaxis(data, 0, -1).reshape(-1, src.count)
                    valid = mask.ravel() & np.isfinite(flat_raw).all(axis=1)

                    out_class = np.zeros(h * w, dtype=dtype)
                    out_conf = np.full(h * w, np.nan, dtype=np.float32)
                    if np.any(valid):
                        X_tile, _ = build_features(
                            flat_raw[valid],
                            selected_band_indices,
                            band_names,
                            wavelengths_nm,
                            feature_context.spec,
                            index_tolerance_nm,
                        )
                        pred = np.asarray(model.predict(X_tile), dtype=int)
                        codes = pred + 1
                        if need_confidence:
                            conf, conf_type = confidence_from_model(model, X_tile)
                            if write_confidence:
                                out_conf[valid] = conf
                            codes = apply_rejection(codes, conf, min_confidence, reject_code)
                        out_class[valid] = codes.astype(dtype)

                    dst_class.write(out_class.reshape(h, w), 1, window=window)
                    if dst_conf_ctx is not None:
                        dst_conf_ctx.write(out_conf.reshape(h, w), 1, window=window)

                    if tile_no == 1 or tile_no % 25 == 0 or tile_no == total:
                        log(f"      map tile {tile_no}/{total}")

                dst_class.update_tags(
                    CLASS_NAMES=json.dumps(list(class_names)),
                    FEATURE_SET=feature_context.spec.name,
                    MODEL=model_name,
                    SPECTRAL_BENCHMARK_VERSION=VERSION,
                    MIN_CONFIDENCE=str(min_confidence),
                    UNCLASSIFIED_CODE=str(reject_code) if min_confidence > 0 else "none",
                )
                if dst_conf_ctx is not None:
                    dst_conf_ctx.update_tags(
                        CONFIDENCE_TYPE=conf_type,
                        FEATURE_SET=feature_context.spec.name,
                        MODEL=model_name,
                    )
            finally:
                if dst_conf_ctx is not None:
                    dst_conf_ctx.close()

    legend_path = output_dir / "legend.csv"
    if min_confidence > 0 or not legend_path.exists():
        pd.DataFrame(
            [
                {"code": 0, "class_name": "NoData", "r": 0, "g": 0, "b": 0},
                *[
                    {
                        "code": i + 1,
                        "class_name": name,
                        "r": colors[i][0],
                        "g": colors[i][1],
                        "b": colors[i][2],
                    }
                    for i, name in enumerate(class_names)
                ],
                *(
                    [{"code": reject_code, "class_name": "Unclassified", "r": 128, "g": 128, "b": 128}]
                    if min_confidence > 0
                    else []
                ),
            ]
        ).to_csv(legend_path, index=False)

    return class_path, conf_path if write_confidence else None


def save_classification_preview(
    class_path: Path,
    colors: Sequence[tuple[int, int, int]],
    class_names: Sequence[str],
    path: Path,
    max_dim: int = 3000,
) -> None:
    with rasterio.open(class_path) as ds:
        scale = max(ds.width / max_dim, ds.height / max_dim, 1.0)
        out_w = max(1, int(round(ds.width / scale)))
        out_h = max(1, int(round(ds.height / scale)))
        arr = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    lut = np.zeros((len(class_names) + 2, 3), dtype=np.uint8)
    lut[len(class_names) + 1] = (128, 128, 128)  # "Unclassified" when a rejection threshold was used
    for i, rgb in enumerate(colors, start=1):
        if i <= len(class_names):
            lut[i] = rgb
    clipped = np.clip(arr.astype(int), 0, len(lut) - 1)
    rgb = lut[clipped]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title(class_path.stem)
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Configuration loading
# -----------------------------------------------------------------------------


def load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_value(args_value: Any, config: dict[str, Any], key: str, default: Any = None) -> Any:
    return args_value if args_value is not None else config.get(key, default)


def feature_specs_from_inputs(
    cli_feature_sets: str | None,
    cli_indices: str | None,
    config: dict[str, Any],
) -> list[FeatureSpec]:
    indices = parse_csv_list(cli_indices)
    if not indices:
        indices = [str(x) for x in config.get("indices", ["GLI", "TGI", "RGRI", "VARI"])]

    if isinstance(config.get("feature_sets"), list) and config["feature_sets"] and isinstance(config["feature_sets"][0], dict):
        specs: list[FeatureSpec] = []
        for item in config["feature_sets"]:
            specs.append(
                FeatureSpec(
                    name=str(item["name"]),
                    normalization=str(item.get("normalization", "none")),
                    indices=[str(x) for x in item.get("indices", [])],
                    derivative=bool(item.get("derivative", False)),
                )
            )
        if cli_feature_sets is None:
            return specs

    names = parse_csv_list(cli_feature_sets) if cli_feature_sets is not None else [str(x) for x in config.get("feature_sets", ["raw"])]
    if not names:
        names = ["raw"]
    return [feature_preset(x, indices) for x in names]


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    config = load_config_file(Path(args.config) if args.config else None)
    unknown_keys = sorted(set(config) - KNOWN_CONFIG_KEYS)
    if unknown_keys:
        warnings.warn(f"Ignoring unknown config key(s): {unknown_keys}. Check spelling against the README.")

    image = Path(config_value(args.image, config, "image"))
    output_dir = Path(config_value(args.output_dir, config, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    maps_dir = output_dir / "maps"
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    seed = int(config_value(args.seed, config, "seed", 0))
    outer_folds = int(config_value(args.outer_folds, config, "outer_folds", 5))
    inner_folds = int(config_value(args.inner_folds, config, "inner_folds", 3))
    bootstrap_reps = int(config_value(args.bootstrap, config, "bootstrap", 300))
    pairwise_bootstrap_reps = int(config_value(args.pairwise_bootstrap, config, "pairwise_bootstrap", 300))
    max_samples_per_class = int(config_value(args.max_samples_per_class, config, "max_samples_per_class", 3000))
    group_block_size = int(config_value(args.group_block_size, config, "group_block_size_px", 2048))
    n_jobs = int(config_value(args.n_jobs, config, "n_jobs", max(1, (os.cpu_count() or 4) - 2)))
    svm_cap = int(config_value(args.svm_train_cap, config, "svm_train_cap", 12000))
    knn_cap = int(config_value(args.knn_train_cap, config, "knn_train_cap", 20000))
    tile_size = int(config_value(args.tile_size, config, "tile_size", 512))
    index_tolerance_nm = float(config_value(args.index_tolerance_nm, config, "index_tolerance_nm", 60.0))
    global_band_samples = int(config_value(args.band_analysis_samples, config, "band_analysis_samples", 100000))
    buffer_px = float(config_value(args.buffer_px, config, "buffer_px", 0.0))
    repeats = int(config_value(args.repeats, config, "repeats", 1))
    group_merge_px = float(config_value(args.group_merge_px, config, "group_merge_px", 0.0))
    class_prevalence_path = config_value(args.class_prevalence, config, "class_prevalence")
    holdout_fraction = float(config_value(args.holdout_fraction, config, "holdout_fraction", 0.0))
    min_confidence = float(config_value(args.min_confidence, config, "min_confidence", 0.0))
    if not 0.0 <= min_confidence < 1.0:
        raise ValueError("min_confidence must be in [0, 1).")
    if buffer_px < 0 or group_merge_px < 0 or repeats < 1:
        raise ValueError("buffer_px and group_merge_px must be >= 0 and repeats must be >= 1.")

    exclude_classes = set(parse_csv_list(args.exclude_classes))
    if not exclude_classes:
        exclude_classes = set(str(x) for x in config.get("exclude_classes", []))

    models = [x.upper() for x in parse_csv_list(args.models)] if args.models is not None else [str(x).upper() for x in config.get("models", DEFAULT_MODELS)]
    if not models:
        models = list(DEFAULT_MODELS)
    unknown = [m for m in models if m not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Available: {DEFAULT_MODELS}")

    feature_specs = feature_specs_from_inputs(args.feature_sets, args.indices, config)

    log(f"Spectral Benchmark v{VERSION}")
    log(f"Input image: {image}")
    log(f"Output directory: {output_dir}")

    with rasterio.open(image) as ds:
        wavelengths_nm = infer_wavelengths_nm(ds)
        if args.wavelengths:
            wl = np.asarray(parse_float_list(args.wavelengths), dtype=float)
            if len(wl) != ds.count:
                raise ValueError(f"--wavelengths supplied {len(wl)} values but raster has {ds.count} bands.")
            wavelengths_nm = wl
        elif config.get("wavelengths_nm") is not None:
            wl = np.asarray(config["wavelengths_nm"], dtype=float)
            if len(wl) != ds.count:
                raise ValueError("config wavelengths_nm length does not match raster band count.")
            wavelengths_nm = wl

        band_names = infer_band_names(ds, wavelengths_nm)
        selected_band_indices = resolve_band_selection(
            ds.count,
            wavelengths_nm,
            parse_int_list(args.keep_bands) if args.keep_bands is not None else [int(x) for x in config.get("keep_bands", [])],
            parse_int_list(args.drop_bands) if args.drop_bands is not None else [int(x) for x in config.get("drop_bands", [])],
            parse_float_list(args.keep_wavelengths) if args.keep_wavelengths is not None else [float(x) for x in config.get("keep_wavelengths_nm", [])],
            parse_float_list(args.drop_wavelengths) if args.drop_wavelengths is not None else [float(x) for x in config.get("drop_wavelengths_nm", [])],
            index_tolerance_nm,
        )

        raster_info = {
            "path": str(image),
            "width": ds.width,
            "height": ds.height,
            "bands": ds.count,
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs),
            "transform": tuple(ds.transform),
            "wavelengths_nm": wavelengths_nm.tolist() if wavelengths_nm is not None else None,
            "band_names": band_names,
            "selected_bands_1based": (selected_band_indices + 1).tolist(),
        }

        # Label source selection.
        roi_xml = config_value(args.roi_xml, config, "roi_xml")
        samples_csv = config_value(args.samples_csv, config, "samples_csv")
        label_raster = config_value(args.label_raster, config, "label_raster")
        sources = [x is not None for x in (roi_xml, samples_csv, label_raster)]
        if sum(sources) != 1:
            raise ValueError("Specify exactly one label source: --roi-xml, --samples-csv, or --label-raster.")

        if roi_xml is not None:
            labels = load_envi_pixeldef_xml(Path(roi_xml), exclude_classes, transform=ds.transform)
            label_source = {"type": "envi_pixeldef_xml", "path": str(roi_xml)}
        elif samples_csv is not None:
            labels = load_samples_csv(Path(samples_csv), exclude_classes)
            label_source = {"type": "samples_csv", "path": str(samples_csv)}
        else:
            class_map = load_class_map(Path(args.class_map) if args.class_map else (Path(config["class_map"]) if config.get("class_map") else None))
            labels = load_label_raster_samples(
                Path(label_raster),
                max_pool_per_class=max(4 * max_samples_per_class, max_samples_per_class),
                seed=seed,
                class_map=class_map,
                exclude_classes=exclude_classes,
            )
            label_source = {"type": "label_raster", "path": str(label_raster)}

        labels = check_label_bounds(labels, ds.height, ds.width)
        labels = drop_conflicting_pixels(labels)
        if labels.provided_groups is not None:
            groups_all = labels.provided_groups.copy()
            unmatched = groups_all < 0
            if np.any(unmatched):
                fallback = assign_spatial_groups(labels.rows[unmatched], labels.cols[unmatched], group_block_size)
                offset = int(groups_all[~unmatched].max() + 1) if np.any(~unmatched) else 0
                groups_all[unmatched] = fallback + offset
                group_source = "ENVI_GeometryDef_polygons_plus_spatial_fallback"
            elif label_source["type"] == "samples_csv":
                group_source = "csv_group_column"
            else:
                group_source = "ENVI_GeometryDef_polygons"
        else:
            groups_all = assign_spatial_groups(labels.rows, labels.cols, group_block_size)
            group_source = f"spatial_blocks_{group_block_size}px"

        if group_merge_px > 0:
            n_before = int(np.unique(groups_all).size)
            groups_all = merge_groups_by_proximity(groups_all, labels.rows, labels.cols, group_merge_px)
            n_after = int(np.unique(groups_all).size)
            group_source += f"+merged_within_{group_merge_px:g}px"
            log(f"Merged spatially adjacent groups (<= {group_merge_px:g} px): {n_before} -> {n_after}")

        sample_idx = spatially_balanced_sample_indices(
            labels.class_ids,
            groups_all,
            max_per_class=max_samples_per_class,
            seed=seed,
        )
        rows = labels.rows[sample_idx]
        cols = labels.cols[sample_idx]
        y = labels.class_ids[sample_idx]
        groups = groups_all[sample_idx]

        log("Classes:")
        for cid, name in enumerate(labels.class_names):
            log(
                f"  {cid}: {name} | available={int(np.sum(labels.class_ids == cid)):,} | "
                f"sampled={int(np.sum(y == cid)):,} | groups={int(np.unique(groups[y == cid]).size)}"
            )

        raw_spectra, valid = read_sample_spectra(ds, rows, cols, tile_size=tile_size)
        if not np.all(valid):
            n_bad = int((~valid).sum())
            warnings.warn(f"Dropping {n_bad} sampled training pixels that are masked or non-finite in the raster.")
            rows, cols, y, groups, raw_spectra = (
                rows[valid],
                cols[valid],
                y[valid],
                groups[valid],
                raw_spectra[valid],
            )

        if len(np.unique(y)) != len(labels.class_names):
            missing_names = [labels.class_names[c] for c in range(len(labels.class_names)) if not np.any(y == c)]
            raise RuntimeError(f"Some classes have no valid sampled pixels after raster masking: {missing_names}")

        coords = np.column_stack([rows, cols]).astype(np.float64)

        prevalence = (
            load_class_prevalence(Path(class_prevalence_path), labels.class_names)
            if class_prevalence_path
            else None
        )

        split = np.full(len(y), "development", dtype=object)
        hold_data: dict[str, np.ndarray] | None = None
        if holdout_fraction > 0:
            dev_idx, hold_idx = split_holdout_groups(y, groups, holdout_fraction, seed)
            split[hold_idx] = "holdout"
            hold_data = {
                "raw": raw_spectra[hold_idx],
                "y": y[hold_idx],
                "groups": groups[hold_idx],
                "coords": coords[hold_idx],
            }
            log(
                f"Hold-out: {len(hold_idx):,} pixels in {np.unique(groups[hold_idx]).size} groups reserved "
                f"for one final evaluation; {len(dev_idx):,} pixels used for development."
            )

        sample_manifest = pd.DataFrame(
            {
                "row": rows,
                "col": cols,
                "class_id": y,
                "class_name": [labels.class_names[c] for c in y],
                "group": groups,
                "split": split,
            }
        )
        sample_manifest.to_csv(tables_dir / "training_sample_manifest.csv", index=False)

        if hold_data is not None:
            # Everything below (band statistics, model selection, CV) sees development pixels only.
            rows, cols, y, groups, raw_spectra, coords = (
                a[dev_idx] for a in (rows, cols, y, groups, raw_spectra, coords)
            )

        gpc = groups_per_class(y, groups)
        if min(gpc.values()) < 8:
            warnings.warn(
                f"Some classes have fewer than 8 independent groups (groups per class: {gpc}). Bootstrap "
                f"confidence intervals and fold-to-fold estimates are unreliable at this size; treat the "
                f"results as indicative and add independent training regions if possible."
            )

        global_sample = sample_global_raster_pixels(ds, global_band_samples, seed + 81)
        band_diag = band_diagnostics(
            global_sample,
            raw_spectra,
            y,
            band_names,
            wavelengths_nm,
            ds.dtypes[0],
        )
        band_diag.to_csv(tables_dir / "band_diagnostics.csv", index=False)
        log("Note: band_diagnostics.csv is screening information only. Dropping bands based on it and then "
            "cross-validating on the same pixels leaks label information (see docs/METHODOLOGY.md).")
        save_band_diagnostics_figure(band_diag, figures_dir / "band_diagnostics.png")

        class_stats = class_spectral_statistics(
            raw_spectra,
            y,
            labels.class_names,
            band_names,
            wavelengths_nm,
        )
        class_stats.to_csv(tables_dir / "class_spectral_statistics.csv", index=False)
        save_spectral_signatures_figure(
            class_stats,
            labels.class_names,
            figures_dir / "class_spectral_signatures.png",
        )

        separability = class_mean_spectral_separability(raw_spectra, y, labels.class_names)
        separability.to_csv(tables_dir / "class_mean_spectral_separability.csv", index=False)

    # Raster is closed here; validation uses sampled arrays only.
    validation_results: list[ValidationResult] = []
    failed_runs: list[dict[str, str]] = []
    feature_cache: dict[str, tuple[np.ndarray, FeatureContext]] = {}

    for spec in feature_specs:
        log(f"\nFeature set: {spec.name}")
        X, context = build_features(
            raw_spectra,
            selected_band_indices,
            band_names,
            wavelengths_nm,
            spec,
            index_tolerance_nm,
        )
        feature_cache[spec.name] = (X, context)
        pd.DataFrame({"feature": context.feature_names}).to_csv(
            tables_dir / f"features_{sanitize_name(spec.name)}.csv", index=False
        )
        if context.index_definitions_used:
            (tables_dir / f"index_definitions_{sanitize_name(spec.name)}.txt").write_text(
                "\n".join(context.index_definitions_used) + "\n",
                encoding="utf-8",
            )

        for model_name in models:
            log(f"  Model: {model_name}")
            try:
                result = evaluate_model(
                    X=X,
                    y=y,
                    groups=groups,
                    class_names=labels.class_names,
                    feature_context=context,
                    model_name=model_name,
                    outer_folds=outer_folds,
                    inner_folds=inner_folds,
                    bootstrap_reps=bootstrap_reps,
                    n_jobs=n_jobs,
                    seed=seed,
                    svm_cap=svm_cap,
                    knn_cap=knn_cap,
                    coords=coords,
                    buffer_px=buffer_px,
                    repeats=repeats,
                )
            except Exception as exc:
                if os.environ.get("SPECTRAL_BENCHMARK_DEBUG") == "1":
                    raise
                warnings.warn(f"Model {model_name} / feature set {spec.name} failed: {exc}")
                failed_runs.append(
                    {
                        "feature_set": spec.name,
                        "model": model_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            validation_results.append(result)

            base = f"{sanitize_name(spec.name)}_{sanitize_name(model_name)}"
            pd.DataFrame(result.fold_metrics).to_csv(tables_dir / f"fold_metrics_{base}.csv", index=False)
            result.per_class.assign(feature_set=spec.name, model=model_name).to_csv(
                tables_dir / f"per_class_{base}.csv", index=False
            )
            pd.DataFrame(result.confusion, index=labels.class_names, columns=labels.class_names).to_csv(
                tables_dir / f"confusion_counts_{base}.csv"
            )
            pd.DataFrame(result.confusion_norm, index=labels.class_names, columns=labels.class_names).to_csv(
                tables_dir / f"confusion_row_normalized_{base}.csv"
            )
            pd.DataFrame(
                [
                    {"fold": i + 1, **params}
                    for i, params in enumerate(result.best_params_per_fold)
                ]
            ).to_csv(tables_dir / f"best_params_{base}.csv", index=False)
            oof = pd.DataFrame(
                {
                    "row": rows,
                    "col": cols,
                    "group": groups,
                    "reference_id": result.y_true,
                    "reference_class": [labels.class_names[c] for c in result.y_true],
                    "predicted_id": result.y_pred,
                    "predicted_class": [labels.class_names[c] for c in result.y_pred],
                }
            )
            if result.y_pred_repeats is not None:
                for r_i in range(1, result.n_repeats):
                    oof[f"predicted_id_repeat{r_i + 1}"] = result.y_pred_repeats[r_i]
            oof.to_csv(tables_dir / f"oof_predictions_{base}.csv", index=False)
            save_confusion_figure(
                result.confusion_norm,
                labels.class_names,
                f"{spec.name} | {model_name}",
                figures_dir / f"confusion_{base}.png",
            )

    if failed_runs:
        pd.DataFrame(failed_runs).to_csv(tables_dir / "failed_runs.csv", index=False)
        log(f"\n{len(failed_runs)} model/feature-set run(s) failed; see tables/failed_runs.csv")
    if not validation_results:
        raise RuntimeError("No classifier completed successfully.")

    comparison_rows: list[dict[str, Any]] = []
    area_weighted_rows: list[pd.DataFrame] = []
    all_fold_rows: list[dict[str, Any]] = []
    all_per_class_rows: list[pd.DataFrame] = []
    for result in validation_results:
        context = feature_cache[result.feature_set][1]
        comparison_rows.append({
            "feature_set": result.feature_set,
            "model": result.model,
            "feature_usage": "base_spectrum_only" if result.model.upper() in {"SAM", "SID"} else "all_features",
            "n_features_total": len(context.feature_names),
            "n_spectral_features": context.base_feature_count,
            **result.summary,
            **(area_weighted_metrics(result.confusion, prevalence, labels.class_names)[0] if prevalence is not None else {}),
        })
        if prevalence is not None:
            area_weighted_rows.append(
                area_weighted_metrics(result.confusion, prevalence, labels.class_names)[1].assign(
                    feature_set=result.feature_set, model=result.model
                )
            )
        for rec in result.fold_metrics:
            all_fold_rows.append({"feature_set": result.feature_set, "model": result.model, **rec})
        all_per_class_rows.append(result.per_class.assign(feature_set=result.feature_set, model=result.model))

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["macro_F1", "OA"], ascending=[False, False]
    )
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    if area_weighted_rows:
        pd.concat(area_weighted_rows, ignore_index=True).to_csv(
            tables_dir / "area_weighted_per_class_metrics.csv", index=False
        )
    pd.DataFrame(all_fold_rows).to_csv(tables_dir / "all_fold_metrics.csv", index=False)
    per_class_all = pd.concat(all_per_class_rows, ignore_index=True)
    per_class_all.to_csv(tables_dir / "all_per_class_metrics.csv", index=False)

    pairwise = pairwise_bootstrap_comparisons(
        validation_results,
        pairwise_bootstrap_reps,
        seed + 12000,
    )
    pairwise.to_csv(tables_dir / "pairwise_model_comparisons.csv", index=False)

    save_model_comparison_figures(comparison, figures_dir)
    save_per_class_f1_heatmap(per_class_all, figures_dir)

    best = comparison.iloc[0]
    log(
        f"\nBest validation result: feature_set={best['feature_set']}, model={best['model']}, "
        f"macro-F1={best['macro_F1']:.3f}, OA={best['OA']:.3f}"
    )

    holdout_summary: dict[str, Any] | None = None
    if hold_data is not None:
        X_dev_best, ctx_best = feature_cache[str(best["feature_set"])]
        X_hold_best, _ = build_features(
            hold_data["raw"], selected_band_indices, band_names, wavelengths_nm, ctx_best.spec, index_tolerance_nm
        )
        log(f"\nEvaluating the selected combination once on the hold-out groups: "
            f"{best['feature_set']} | {best['model']}")
        record, hold_per_class, hold_cm = evaluate_holdout(
            str(best["model"]), X_dev_best, y, groups, coords,
            X_hold_best, hold_data["y"], hold_data["groups"], hold_data["coords"],
            labels.class_names, ctx_best, inner_folds, n_jobs, seed, svm_cap, knn_cap, buffer_px, bootstrap_reps,
        )
        record["development_cv_macro_F1"] = float(best["macro_F1"])
        record["development_cv_OA"] = float(best["OA"])
        pd.DataFrame([record]).to_csv(tables_dir / "holdout_result.csv", index=False)
        hold_per_class.to_csv(tables_dir / "holdout_per_class_metrics.csv", index=False)
        pd.DataFrame(hold_cm, index=labels.class_names, columns=labels.class_names).to_csv(
            tables_dir / "holdout_confusion_counts.csv"
        )
        holdout_summary = record
        log(
            f"  hold-out macro-F1={record['macro_F1']:.3f} (development CV: {best['macro_F1']:.3f}), "
            f"OA={record['OA']:.3f} (development CV: {best['OA']:.3f})"
        )

    # Optional final maps.
    map_requests = parse_csv_list(args.map) if args.map is not None else [str(x) for x in config.get("map", [])]
    map_top = int(config_value(args.map_top, config, "map_top", 0))
    if map_top > 0:
        for _, row in comparison.head(map_top).iterrows():
            req = f"{row['feature_set']}:{row['model']}"
            if req not in map_requests:
                map_requests.append(req)

    map_outputs: list[dict[str, Any]] = []
    for request in map_requests:
        if ":" not in request:
            raise ValueError(
                f"Map request {request!r} must use feature_set:model syntax, e.g. raw:SVM."
            )
        feature_name, model_name = request.split(":", 1)
        feature_name = feature_name.strip()
        model_name = model_name.strip().upper()
        if feature_name not in feature_cache:
            raise ValueError(f"Map feature set {feature_name!r} was not evaluated.")
        if model_name not in models:
            raise ValueError(f"Map model {model_name!r} was not evaluated.")

        if min_confidence > 0 and model_name in {"SAM", "SID", "MIN_DIST", "MAHALANOBIS", "SVM"}:
            warnings.warn(
                f"--min-confidence with {model_name}: its confidence is a relative softmax of distances / "
                f"decision values, not a probability, so a fixed threshold can reject most pixels. Prefer "
                f"RF, HGB, LDA, QDA or KNN for rejection, and inspect the confidence raster first."
            )
        X, context = feature_cache[feature_name]
        y_map, groups_map, coords_map = y, groups, coords
        if hold_data is not None:
            # Evaluation is finished, so the deployed map model uses every labelled pixel.
            X_hold_map, _ = build_features(
                hold_data["raw"], selected_band_indices, band_names, wavelengths_nm, context.spec, index_tolerance_nm
            )
            X = np.vstack([X, X_hold_map])
            y_map = np.concatenate([y, hold_data["y"]])
            groups_map = np.concatenate([groups, hold_data["groups"]])
            coords_map = np.vstack([coords, hold_data["coords"]])
        log(f"\nFitting final model for map: {feature_name} | {model_name}")
        final_model, final_params, n_fit = fit_final_model(
            model_name,
            X,
            y_map,
            groups_map,
            context,
            inner_folds,
            n_jobs,
            seed,
            svm_cap,
            knn_cap,
            coords=coords_map,
            buffer_px=buffer_px,
        )
        write_confidence = bool(args.write_confidence or config.get("write_confidence", False))
        class_path, conf_path = map_full_raster(
            image,
            maps_dir,
            final_model,
            model_name,
            context,
            selected_band_indices,
            band_names,
            wavelengths_nm,
            labels.class_names,
            labels.colors,
            tile_size,
            write_confidence,
            index_tolerance_nm,
            min_confidence=min_confidence,
        )
        preview_path = figures_dir / f"preview_{sanitize_name(feature_name)}_{sanitize_name(model_name)}.png"
        save_classification_preview(class_path, labels.colors, labels.class_names, preview_path)
        map_outputs.append(
            {
                "feature_set": feature_name,
                "model": model_name,
                "classification": str(class_path),
                "confidence": str(conf_path) if conf_path else None,
                "preview": str(preview_path),
                "final_best_params": final_params,
                "n_fit": n_fit,
            }
        )

    manifest = {
        "spectral_benchmark_version": VERSION,
        "created_unix": time.time(),
        "libraries": library_versions(),
        "raster": raster_info,
        "labels": label_source,
        "excluded_classes": sorted(exclude_classes),
        "class_names": labels.class_names,
        "class_colors": labels.colors,
        "group_source": group_source,
        "group_block_size_px": group_block_size,
        "max_samples_per_class": max_samples_per_class,
        "outer_folds_requested": outer_folds,
        "inner_folds_requested": inner_folds,
        "bootstrap_reps": bootstrap_reps,
        "pairwise_bootstrap_reps": pairwise_bootstrap_reps,
        "buffer_px": buffer_px,
        "repeats": repeats,
        "group_merge_px": group_merge_px,
        "holdout_fraction": holdout_fraction,
        "class_prevalence": str(class_prevalence_path) if class_prevalence_path else None,
        "min_confidence": min_confidence,
        "holdout_result": holdout_summary,
        "n_groups": int(np.unique(groups).size),
        "seed": seed,
        "n_jobs": n_jobs,
        "svm_train_cap": svm_cap,
        "knn_train_cap": knn_cap,
        "tile_size": tile_size,
        "index_tolerance_nm": index_tolerance_nm,
        "band_analysis_samples": global_band_samples,
        "models": models,
        "feature_sets": [asdict(x) for x in feature_specs],
        "failed_runs": failed_runs,
        "config_file": str(args.config) if args.config else None,
        "map_outputs": map_outputs,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    summary_lines = [
        f"Spectral Benchmark v{VERSION}",
        "",
        f"Input: {image}",
        f"Label source: {label_source}",
        f"Classes: {', '.join(labels.class_names)}",
        f"Training samples used: {len(y):,}",
        f"Spatial groups: {len(np.unique(groups)):,} ({group_source})",
        "",
        "Best validation result",
        f"  Feature set: {best['feature_set']}",
        f"  Model: {best['model']}",
        f"  Macro-F1: {best['macro_F1']:.4f}",
        f"  OA: {best['OA']:.4f}",
        f"  Kappa: {best['kappa']:.4f}",
        "",
        "Primary reusable outputs",
        "  tables/model_comparison.csv",
        "  tables/all_per_class_metrics.csv",
        "  tables/pairwise_model_comparisons.csv",
        "  tables/band_diagnostics.csv",
        "  tables/class_spectral_statistics.csv",
        "  figures/model_comparison_macro_F1.png",
        "  figures/model_comparison_OA.png",
        "  figures/per_class_F1_heatmap.png",
        "  figures/class_spectral_signatures.png",
        "  figures/band_diagnostics.png",
        "",
        *(
            [
                "Independent hold-out (scored once, after model selection)",
                f"  Macro-F1: {holdout_summary['macro_F1']:.4f}   OA: {holdout_summary['OA']:.4f}",
                "  See tables/holdout_result.csv",
                "",
            ]
            if holdout_summary
            else []
        ),
        "Interpretation note",
        "  Band diagnostics are conservative screening aids, not automatic scientific exclusion rules.",
        "  Final band removal should be justified with validation performance and domain knowledge.",
    ]
    (output_dir / "RESULTS_README.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    log(f"\nDone. Results: {output_dir}")
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reusable grouped-validation benchmark for multispectral/hyperspectral supervised classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="Optional JSON configuration file.")
    p.add_argument("--image", help="Input multiband raster readable by GDAL/rasterio.")
    p.add_argument("--output-dir", help="Output directory.")

    label = p.add_argument_group("Training labels")
    label.add_argument("--roi-xml", help="ENVI RegionsOfInterest XML containing PixelDef coordinates.")
    label.add_argument("--samples-csv", help="CSV with row,col,class and optional group,r,g,b columns.")
    label.add_argument("--label-raster", help="Integer one-band training label raster; 0 = unlabelled.")
    label.add_argument("--class-map", help="Optional CSV for label-raster codes: code,name[,r,g,b].")
    label.add_argument("--exclude-classes", help="Comma-separated class names to exclude from training.")

    bands = p.add_argument_group("Bands and feature sets")
    bands.add_argument("--wavelengths", help="Comma-separated wavelength centres in nm, one per raster band.")
    bands.add_argument("--keep-bands", help="Comma-separated 1-based band indices to keep.")
    bands.add_argument("--drop-bands", help="Comma-separated 1-based band indices to drop.")
    bands.add_argument("--keep-wavelengths", help="Comma-separated wavelengths to keep (nearest match).")
    bands.add_argument("--drop-wavelengths", help="Comma-separated wavelengths to drop (nearest match).")
    bands.add_argument(
        "--feature-sets",
        help=(
            "Comma-separated presets: raw,mean_norm,raw_indices,mean_norm_indices,"
            "raw_derivative,mean_norm_derivative,raw_indices_derivative,"
            "mean_norm_indices_derivative"
        ),
    )
    bands.add_argument(
        "--indices",
        help="Comma-separated built-in indices: GLI,TGI,RGRI,VARI,NGRDI,BG_RATIO,GR_RATIO,NDVI.",
    )
    bands.add_argument("--index-tolerance-nm", type=float, default=None)

    validation = p.add_argument_group("Validation")
    validation.add_argument("--models", help=f"Comma-separated models. Available: {','.join(DEFAULT_MODELS)}")
    validation.add_argument("--outer-folds", type=int, default=None)
    validation.add_argument("--inner-folds", type=int, default=None)
    validation.add_argument("--bootstrap", type=int, default=None, help="Grouped bootstrap replicates for metric CIs.")
    validation.add_argument("--pairwise-bootstrap", type=int, default=None)
    validation.add_argument("--max-samples-per-class", type=int, default=None)
    validation.add_argument("--group-block-size", type=int, default=None, help="Spatial grouping block size in pixels.")
    validation.add_argument("--svm-train-cap", type=int, default=None, help="Maximum training samples per SVM fit; 0 disables cap.")
    validation.add_argument("--knn-train-cap", type=int, default=None, help="Maximum training samples per kNN fit; 0 disables cap.")
    validation.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for inner GridSearchCV.")
    validation.add_argument("--seed", type=int, default=None)
    validation.add_argument(
        "--buffer-px",
        type=float,
        default=None,
        help="Exclude training pixels closer than this many pixels to any test pixel (outer and inner folds). 0 = off.",
    )
    validation.add_argument(
        "--group-merge-px",
        type=float,
        default=None,
        help="Merge groups (polygons/blocks) that have pixels within this many pixels of each other. 0 = off.",
    )
    validation.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Repeat the outer grouped CV with different fold assignments and average (1 = single run).",
    )

    validation.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help="Reserve this fraction of spatial groups (0-0.5) as an independent test set, scored once for the "
        "best development combination. 0 = off.",
    )
    validation.add_argument(
        "--class-prevalence",
        help="CSV with columns class,area (or proportion): scene share of each reference class. Adds "
        "area-weighted OA / F1 / kappa.",
    )

    diagnostics = p.add_argument_group("Diagnostics")
    diagnostics.add_argument("--band-analysis-samples", type=int, default=None, help="Approximate global raster pixels sampled for band diagnostics.")

    mapping = p.add_argument_group("Optional full-raster mapping")
    mapping.add_argument(
        "--map",
        help="Comma-separated feature_set:model requests, e.g. raw:SVM,raw:SAM. No mapping is done by default.",
    )
    mapping.add_argument("--map-top", type=int, default=None, help="Also map the top N validation combinations.")
    mapping.add_argument("--tile-size", type=int, default=None)
    mapping.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Map pixels whose confidence is below this value to an 'Unclassified' code. Confidence is only "
        "meaningful for models with predict_proba (RF, HGB, LDA, QDA, KNN); 0 = off.",
    )
    mapping.add_argument("--write-confidence", action="store_true", help="Write float32 confidence raster(s); can be very large.")

    return p


def validate_required_args(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for key in ("image", "output_dir"):
        if getattr(args, key) is None and config.get(key) is None:
            raise SystemExit(f"Missing required --{key.replace('_', '-')} (or set {key!r} in config JSON).")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config_file(Path(args.config) if args.config else None)
    validate_required_args(args, config)
    try:
        return run(args)
    except KeyboardInterrupt:
        log("Interrupted by user.")
        return 130
    except Exception as exc:
        log(f"ERROR: {exc}")
        if os.environ.get("SPECTRAL_BENCHMARK_DEBUG") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
