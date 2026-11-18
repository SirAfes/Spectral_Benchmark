#!/usr/bin/env python3
"""End-to-end smoke test for Spectral Benchmark.

Creates a small synthetic four-band raster and ENVI PixelDef XML, runs several
classifier families, tests feature engineering, and verifies key output files.

Run with ``pytest`` (needs rasterio) or directly: ``python tests/test_smoke.py``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402


def make_data(root: Path) -> tuple[Path, Path]:
    h = w = 96
    wavelengths = [450, 550, 650, 850]
    means = np.array([[45, 90, 55, 20], [35, 70, 95, 25], [70, 55, 40, 60]], dtype=float)
    rng = np.random.default_rng(4)
    data = np.zeros((4, h, w), dtype=np.uint8)

    for br in range(3):
        for bc in range(3):
            cid = (br + bc) % 3
            r0, r1 = br * 32, (br + 1) * 32
            c0, c1 = bc * 32, (bc + 1) * 32
            block = means[cid, :, None, None] * (0.8 + 0.15 * ((br * 3 + bc) % 3))
            block = block + rng.normal(0, 4, (4, 32, 32))
            data[:, r0:r1, c0:c1] = np.clip(block, 0, 255).astype(np.uint8)

    image = root / "image.tif"
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=4,
        dtype="uint8",
        transform=from_origin(500000, 6000000, 0.1, 0.1),
        crs="EPSG:32629",
    ) as ds:
        ds.write(data)
        ds.update_tags(WAVELENGTH="{450, 550, 650, 850}", WAVELENGTH_UNITS="Nanometers")
        for i, wl in enumerate(wavelengths, start=1):
            ds.set_band_description(i, f"{wl} nm")

    xml_path = root / "rois.xml"
    root_xml = ET.Element("RegionsOfInterest", version="1.1")
    for cid, name in enumerate(["A", "B", "C"]):
        region = ET.SubElement(root_xml, "Region", name=name, color="100,150,200")
        pixel_def = ET.SubElement(region, "PixelDef")
        coords = ET.SubElement(pixel_def, "Coordinates")
        values: list[str] = []
        for br in range(3):
            for bc in range(3):
                if (br + bc) % 3 != cid:
                    continue
                for row in range(br * 32 + 8, br * 32 + 24):
                    for col in range(bc * 32 + 8, bc * 32 + 24):
                        values.extend([str(col), str(row)])
        coords.text = " ".join(values)
    ET.ElementTree(root_xml).write(xml_path, encoding="utf-8", xml_declaration=True)
    return image, xml_path


def run_pipeline(root: Path, extra: list[str] | None = None) -> Path:
    script = Path(__file__).resolve().parents[1] / "spectral_benchmark.py"
    image, xml_path = make_data(root)
    out = root / "results"
    cmd = [
        sys.executable,
        str(script),
        "--image", str(image),
        "--roi-xml", str(xml_path),
        "--output-dir", str(out),
        "--models", "SVM,SAM,SID,MIN_DIST,MAHALANOBIS,QDA",
        "--feature-sets", "raw,mean_norm,raw_indices,raw_derivative",
        "--indices", "GLI,RGRI,NDVI",
        "--outer-folds", "3",
        "--inner-folds", "2",
        "--bootstrap", "3",
        "--pairwise-bootstrap", "3",
        "--max-samples-per-class", "60",
        "--group-block-size", "32",
        "--svm-train-cap", "150",
        "--n-jobs", "1",
        "--band-analysis-samples", "200",
        "--map-top", "1",
        # 50 is deliberately not a multiple of 16: GeoTIFF block sizes must be.
        "--tile-size", "50",
    ]
    # Later occurrences override earlier ones in argparse, so ``extra`` can change any option.
    subprocess.run(cmd + (extra or []), check=True)
    return out


def check_outputs(out: Path) -> None:
    required = [
        out / "tables" / "model_comparison.csv",
        out / "tables" / "all_per_class_metrics.csv",
        out / "tables" / "band_diagnostics.csv",
        out / "figures" / "class_spectral_signatures.png",
        out / "figures" / "model_comparison_macro_F1.png",
        out / "run_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if not list((out / "maps").glob("classification_*.tif")):
        missing.append("classification map")
    assert not missing, f"Smoke test missing outputs: {missing}"

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 0 and manifest["failed_runs"] == []
    assert not (out / "tables" / "failed_runs.csv").exists()


def test_end_to_end(tmp_path: Path) -> None:
    check_outputs(run_pipeline(tmp_path))


def test_end_to_end_with_spatial_options(tmp_path: Path) -> None:
    """Hold-out, spatial buffer, repeats, area weighting and rejection run together."""
    prev = tmp_path / "prevalence.csv"
    prev.write_text("class,area\nA,700\nB,200\nC,100\n", encoding="utf-8")
    out = run_pipeline(
        tmp_path,
        [
            "--models", "LDA,SAM",
            "--feature-sets", "raw",
            "--buffer-px", "3",
            "--group-merge-px", "4",
            "--repeats", "2",
            "--holdout-fraction", "0.34",
            "--class-prevalence", str(prev),
            "--map", "raw:LDA",
            "--min-confidence", "0.5",
        ],
    )
    import pandas as pd

    tables = out / "tables"
    assert (tables / "holdout_result.csv").exists()
    assert (tables / "area_weighted_per_class_metrics.csv").exists()
    comparison = pd.read_csv(tables / "model_comparison.csv")
    assert {"OA_area_weighted", "min_train_test_dist_px", "min_groups_per_class"} <= set(comparison.columns)
    assert (comparison["n_repeats"] == 2).all()
    split = pd.read_csv(tables / "training_sample_manifest.csv")["split"]
    assert set(split) == {"development", "holdout"}
    legend = pd.read_csv(out / "maps" / "legend.csv")
    assert legend["class_name"].iloc[-1] == "Unclassified"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout_result"] is not None and manifest["holdout_fraction"] == 0.34


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        check_outputs(run_pipeline(Path(tmp)))
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
