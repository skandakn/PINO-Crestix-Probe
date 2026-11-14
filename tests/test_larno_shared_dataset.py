"""Unit tests for the official 1000-event LarNO shared data pipeline in pino-crestix."""

from __future__ import annotations

import csv
import json
import sys
import unittest.mock as mock
from pathlib import Path

# Provide mock for torch if torch is not installed in the testing environment
if "torch" not in sys.modules:
    try:
        import torch
    except ImportError:
        m = mock.MagicMock()
        sys.modules["torch"] = m
        sys.modules["torch.nn"] = m
        sys.modules["torch.nn.functional"] = m
        sys.modules["torch.utils"] = m
        sys.modules["torch.utils.data"] = m

import numpy as np
import pytest

from crestix_pino.data.preprocessing import (
    align_rainfall_substeps,
    preprocess_dem,
    LarNONormalizer,
    rainfall_to_flux_m_s,
)


def test_event_manifest_counts_and_attributes() -> None:
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "event_manifest.csv"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 1000, f"Expected 1000 events, found {len(rows)}"

    futian_rows = [r for r in rows if r["catchment"] == "Futian"]
    ukea_rows = [r for r in rows if r["catchment"] == "UKEA"]

    assert len(futian_rows) == 80, f"Expected 80 Futian events, got {len(futian_rows)}"
    assert len(ukea_rows) == 20, f"Expected 20 UKEA events, got {len(ukea_rows)}"

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]

    assert len(train_rows) == 80
    assert len(val_rows) == 10
    assert len(test_rows) == 10

    for r in rows:
        assert int(r["dem_nan_count"]) == 0
        assert int(r["rainfall_nan_count"]) == 0
        assert int(r["water_depth_nan_count"]) == 0
        assert r["water_depth_units"] == "m"


def test_split_manifest_disjointness() -> None:
    split_path = Path(__file__).resolve().parent.parent / "data" / "split_manifest.json"
    assert split_path.exists()

    with open(split_path, "r", encoding="utf-8") as fp:
        split_data = json.load(fp)

    splits = split_data["splits"]
    train_events = set(splits["train"]["futian_events"] + splits["train"]["ukea_events"])
    val_events = set(splits["val"]["futian_events"] + splits["val"]["ukea_events"])
    test_events = set(splits["test"]["futian_events"] + splits["test"]["ukea_events"])

    assert len(train_events) == 80
    assert len(val_events) == 10
    assert len(test_events) == 10

    # Ensure zero overlap
    assert train_events.isdisjoint(val_events)
    assert train_events.isdisjoint(test_events)
    assert val_events.isdisjoint(test_events)
    assert len(train_events | val_events | test_events) == 1000


def test_rainfall_alignment_and_unit_conversion() -> None:
    # 360 sub-steps (30s) summed into 36 frames (5min)
    r_360 = np.ones((360, 50, 120), dtype=np.float32) * 2.0
    aligned = align_rainfall_substeps(r_360, target_steps=36, mode="sum")
    assert aligned.shape == (36, 50, 120)
    assert np.allclose(aligned, 20.0)

    # Convert 300 mm / 5min to flux m/s
    rain_mm = np.array([300.0], dtype=np.float32)
    flux = rainfall_to_flux_m_s(rain_mm)
    assert np.isclose(flux[0], 0.001, atol=1e-6)


def test_dem_preprocessing_and_building_masks() -> None:
    dem = np.full((50, 120), 25.0, dtype=np.float32)
    dem[5, 5] = np.nan

    clean, bmask, vmask = preprocess_dem(dem, wall_height=40.0)
    assert not np.isnan(clean).any()
    assert bmask[5, 5] == True
    assert vmask[0, 0] == True
    assert (bmask == ~vmask).all()


def test_train_only_normalization_moments() -> None:
    stats_path = Path(__file__).resolve().parent.parent / "data" / "dataset_statistics.json"
    norm = LarNONormalizer("futian", stats_file=stats_path)
    dem_mean = norm.dem_stats["mean"]
    arr = np.full((10, 10), dem_mean, dtype=np.float32)
    norm_arr = norm.normalize_dem(arr)
    assert np.allclose(norm_arr, 0.0, atol=1e-4)
