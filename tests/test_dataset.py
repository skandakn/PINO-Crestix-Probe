"""Unit tests for UKEA 32x32 dataset adapter, preprocessing, and event-level splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from crestix_pino.data.dataloader import build_dataloaders
from crestix_pino.data.dataset import UKEAPinoDataset
from crestix_pino.data.preprocessing import (
    UKEA_TEST_EVENTS,
    UKEA_TRAIN_EVENTS,
    align_rainfall_to_depth,
    extract_region_32x32,
    inspect_ukea_event,
    preprocess_dem,
    split_events_deterministically,
)
from crestix_pino.training.config import DataConfig


def _create_synthetic_ukea(root: Path, height: int = 50, width: int = 120) -> Path:
    geo_dir = root / "geodata"
    geo_dir.mkdir(parents=True, exist_ok=True)
    dem = np.random.RandomState(0).uniform(10.0, 20.0, size=(height, width)).astype(np.float32)
    dem[12:15, 50:53] = np.nan  # synthetic building footprint
    np.save(geo_dir / "dem.npy", dem)

    for idx, ev in enumerate(list(UKEA_TRAIN_EVENTS) + list(UKEA_TEST_EVENTS)):
        ev_dir = root / "flood" / ev
        ev_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(idx + 1)
        h = rng.uniform(0.0, 0.5, size=(36, height, width)).astype(np.float64)
        rain = rng.uniform(0.0, 2.0, size=(360, height, width)).astype(np.float64)
        np.save(ev_dir / "h.npy", h)
        np.save(ev_dir / "rainfall.npy", rain)

    return root


def test_official_event_split_is_disjoint() -> None:
    splits = split_events_deterministically(val_count=2)
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 12
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_dimension_verification_and_32x32_crop(tmp_path: Path) -> None:
    root = _create_synthetic_ukea(tmp_path)
    ev_dir = root / "flood" / UKEA_TRAIN_EVENTS[0]
    meta = inspect_ukea_event(ev_dir, root / "geodata" / "dem.npy")

    assert meta["dem_shape"] == (50, 120)
    assert meta["h_shape"] == (36, 50, 120)
    assert meta["rainfall_shape"] == (360, 50, 120)
    assert meta["num_time_steps"] == 36
    assert meta["rainfall_time_steps"] == 360
    assert meta["substeps_per_frame"] == 10

    dem = np.load(root / "geodata" / "dem.npy")
    dem_32 = extract_region_32x32(dem, crop_y=9, crop_x=44, height=32, width=32)
    assert dem_32.shape == (32, 32)
    dem_clean, bmask, vmask = preprocess_dem(dem_32)
    assert not np.isnan(dem_clean).any()
    assert bmask.sum() > 0
    assert (bmask == ~vmask).all()


def test_rainfall_alignment_and_dataset_mmap(tmp_path: Path) -> None:
    root = _create_synthetic_ukea(tmp_path)
    ds = UKEAPinoDataset(data_root=root, split="train", window_t=12, stride_t=6)
    assert len(ds) > 0

    sample = ds[0]
    assert sample["inputs"].shape == (4, 12, 32, 32)
    assert sample["targets"].shape == (12, 32, 32)
    assert sample["dem"].shape == (32, 32)
    assert sample["rainfall_m_s"].shape == (12, 32, 32)
    assert sample["h_init"].shape == (32, 32)
    assert sample["valid_mask"].shape == (32, 32)
    assert torch.isfinite(sample["inputs"]).all()
    assert torch.isfinite(sample["targets"]).all()
    assert sample["event_id"] in UKEA_TRAIN_EVENTS

    # Verify rainfall alignment helper directly
    r_360 = np.ones((360, 32, 32), dtype=np.float32)
    aligned = align_rainfall_to_depth(r_360, target_steps=36, unit_scale_to_m_s=1.0)
    assert aligned.shape == (36, 32, 32)
    assert np.allclose(aligned, 10.0)


def test_dataloaders_respect_event_splits(tmp_path: Path) -> None:
    root = _create_synthetic_ukea(tmp_path)
    cfg = DataConfig(data_root=str(root), batch_size=2, window_t=12, stride_t=12)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, seed=42)

    train_events = {ev for b in train_loader for ev in b["event_id"]}
    val_events = {ev for b in val_loader for ev in b["event_id"]}
    test_events = {ev for b in test_loader for ev in b["event_id"]}

    assert train_events.isdisjoint(val_events)
    assert train_events.isdisjoint(test_events)
    assert val_events.isdisjoint(test_events)
