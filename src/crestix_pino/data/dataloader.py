"""Deterministic, memory-efficient PyTorch DataLoader builders."""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import UKEAPinoDataset


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(
    data_cfg: Any,
    seed: int = 42,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders using official event-level splits."""
    common_kwargs: Dict[str, Any] = {
        "data_root": data_cfg.data_root,
        "val_event_count": data_cfg.val_event_count,
        "crop_y": data_cfg.crop_y,
        "crop_x": data_cfg.crop_x,
        "region_height": data_cfg.region_height,
        "region_width": data_cfg.region_width,
        "window_t": data_cfg.window_t,
        "stride_t": data_cfg.stride_t,
        "wall_height": data_cfg.wall_height,
        "manning_n": data_cfg.manning_n,
        "rainfall_unit_scale": data_cfg.rainfall_unit_scale,
    }

    train_ds = UKEAPinoDataset(split="train", **common_kwargs)
    val_ds = UKEAPinoDataset(split="val", **common_kwargs)
    test_ds = UKEAPinoDataset(split="test", **common_kwargs)

    # Verify strict event-level disjointness across train/val/test
    if set(train_ds.event_ids) & set(test_ds.event_ids):
        raise AssertionError("Train and test event IDs must be disjoint.")
    if data_cfg.val_event_count > 0 and (set(train_ds.event_ids) & set(val_ds.event_ids)):
        raise AssertionError("Train and val event IDs must be disjoint.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs: Dict[str, Any] = {
        "batch_size": data_cfg.batch_size,
        "num_workers": data_cfg.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker if data_cfg.num_workers > 0 else None,
        "persistent_workers": bool(data_cfg.num_workers > 0),
    }

    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
