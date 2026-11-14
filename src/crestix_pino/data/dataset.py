"""Memory-efficient PyTorch Dataset for the LarNO UKEA 32x32 flood benchmark."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import urllib.request

import numpy as np
from torch.utils.data import Dataset

from .preprocessing import (
    UKEA_TEST_EVENTS,
    UKEA_TRAIN_EVENTS,
    align_rainfall_to_depth,
    build_pino_sample,
    extract_region_32x32,
    inspect_ukea_event,
    preprocess_dem,
    split_events_deterministically,
)

logger = logging.getLogger(__name__)

HF_DATASET_BASE_URL = "https://huggingface.co/datasets/holmescao/LarNO-dataset/resolve/main"


def download_ukea_dataset(
    data_root: str | Path,
    events: Optional[Sequence[str]] = None,
    timeout: int = 60,
) -> Path:
    """Download LarNO ukea_8m_5min DEM and event arrays from Hugging Face if missing."""
    root = Path(data_root)
    geo_dir = root / "geodata"
    flood_dir = root / "flood"
    geo_dir.mkdir(parents=True, exist_ok=True)
    flood_dir.mkdir(parents=True, exist_ok=True)

    dem_path = geo_dir / "dem.npy"
    if not dem_path.exists():
        url = f"{HF_DATASET_BASE_URL}/geodata/ukea_8m_5min/dem.npy"
        logger.info("Downloading UKEA DEM from %s", url)
        urllib.request.urlretrieve(url, dem_path)

    target_events = list(events) if events is not None else list(UKEA_TRAIN_EVENTS) + list(UKEA_TEST_EVENTS)
    for event_id in target_events:
        ev_dir = flood_dir / event_id
        ev_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("h.npy", "rainfall.npy"):
            dst = ev_dir / fname
            if not dst.exists():
                url = f"{HF_DATASET_BASE_URL}/flood/ukea_8m_5min/{event_id}/{fname}"
                logger.info("Downloading %s/%s", event_id, fname)
                urllib.request.urlretrieve(url, dst)

    return root


class UKEAPinoDataset(Dataset):
    """Memory-mapped PyTorch Dataset for UKEA 32x32 spatio-temporal flood forecasting.

    Arrays are opened with ``np.load(..., mmap_mode="r")`` on demand so only the
    requested spatial 32x32 region and temporal window are materialized in RAM.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        event_ids: Optional[Sequence[str]] = None,
        val_event_count: int = 2,
        crop_y: int = 9,
        crop_x: int = 44,
        region_height: int = 32,
        region_width: int = 32,
        window_t: Optional[int] = 12,
        stride_t: int = 4,
        wall_height: float = 50.0,
        manning_n: float = 0.035,
        rainfall_unit_scale: float = 1.0 / (300.0 * 1000.0),
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.crop_y = int(crop_y)
        self.crop_x = int(crop_x)
        self.region_height = int(region_height)
        self.region_width = int(region_width)
        self.window_t = window_t
        self.stride_t = int(stride_t)
        self.wall_height = float(wall_height)
        self.manning_n = float(manning_n)
        self.rainfall_unit_scale = float(rainfall_unit_scale)

        if event_ids is not None:
            self.event_ids = list(event_ids)
        else:
            splits = split_events_deterministically(val_count=val_event_count)
            if split not in splits:
                raise ValueError(f"split must be one of {list(splits.keys())}, got {split!r}")
            self.event_ids = splits[split]

        self.dem_path = self._resolve_dem_path()
        raw_dem = np.load(self.dem_path, mmap_mode="r")
        dem_crop = extract_region_32x32(
            raw_dem,
            crop_y=self.crop_y,
            crop_x=self.crop_x,
            height=self.region_height,
            width=self.region_width,
        )
        self.dem_clean, self.building_mask, self.valid_mask = preprocess_dem(
            dem_crop, wall_height=self.wall_height
        )

        self.event_metadata: List[Dict[str, Any]] = []
        self.samples: List[Tuple[int, int, int]] = []  # (event_idx, t_start, t_len)
        self._build_index()

    def _resolve_dem_path(self) -> Path:
        candidates = [
            self.data_root / "geodata" / "dem.npy",
            self.data_root / "geodata" / "ukea_8m_5min" / "dem.npy",
            self.data_root / "dem.npy",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Could not find dem.npy under {self.data_root}")

    def _resolve_event_dir(self, event_id: str) -> Path:
        candidates = [
            self.data_root / "flood" / event_id,
            self.data_root / "flood" / "ukea_8m_5min" / event_id,
            self.data_root / event_id,
        ]
        for cand in candidates:
            if (cand / "h.npy").exists() and (cand / "rainfall.npy").exists():
                return cand
        raise FileNotFoundError(
            f"Could not find h.npy and rainfall.npy for event {event_id!r} under {self.data_root}"
        )

    def _build_index(self) -> None:
        for ev_idx, event_id in enumerate(self.event_ids):
            ev_dir = self._resolve_event_dir(event_id)
            meta = inspect_ukea_event(ev_dir, self.dem_path)
            meta["event_dir"] = str(ev_dir)
            self.event_metadata.append(meta)

            total_t = meta["num_time_steps"]
            if self.window_t is None or self.window_t >= total_t:
                self.samples.append((ev_idx, 0, total_t))
            else:
                if self.window_t < 2:
                    raise ValueError(f"window_t must be >= 2, got {self.window_t}")
                starts = list(range(0, total_t - self.window_t + 1, self.stride_t))
                if starts[-1] != total_t - self.window_t:
                    starts.append(total_t - self.window_t)
                for t0 in starts:
                    self.samples.append((ev_idx, t0, self.window_t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ev_idx, t0, t_len = self.samples[index]
        meta = self.event_metadata[ev_idx]
        ev_dir = Path(meta["event_dir"])

        h_mmap = np.load(ev_dir / "h.npy", mmap_mode="r")
        rain_mmap = np.load(ev_dir / "rainfall.npy", mmap_mode="r")

        y0, y1 = self.crop_y, self.crop_y + self.region_height
        x0, x1 = self.crop_x, self.crop_x + self.region_width

        # Slice only the required temporal window and 32x32 spatial patch
        if h_mmap.shape[1] == self.region_height and h_mmap.shape[2] == self.region_width:
            h_slice = np.array(h_mmap[t0 : t0 + t_len], copy=True)
        else:
            h_slice = np.array(h_mmap[t0 : t0 + t_len, y0:y1, x0:x1], copy=True)

        total_h_steps = meta["num_time_steps"]
        total_r_steps = meta["rainfall_time_steps"]
        if total_r_steps % total_h_steps == 0:
            factor = total_r_steps // total_h_steps
            r_start, r_end = t0 * factor, (t0 + t_len) * factor
        else:
            r_start = int(t0 * total_r_steps / total_h_steps)
            r_end = int((t0 + t_len) * total_r_steps / total_h_steps)

        if rain_mmap.shape[1] == self.region_height and rain_mmap.shape[2] == self.region_width:
            rain_slice = np.array(rain_mmap[r_start:r_end], copy=True)
        else:
            rain_slice = np.array(rain_mmap[r_start:r_end, y0:y1, x0:x1], copy=True)

        rainfall_m_s = align_rainfall_to_depth(
            rain_slice,
            target_steps=t_len,
            unit_scale_to_m_s=self.rainfall_unit_scale,
        )
        h_init = h_slice[0]

        sample = build_pino_sample(
            dem_clean=self.dem_clean,
            rainfall_m_s=rainfall_m_s,
            h_window=h_slice,
            h_init=h_init,
            valid_mask=self.valid_mask,
            manning_n=self.manning_n,
            event_id=meta["event_id"],
        )
        sample["t_start"] = t0
        sample["t_len"] = t_len
        return sample
