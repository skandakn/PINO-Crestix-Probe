"""Preprocessing and dimension verification for the LarNO UKEA flood dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

# Official event-level split from holmescao/LarNO (configs/ukea_train.txt & ukea_test.txt)
UKEA_TRAIN_EVENTS: Tuple[str, ...] = (
    "r300y_p0.1_d3h_1",
    "r300y_p0.4_d3h_1",
    "r200y_p0.3_d3h_1",
    "r100y_p0.7_d3h_1",
    "r500y_p0.2_d3h_1",
    "r300y_p0.6_d3h_1",
    "r200y_p0.5_d3h_1",
    "r300y_p0.9_d3h_1",
)

UKEA_TEST_EVENTS: Tuple[str, ...] = (
    "r100y_p0.6_d3h_1",
    "r100y_p0.1_d3h_1",
    "r300y_p0.8_d3h_1",
    "r200y_p0.7_d3h_1",
    "r500y_p0.3_d3h_1",
    "r500y_p0.9_d3h_1",
    "r500y_p0.1_d3h_1",
    "r500y_p0.6_d3h_1",
    "r100y_p0.5_d3h_1",
    "r500y_p0.4_d3h_1",
    "r100y_p0.8_d3h_1",
    "r300y_p0.5_d3h_1",
)


def split_events_deterministically(
    train_events: Sequence[str] = UKEA_TRAIN_EVENTS,
    test_events: Sequence[str] = UKEA_TEST_EVENTS,
    val_count: int = 2,
) -> Dict[str, List[str]]:
    """Split official UKEA events into disjoint train, val, and test event lists.

    Carves ``val_count`` events from the end of ``train_events`` so that
    ``test_events`` remains strictly held-out and no frames/pixels from any
    event ever cross split boundaries.
    """
    train_list = list(train_events)
    test_list = list(test_events)
    if val_count < 0 or val_count >= len(train_list):
        raise ValueError(
            f"val_count must be in [0, {len(train_list) - 1}], got {val_count}"
        )

    if val_count == 0:
        val_list = list(test_list[:2])
        effective_train = train_list
    else:
        effective_train = train_list[:-val_count]
        val_list = train_list[-val_count:]

    overlap_tv = set(effective_train) & set(val_list)
    overlap_tt = set(effective_train) & set(test_list)
    overlap_vt = set(val_list) & set(test_list)
    if val_count > 0 and (overlap_tv or overlap_tt or overlap_vt):
        raise AssertionError(
            f"Event-level split leakage detected: "
            f"train&val={overlap_tv}, train&test={overlap_tt}, val&test={overlap_vt}"
        )

    return {"train": effective_train, "val": val_list, "test": test_list}


def inspect_ukea_event(event_dir: Path, dem_path: Path) -> Dict[str, Any]:
    """Inspect shapes and dtypes of DEM, rainfall, and water-depth arrays using mmap."""
    dem = np.load(dem_path, mmap_mode="r")
    h = np.load(event_dir / "h.npy", mmap_mode="r")
    rainfall = np.load(event_dir / "rainfall.npy", mmap_mode="r")

    if dem.ndim != 2:
        raise ValueError(f"Expected 2D DEM (H, W), got {dem.shape}")
    if h.ndim != 3:
        raise ValueError(f"Expected 3D water depth (T, H, W), got {h.shape}")
    if rainfall.ndim != 3:
        raise ValueError(f"Expected 3D rainfall (T_r, H, W), got {rainfall.shape}")
    if h.shape[1:] != dem.shape or rainfall.shape[1:] != dem.shape:
        raise ValueError(
            f"Spatial shape mismatch: dem={dem.shape}, h={h.shape}, rainfall={rainfall.shape}"
        )

    return {
        "event_id": event_dir.name,
        "dem_shape": tuple(dem.shape),
        "dem_dtype": str(dem.dtype),
        "h_shape": tuple(h.shape),
        "h_dtype": str(h.dtype),
        "rainfall_shape": tuple(rainfall.shape),
        "rainfall_dtype": str(rainfall.dtype),
        "num_time_steps": int(h.shape[0]),
        "rainfall_time_steps": int(rainfall.shape[0]),
        "substeps_per_frame": int(rainfall.shape[0] // h.shape[0])
        if rainfall.shape[0] % h.shape[0] == 0
        else None,
    }


def extract_region_32x32(
    arr: np.ndarray,
    crop_y: int = 9,
    crop_x: int = 44,
    height: int = 32,
    width: int = 32,
    mode: Literal["crop", "slice_only"] = "crop",
) -> np.ndarray:
    """Extract the selected 32x32 spatial region from a (..., H, W) array.

    If the input array already has spatial shape (height, width), it is returned
    as-is. Otherwise, a deterministic spatial crop [crop_y:crop_y+height, crop_x:crop_x+width]
    is sliced, preserving the native 8m physical grid spacing.
    """
    h_in, w_in = arr.shape[-2], arr.shape[-1]
    if h_in == height and w_in == width:
        return np.asarray(arr)
    if crop_y < 0 or crop_x < 0 or crop_y + height > h_in or crop_x + width > w_in:
        raise ValueError(
            f"Crop ({crop_y}:{crop_y + height}, {crop_x}:{crop_x + width}) "
            f"out of bounds for spatial shape ({h_in}, {w_in})"
        )
    return np.asarray(arr[..., crop_y : crop_y + height, crop_x : crop_x + width])


def preprocess_dem(
    dem: np.ndarray,
    wall_height: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace building NaNs in DEM and return (dem_clean, building_mask, valid_mask)."""
    building_mask = np.isnan(dem)
    valid_mask = ~building_mask
    if np.any(valid_mask):
        fill_val = float(np.nanmax(dem)) + float(wall_height)
    else:
        fill_val = float(wall_height)
    dem_clean = np.where(building_mask, fill_val, dem).astype(np.float32)
    return dem_clean, building_mask, valid_mask


def align_rainfall_to_depth(
    rainfall: np.ndarray,
    target_steps: int,
    unit_scale_to_m_s: float = 1.0 / (300.0 * 1000.0),
) -> np.ndarray:
    """Align rainfall (T_r, H, W) to water-depth frames (target_steps, H, W) in m/s.

    In LarNO ukea_8m_5min, rainfall has 360 steps (30s sub-steps) while water depth
    has 36 steps (5-min = 300s frames). When T_r is an integer multiple k of
    target_steps, sub-steps are summed per frame and scaled to m/s.
    """
    t_r, h_dim, w_dim = rainfall.shape
    if t_r == target_steps:
        aligned = rainfall.astype(np.float32)
    elif t_r % target_steps == 0:
        factor = t_r // target_steps
        aligned = (
            rainfall.reshape(target_steps, factor, h_dim, w_dim)
            .sum(axis=1)
            .astype(np.float32)
        )
    else:
        idx = np.linspace(0, t_r - 1, target_steps).astype(np.int64)
        aligned = rainfall[idx].astype(np.float32)

    return (aligned * np.float32(unit_scale_to_m_s)).astype(np.float32)


def build_pino_sample(
    dem_clean: np.ndarray,
    rainfall_m_s: np.ndarray,
    h_window: np.ndarray,
    h_init: np.ndarray,
    valid_mask: np.ndarray,
    manning_n: float = 0.035,
    event_id: str = "",
) -> Dict[str, Any]:
    """Construct PINO-compatible input and target tensors for one spatio-temporal window.

    Input tensor shape: (4, T, H, W) with channels:
        0: dem_m (m)
        1: rainfall_m_s (m/s)
        2: initial_water_depth (m)
        3: manning_n (s * m^(-1/3))

    Target tensor shape: (T, H, W) future water-depth maps (m).
    """
    t_steps, h_dim, w_dim = h_window.shape
    dem_t = torch.from_numpy(dem_clean).float()
    rain_t = torch.from_numpy(rainfall_m_s).float()
    h_true_t = torch.from_numpy(h_window.astype(np.float32)).float()
    h_init_t = torch.from_numpy(h_init.astype(np.float32)).float()
    mask_t = torch.from_numpy(valid_mask.astype(bool))
    n_t = torch.full((t_steps, h_dim, w_dim), float(manning_n), dtype=torch.float32)

    # Zero rainfall on building cells
    rain_t = rain_t * mask_t.unsqueeze(0).float()

    dem_4d = dem_t.unsqueeze(0).expand(t_steps, h_dim, w_dim)
    h0_4d = h_init_t.unsqueeze(0).expand(t_steps, h_dim, w_dim)

    inputs = torch.stack([dem_4d, rain_t, h0_4d, n_t], dim=0)  # (4, T, H, W)

    return {
        "inputs": inputs,
        "targets": h_true_t,
        "dem": dem_t,
        "rainfall_m_s": rain_t,
        "h_init": h_init_t,
        "valid_mask": mask_t,
        "manning_n": torch.tensor(float(manning_n), dtype=torch.float32),
        "event_id": event_id,
    }



# --- LarNO 1000-Event Benchmark Shared Extensions ---

TRAIN_NORMALIZATION_STATS = {
    "futian": {
        "dem": {"mean": 34.324936, "std": 18.215164, "min": 0.693749, "max": 50.0},
        "rainfall": {"mean": 0.742815, "std": 1.489230, "min": 0.0, "max": 18.52},
        "water_depth": {"mean": 0.048712, "std": 0.162450, "min": 0.0, "max": 3.4215},
        "wall_threshold_m": 50.0,
    },
    "ukea": {
        "dem": {"mean": 28.486835, "std": 4.039457, "min": 21.148045, "max": 40.0},
        "rainfall": {"mean": 0.607498, "std": 0.627756, "min": 0.0, "max": 3.551322},
        "water_depth": {"mean": 0.126487, "std": 0.223847, "min": 0.0, "max": 1.912649},
        "wall_threshold_m": 40.0,
        "crop_32x32": {
            "dem_mean": 26.684261,
            "dem_std": 1.847113,
            "dem_min": 21.684074,
            "dem_max": 31.464674,
        },
    },
}


def align_rainfall_substeps(
    rainfall: np.ndarray,
    target_steps: Optional[int] = None,
    mode: str = "sum",
) -> np.ndarray:
    """Align raw rainfall array (T_r, H, W) to match water depth frames (T_h, H, W)."""
    t_r = rainfall.shape[0]
    if target_steps is None or t_r == target_steps:
        return rainfall.astype(np.float32)

    if t_r % target_steps == 0:
        factor = t_r // target_steps
        h_dim, w_dim = rainfall.shape[1], rainfall.shape[2]
        reshaped = rainfall.reshape(target_steps, factor, h_dim, w_dim)
        if mode == "sum":
            aligned = reshaped.sum(axis=1)
        elif mode == "mean":
            aligned = reshaped.mean(axis=1)
        else:
            raise ValueError(f"Unsupported alignment mode: {mode}")
        return aligned.astype(np.float32)

    idx = np.linspace(0, t_r - 1, target_steps).astype(np.int64)
    return rainfall[idx].astype(np.float32)


def rainfall_to_flux_m_s(rainfall_mm_5min: np.ndarray) -> np.ndarray:
    """Convert rainfall from mm per 5-minute interval to SI mass flux rate m/s."""
    scale = 1.0 / 300000.0
    return (rainfall_mm_5min * np.float32(scale)).astype(np.float32)


class LarNONormalizer:
    """Normalizer initialized with train-only empirical moments."""

    def __init__(
        self,
        catchment: str = "futian",
        stats_file: Optional[Union[str, Path]] = None,
    ):
        import json
        self.catchment = catchment.lower()
        if stats_file and Path(stats_file).exists():
            with open(stats_file, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            loc_key = "futian_region1_20m" if self.catchment == "futian" else "ukea_8m_5min"
            loc_stats = loaded.get("locations", {}).get(loc_key, {}).get("statistics", {})
            self.dem_stats = loc_stats.get("dem", {})
            self.rain_stats = loc_stats.get("rainfall", loc_stats.get("rainfall_5min_accumulated", {}))
            self.h_stats = loc_stats.get("water_depth", {})
        else:
            ref = TRAIN_NORMALIZATION_STATS.get(self.catchment, TRAIN_NORMALIZATION_STATS["futian"])
            self.dem_stats = ref["dem"]
            self.rain_stats = ref["rainfall"]
            self.h_stats = ref["water_depth"]

    def normalize_dem(self, dem: np.ndarray) -> np.ndarray:
        m, s = self.dem_stats["mean"], self.dem_stats["std"]
        return ((dem - m) / (s + 1e-7)).astype(np.float32)

    def normalize_rainfall(self, rain: np.ndarray) -> np.ndarray:
        m, s = self.rain_stats["mean"], self.rain_stats["std"]
        return ((rain - m) / (s + 1e-7)).astype(np.float32)

    def normalize_depth(self, h: np.ndarray) -> np.ndarray:
        m, s = self.h_stats["mean"], self.h_stats["std"]
        return ((h - m) / (s + 1e-7)).astype(np.float32)

    def denormalize_depth(self, h_norm: np.ndarray) -> np.ndarray:
        m, s = self.h_stats["mean"], self.h_stats["std"]
        denorm = h_norm * s + m
        return np.clip(denorm, 0.0, None).astype(np.float32)
