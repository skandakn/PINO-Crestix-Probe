"""Memory-efficient, multi-catchment LarNO PyTorch Dataset."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore[no-redef]
        pass

from .preprocessing import (
    LarNONormalizer,
    align_rainfall_substeps,
    preprocess_dem,
    rainfall_to_flux_m_s,
)

logger = logging.getLogger(__name__)


def _to_tensor_or_numpy(arr: np.ndarray, dtype_torch=None):
    if HAS_TORCH:
        t = torch.from_numpy(arr)
        if dtype_torch is not None:
            t = t.to(dtype=dtype_torch)
        return t
    return arr


class LarNOUnifiedDataset(Dataset):
    """Universal Dataset for LarNO (Futian & UKEA) spatiotemporal flood forecasting.

    Supports:
      1. Memory-mapped array loading for negligible RAM footprint on multi-GB datasets.
      2. Dynamic window slicing for recurrent models (ConvLSTM U-Net) and neural operators (PINO).
      3. Strict event-level split filtering (train/val/test) avoiding any cross-event data leakage.
      4. Preserves native spatial resolutions (20m Futian vs 8m UKEA).
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: Literal["train", "val", "test"] = "train",
        catchment: Literal["Futian", "UKEA", "all"] = "all",
        manifest_path: Optional[Union[str, Path]] = None,
        split_manifest_path: Optional[Union[str, Path]] = None,
        format_mode: Literal["convlstm", "pino", "raw"] = "convlstm",
        window_in: int = 6,
        window_out: int = 6,
        stride_t: int = 2,
        stats_file: Optional[Union[str, Path]] = None,
        manning_n: float = 0.035,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.catchment = catchment
        self.format_mode = format_mode
        self.window_in = window_in
        self.window_out = window_out
        self.stride_t = stride_t
        self.manning_n = manning_n

        self.normalizers = {
            "futian": LarNONormalizer("futian", stats_file),
            "ukea": LarNONormalizer("ukea", stats_file),
        }

        # 1. Load manifest and filter events
        self.event_records = self._load_manifest(manifest_path, split_manifest_path)
        logger.info(
            "Initialized LarNOUnifiedDataset: %d events (split=%s, catchment=%s, format=%s)",
            len(self.event_records),
            split,
            catchment,
            format_mode,
        )

        # 2. Cache DEMs and building masks per catchment
        self.dem_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}
        self._load_dems()

        # 3. Build sliding-window index
        self.samples: List[Tuple[int, int]] = []  # (event_idx, t_start)
        self._build_index()

    def _load_manifest(
        self,
        manifest_path: Optional[Union[str, Path]],
        split_manifest_path: Optional[Union[str, Path]],
    ) -> List[Dict[str, Any]]:
        candidates = [
            manifest_path,
            self.data_root / "event_manifest.csv",
            self.data_root / "data" / "event_manifest.csv",
            Path("data/event_manifest.csv"),
            Path("larno_pipeline_outputs/event_manifest.csv"),
        ]
        chosen = None
        for cand in candidates:
            if cand and Path(cand).exists():
                chosen = Path(cand)
                break

        if not chosen:
            raise FileNotFoundError("Could not locate event_manifest.csv. Please pass manifest_path.")

        with open(chosen, "r", encoding="utf-8") as fp:
            records = list(csv.DictReader(fp))

        filtered = []
        for r in records:
            if r["split"] != self.split:
                continue
            if self.catchment != "all" and r["catchment"].lower() != self.catchment.lower():
                continue
            filtered.append(r)

        return filtered

    def _load_dems(self) -> None:
        for cat in ("Futian", "UKEA"):
            reg_tag = "region1_20m" if cat == "Futian" else "ukea_8m_5min"
            dem_candidates = [
                self.data_root / "geodata" / reg_tag / "dem.npy",
                self.data_root / "geodata" / "dem.npy",
                self.data_root / f"dem_{cat.lower()}.npy",
            ]
            dem_path = None
            for cand in dem_candidates:
                if cand.exists():
                    dem_path = cand
                    break

            if dem_path and dem_path.exists():
                raw_dem = np.load(dem_path, mmap_mode="r")
                threshold = 50.0 if cat == "Futian" else 40.0
                dem_clean, bmask, vmask = preprocess_dem(raw_dem, wall_threshold=threshold)
                self.dem_cache[cat.lower()] = {
                    "dem_clean": dem_clean,
                    "building_mask": bmask,
                    "valid_mask": vmask,
                }
            else:
                self.dem_cache[cat.lower()] = None

    def _resolve_event_arrays(self, ev_meta: Dict[str, Any]) -> Tuple[Path, Path, Optional[Path]]:
        ev_id = ev_meta["event_id"]
        reg_tag = ev_meta["region_tag"]

        dir_candidates = [
            self.data_root / "flood" / reg_tag / ev_id,
            self.data_root / "flood" / ev_id,
            self.data_root / ev_id,
        ]
        ev_dir = None
        for cand in dir_candidates:
            if (cand / "h.npy").exists() and (cand / "rainfall.npy").exists():
                ev_dir = cand
                break

        if not ev_dir:
            raise FileNotFoundError(f"Missing h.npy and rainfall.npy for event '{ev_id}' under {self.data_root}")

        dem_file = ev_dir / "dem.npy"
        return ev_dir / "h.npy", ev_dir / "rainfall.npy", dem_file if dem_file.exists() else None

    def _build_index(self) -> None:
        window_total = self.window_in + self.window_out
        for ev_idx, ev in enumerate(self.event_records):
            h_steps = int(ev["water_depth_steps"])
            if h_steps < window_total:
                continue
            starts = list(range(0, h_steps - window_total + 1, self.stride_t))
            for t0 in starts:
                self.samples.append((ev_idx, t0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ev_idx, t0 = self.samples[index]
        ev_meta = self.event_records[ev_idx]
        cat = ev_meta["catchment"].lower()
        normalizer = self.normalizers[cat]

        h_path, r_path, dem_path = self._resolve_event_arrays(ev_meta)
        h_mmap = np.load(h_path, mmap_mode="r")
        r_mmap = np.load(r_path, mmap_mode="r")

        # Resolve DEM
        if self.dem_cache.get(cat) is not None:
            dem_clean = self.dem_cache[cat]["dem_clean"]
            building_mask = self.dem_cache[cat]["building_mask"]
            valid_mask = self.dem_cache[cat]["valid_mask"]
        elif dem_path and dem_path.exists():
            raw_dem = np.load(dem_path, mmap_mode="r")
            dem_clean, building_mask, valid_mask = preprocess_dem(raw_dem)
        else:
            raise RuntimeError(f"DEM not available for event {ev_meta['event_id']}")

        window_total = self.window_in + self.window_out
        h_seq = np.array(h_mmap[t0 : t0 + window_total], copy=True, dtype=np.float32)

        # Align rainfall to water depth frames
        if cat == "ukea" and r_mmap.shape[0] == 360:
            substep_factor = 10
            r_start, r_end = t0 * substep_factor, (t0 + window_total) * substep_factor
            r_sub = np.array(r_mmap[r_start:r_end], copy=True, dtype=np.float32)
            r_seq = align_rainfall_substeps(r_sub, target_steps=window_total, mode="sum")
        else:
            r_seq = np.array(r_mmap[t0 : t0 + window_total], copy=True, dtype=np.float32)

        # Zero out rainfall over building footprints
        r_seq = r_seq * valid_mask[np.newaxis, ...].astype(np.float32)

        if self.format_mode == "convlstm":
            r_in = r_seq[: self.window_in]
            r_cumsum = np.cumsum(r_in, axis=0)
            h_in_state = h_seq[self.window_in - 1]
            h_out = h_seq[self.window_in : window_total]

            r_norm = normalizer.normalize_rainfall(r_in)
            r_cs_norm = normalizer.normalize_rainfall(r_cumsum)
            dem_norm = normalizer.normalize_dem(dem_clean)

            t_in, h_dim, w_dim = r_in.shape
            dem_exp = np.broadcast_to(dem_norm[np.newaxis, np.newaxis, ...], (t_in, 1, h_dim, w_dim))
            h_exp = np.broadcast_to(h_in_state[np.newaxis, np.newaxis, ...], (t_in, 1, h_dim, w_dim))
            bmask_exp = np.broadcast_to(building_mask[np.newaxis, np.newaxis, ...].astype(np.float32), (t_in, 1, h_dim, w_dim))

            r_exp = r_norm[:, np.newaxis, ...]
            rcs_exp = r_cs_norm[:, np.newaxis, ...]

            x_tensor = np.concatenate([r_exp, rcs_exp, dem_exp, h_exp, bmask_exp], axis=1)  # (T_in, 5, H, W)
            y_tensor = h_out[:, np.newaxis, ...]  # (T_out, 1, H, W)

            return {
                "inputs": _to_tensor_or_numpy(x_tensor),
                "targets": _to_tensor_or_numpy(y_tensor),
                "event_id": ev_meta["event_id"],
                "catchment": ev_meta["catchment"],
                "t_start": t0,
                "valid_mask": _to_tensor_or_numpy(valid_mask),
            }

        elif self.format_mode == "pino":
            r_m_s = rainfall_to_flux_m_s(r_seq)
            h_init = h_seq[0]
            t_len, h_dim, w_dim = r_seq.shape

            dem_exp = np.broadcast_to(dem_clean[np.newaxis, ...], (t_len, h_dim, w_dim))
            h0_exp = np.broadcast_to(h_init[np.newaxis, ...], (t_len, h_dim, w_dim))
            n_exp = np.full((t_len, h_dim, w_dim), self.manning_n, dtype=np.float32)

            inputs = np.stack([dem_exp, r_m_s, h0_exp, n_exp], axis=0)  # (4, T, H, W)
            targets = h_seq[np.newaxis, ...]  # (1, T, H, W)

            return {
                "inputs": _to_tensor_or_numpy(inputs),
                "targets": _to_tensor_or_numpy(targets),
                "dem": _to_tensor_or_numpy(dem_clean),
                "rainfall_m_s": _to_tensor_or_numpy(r_m_s),
                "h_init": _to_tensor_or_numpy(h_init),
                "valid_mask": _to_tensor_or_numpy(valid_mask),
                "manning_n": self.manning_n,
                "event_id": ev_meta["event_id"],
                "catchment": ev_meta["catchment"],
                "t_start": t0,
            }

        else:
            return {
                "dem": _to_tensor_or_numpy(dem_clean),
                "rainfall": _to_tensor_or_numpy(r_seq),
                "water_depth": _to_tensor_or_numpy(h_seq),
                "valid_mask": _to_tensor_or_numpy(valid_mask),
                "event_id": ev_meta["event_id"],
                "catchment": ev_meta["catchment"],
            }
