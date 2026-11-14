"""Validation & test evaluation script for saved PINO checkpoints on UKEA/LarNO splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crestix_pino.data.dataloader import build_dataloaders  # noqa: E402
from crestix_pino.evaluation.metrics import evaluate_predictions  # noqa: E402
from crestix_pino.inference.model_loader import load_model  # noqa: E402
from crestix_pino.models.pino import PINOFloodModel  # noqa: E402
from crestix_pino.training.config import load_config  # noqa: E402
from crestix_pino.training.train import _ensure_smoke_ukea_data  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a saved PINO checkpoint on validation or held-out test UKEA splits."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to PINO checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ukea_32x32.yaml"),
        help="Training/evaluation YAML config.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional override for UKEA dataset directory (defaults to config or smoke data fallback).",
    )
    parser.add_argument("--device", default="cpu", help="Target evaluation device.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON file for aggregate validation metrics.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.data.data_root = str(args.data_root)
    elif not (Path(cfg.data.data_root) / "geodata" / "dem.npy").exists():
        ckpt_dir = Path(args.checkpoint).resolve().parent.parent
        candidate = ckpt_dir / "synthetic_ukea_8m_5min"
        if (candidate / "geodata" / "dem.npy").exists():
            cfg.data.data_root = str(candidate)
        else:
            cfg.data.data_root = _ensure_smoke_ukea_data(ckpt_dir / "synthetic_ukea_8m_5min")

    _, val_loader, test_loader = build_dataloaders(cfg.data, seed=cfg.seed, pin_memory=False)
    loader = val_loader if args.split == "val" else test_loader

    ckpt_payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    loaded = load_model(args.checkpoint, device=args.device)

    all_preds = []
    all_targets = []
    all_rain_mm_h = []
    all_h0 = []
    dem_ref: Optional[np.ndarray] = None

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(args.device)  # (B, 4, T, H, W)
            y = batch["targets"].cpu().numpy()   # (B, T, H, W)

            if dem_ref is None:
                dem_ref = batch["dem"][0].cpu().numpy()

            b, _, t_dim, h_dim, w_dim = x.shape
            preds_b = np.empty((b, t_dim, h_dim, w_dim), dtype=np.float32)
            for idx in range(b):
                dem_i = batch["dem"][idx].cpu().numpy()
                rain_mm_h = batch["rainfall_m_s"][idx].cpu().numpy() * 3_600_000.0
                h0_i = batch["h_init"][idx].cpu().numpy()
                valid_i = batch["valid_mask"][idx].cpu().numpy().astype(bool)
                dem_masked = np.where(valid_i, dem_i, np.nan)
                preds_b[idx] = np.asarray(
                    loaded(dem_masked, rain_mm_h, h0_i, t_dim),
                    dtype=np.float32,
                )

            all_preds.append(preds_b)
            all_targets.append(y)
            all_rain_mm_h.append(batch["rainfall_m_s"].cpu().numpy() * 3_600_000.0)
            all_h0.append(batch["h_init"].cpu().numpy())

    elapsed_sec = time.perf_counter() - t0
    pred_arr = np.concatenate(all_preds, axis=0)
    target_arr = np.concatenate(all_targets, axis=0)
    rain_arr_mm_h = np.concatenate(all_rain_mm_h, axis=0)
    h0_arr = np.concatenate(all_h0, axis=0)

    metrics = evaluate_predictions(
        pred=pred_arr,
        target=target_arr,
        dem=dem_ref if dem_ref is not None else np.zeros(pred_arr.shape[-2:], dtype=np.float32),
        rainfall_mm_per_hr=rain_arr_mm_h,
        initial_water_depth=h0_arr,
        dt_seconds=float(cfg.physics.dt),
        cell_size_m=float(cfg.physics.dx),
        manning_n=float(cfg.physics.default_manning_n),
        wet_threshold=0.03,
    )

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt_payload.get("epoch", -1)) if isinstance(ckpt_payload, dict) else None,
        "split": args.split,
        "num_windows": int(pred_arr.shape[0]),
        "dt_seconds": float(cfg.physics.dt),
        "cell_size_m": float(cfg.physics.dx),
        "validation_time_sec": round(elapsed_sec, 4),
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
