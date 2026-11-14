"""Sample inference and checkpoint reproduction CLI for PINO-Crestix-Probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from crestix_pino.inference.model_loader import load_model  # noqa: E402
from crestix_pino.inference.predictor import predict  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standardized Crestix Probe PINO inference and verify checkpoint reproduction."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to PINO checkpoint (.pt/.pth/.pts).",
    )
    parser.add_argument("--height", type=int, default=32, help="Spatial grid height H.")
    parser.add_argument("--width", type=int, default=32, help="Spatial grid width W.")
    parser.add_argument("--steps", type=int, default=12, help="Forecast horizon steps T.")
    parser.add_argument("--device", default="cpu", help="Target inference device (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic synthetic input seed.")
    parser.add_argument(
        "--clamp-negative",
        action="store_true",
        help="Clamp any negative depth outputs to 0.0 m during postprocessing.",
    )
    parser.add_argument(
        "--verify-reproduction",
        action="store_true",
        help="Load checkpoint independently twice and assert identical water_depth predictions.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path to record inference summary and reproduction check.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, args.height, dtype=np.float32),
        np.linspace(0.0, 1.0, args.width, dtype=np.float32),
        indexing="ij",
    )
    dem = (25.0 + 3.0 * (yy - 0.5) ** 2 + 2.0 * (xx - 0.5) ** 2 + rng.normal(0.0, 0.02, size=(args.height, args.width))).astype(np.float32)
    dem[:2, :2] = np.nan  # building / invalid DEM cells
    h0 = np.zeros((args.height, args.width), dtype=np.float32)
    rainfall_mm_h = np.full((args.steps,), 42.0, dtype=np.float32)

    res1 = predict(
        dem=dem,
        rainfall=rainfall_mm_h,
        initial_water_depth=h0,
        forecast_steps=args.steps,
        model=args.checkpoint,
        device=args.device,
        clamp_negative=args.clamp_negative,
        allow_dem_nan=True,
    )

    reproduced = True
    max_abs_diff = 0.0
    if args.verify_reproduction:
        reloaded = load_model(args.checkpoint, device=args.device)
        res2 = predict(
            dem=dem,
            rainfall=rainfall_mm_h,
            initial_water_depth=h0,
            forecast_steps=args.steps,
            model=reloaded,
            device=args.device,
            clamp_negative=args.clamp_negative,
            allow_dem_nan=True,
        )
        max_abs_diff = float(np.max(np.abs(res1["water_depth"] - res2["water_depth"])))
        reproduced = max_abs_diff == 0.0
        if not reproduced:
            raise AssertionError(
                f"Checkpoint reproduction check failed: max_abs_diff={max_abs_diff}"
            )

    depths = res1["water_depth"]
    summary = {
        "checkpoint": str(args.checkpoint),
        "framework": res1["metadata"]["framework"],
        "input_dem_shape": list(dem.shape),
        "output_water_depth_shape": list(depths.shape),
        "units": res1["metadata"]["units"],
        "dt_seconds": res1["metadata"]["dt_seconds"],
        "cell_size_m": res1["metadata"]["cell_size_m"],
        "min_depth_m": float(np.min(depths)),
        "mean_depth_m": float(np.mean(depths)),
        "max_depth_m": float(np.max(depths)),
        "nan_cell_zeroed": bool(np.all(depths[:, :2, :2] == 0.0)),
        "raw_negative_cell_count": res1["metadata"]["raw_negative_cell_count"],
        "reproduction_verified": reproduced if args.verify_reproduction else None,
        "reproduction_max_abs_diff_m": max_abs_diff if args.verify_reproduction else None,
    }
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
