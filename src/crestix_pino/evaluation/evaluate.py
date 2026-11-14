"""Evaluation runner for held-out flood events against LarNO ground truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from crestix_pino.evaluation.metrics import evaluate_predictions
from crestix_pino.evaluation.visualization import (
    plot_horizon_metrics,
    plot_spatial_error_map,
    plot_time_series_comparison,
)
from crestix_pino.inference.predictor import PINOPredictor


def evaluate_event(
    predictor: PINOPredictor,
    dem: Any,
    rainfall: Any,
    initial_water_depth: Any,
    target_water_depth: Any,
    *,
    dt_seconds: float | None = None,
    cell_size_m: float | None = None,
    manning_n: float = 0.035,
    output_dir: str | Path | None = None,
    event_name: str = "event",
) -> dict[str, Any]:
    """Run inference on a single held-out flood event and compute all metrics."""
    target_arr = np.asarray(target_water_depth, dtype=np.float64)
    forecast_steps = target_arr.shape[-3]

    dt = dt_seconds if dt_seconds is not None else predictor.contract.dt_seconds
    cell = cell_size_m if cell_size_m is not None else predictor.contract.cell_size_m

    prediction = predictor.predict(
        dem=dem,
        rainfall=rainfall,
        initial_water_depth=initial_water_depth,
        forecast_steps=forecast_steps,
    )
    pred_depth = prediction["water_depth"]

    metrics = evaluate_predictions(
        pred=pred_depth,
        target=target_arr,
        dem=dem,
        rainfall_mm_per_hr=rainfall,
        initial_water_depth=initial_water_depth,
        dt_seconds=dt,
        cell_size_m=cell,
        manning_n=manning_n,
    )
    result = {
        "event_name": event_name,
        "metrics": metrics,
        "inference_metadata": prediction["metadata"],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / f"{event_name}_predictions.npz",
            predicted_water_depth=pred_depth,
            larno_water_depth=target_arr,
        )
        plot_spatial_error_map(
            pred_depth,
            target_arr,
            out / f"{event_name}_spatial_error.png",
            step_idx=-1,
            title_prefix=f"PINO vs LarNO ({event_name})",
        )
        plot_time_series_comparison(
            pred_depth,
            target_arr,
            out / f"{event_name}_time_series.png",
            dt_seconds=dt,
            cell_size_m=cell,
            initial_water_depth=initial_water_depth,
            rainfall_mm_per_hr=rainfall,
        )
        plot_horizon_metrics(
            metrics["horizon_metrics"],
            out / f"{event_name}_horizon_metrics.png",
        )
        with (out / f"{event_name}_metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

    return result


def _load_event_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        target_key = next(
            (k for k in ("larno_water_depth", "water_depth", "target") if k in keys),
            None,
        )
        if target_key is None:
            raise KeyError(
                f"{path} must contain 'larno_water_depth' (or 'water_depth' / 'target'). "
                f"Found keys: {sorted(keys)}"
            )
        required = ("dem", "rainfall", "initial_water_depth")
        missing = [k for k in required if k not in keys]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")

        return {
            "dem": data["dem"],
            "rainfall": data["rainfall"],
            "initial_water_depth": data["initial_water_depth"],
            "target_water_depth": data[target_key],
            "dt_seconds": float(data["dt_seconds"]) if "dt_seconds" in keys else None,
            "cell_size_m": float(data["cell_size_m"]) if "cell_size_m" in keys else None,
        }


def evaluate_dataset(
    predictor: PINOPredictor,
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
    manning_n: float = 0.035,
) -> dict[str, Any]:
    """Evaluate a trained PINO predictor over a .npz file or directory of .npz flood events."""
    src = Path(dataset_path)
    if not src.exists():
        raise FileNotFoundError(f"Evaluation dataset path not found: {src}")

    event_files = [src] if src.is_file() else sorted(src.glob("*.npz"))
    if not event_files:
        raise ValueError(f"No .npz flood event files found in {src}")

    events: list[dict[str, Any]] = []
    for event_file in event_files:
        payload = _load_event_npz(event_file)
        event_out = Path(output_dir) / event_file.stem if output_dir else None
        res = evaluate_event(
            predictor=predictor,
            dem=payload["dem"],
            rainfall=payload["rainfall"],
            initial_water_depth=payload["initial_water_depth"],
            target_water_depth=payload["target_water_depth"],
            dt_seconds=payload["dt_seconds"],
            cell_size_m=payload["cell_size_m"],
            manning_n=manning_n,
            output_dir=event_out,
            event_name=event_file.stem,
        )
        events.append(res)

    summary = {
        "num_events": len(events),
        "aggregate_metrics": {
            "mean_mae_m": float(np.mean([e["metrics"]["mae_m"] for e in events])),
            "mean_rmse_m": float(np.mean([e["metrics"]["rmse_m"] for e in events])),
            "mean_r2": float(np.mean([e["metrics"]["r2"] for e in events])),
            "mean_model_latency_ms": float(
                np.mean([e["inference_metadata"]["latency_ms"]["model_only"] for e in events])
            ),
            "mean_total_pipeline_latency_ms": float(
                np.mean([e["inference_metadata"]["latency_ms"]["total_pipeline"] for e in events])
            ),
            "total_nonnegative_violations": int(
                sum(e["metrics"]["nonnegative_violations"]["violation_count"] for e in events)
            ),
        },
        "events": events,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "evaluation_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    return summary
