"""Benchmark PINO model-only latency vs. full inference pipeline latency."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from crestix_pino.inference.predictor import PINOPredictor


def _summarize_timings(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def run_benchmark(
    predictor: PINOPredictor,
    batch_size: int,
    height: int,
    width: int,
    forecast_steps: int,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    """Measure model-only and full-pipeline latency under fixed tensor shapes."""
    rng = np.random.default_rng(42)
    if batch_size == 1:
        dem = rng.uniform(0.0, 20.0, size=(height, width)).astype(np.float32)
        h0 = np.zeros((height, width), dtype=np.float32)
    else:
        dem = rng.uniform(0.0, 20.0, size=(batch_size, height, width)).astype(np.float32)
        h0 = np.zeros((batch_size, height, width), dtype=np.float32)
    rainfall = np.full((forecast_steps,), 50.0, dtype=np.float32)

    prepared = predictor.preprocess(dem, rainfall, h0, forecast_steps)

    for _ in range(warmup):
        predictor.infer_model_only(prepared)
        predictor.predict(dem, rainfall, h0, forecast_steps)

    model_only_ms: list[float] = []
    full_pipeline_ms: list[float] = []

    for _ in range(runs):
        t0 = time.perf_counter()
        predictor.infer_model_only(prepared)
        t1 = time.perf_counter()
        model_only_ms.append((t1 - t0) * 1000.0)

        t2 = time.perf_counter()
        predictor.predict(dem, rainfall, h0, forecast_steps)
        t3 = time.perf_counter()
        full_pipeline_ms.append((t3 - t2) * 1000.0)

    model_stats = _summarize_timings(model_only_ms)
    full_stats = _summarize_timings(full_pipeline_ms)

    return {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "device": predictor.loaded_model.device,
            "framework": predictor.loaded_model.framework,
        },
        "settings": {
            "batch_size": batch_size,
            "height": height,
            "width": width,
            "forecast_steps": forecast_steps,
            "warmup_runs": warmup,
            "timed_runs": runs,
        },
        "model_only_latency": model_stats,
        "full_pipeline_latency": full_stats,
        "pipeline_overhead_ms": full_stats["mean_ms"] - model_stats["mean_ms"],
        "full_pipeline_throughput_samples_per_sec": (
            batch_size / (full_stats["mean_ms"] / 1000.0)
            if full_stats["mean_ms"] > 0
            else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PINO model-only vs. full pipeline inference latency."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained PINO checkpoint (.pt, .pts, or .pth).",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (cpu, cuda).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save benchmark JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = PINOPredictor(model=args.checkpoint, device=args.device)
    report = run_benchmark(
        predictor=predictor,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        forecast_steps=args.steps,
        warmup=args.warmup,
        runs=args.runs,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
