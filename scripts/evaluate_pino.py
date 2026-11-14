"""CLI entrypoint to evaluate a trained PINO checkpoint on held-out LarNO flood events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crestix_pino.evaluation.evaluate import evaluate_dataset
from crestix_pino.inference.predictor import PINOPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained PINO model against held-out LarNO flood events."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained PINO checkpoint (.pt, .pts, or .pth).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a .npz flood event file or directory of .npz held-out events.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation",
        help="Directory to write metrics JSON, predicted arrays, and error maps.",
    )
    parser.add_argument("--device", default="cpu", help="Inference device (e.g., cpu, cuda).")
    parser.add_argument(
        "--manning-n",
        type=float,
        default=0.035,
        help="Manning roughness coefficient for physics residual (s * m^-1/3).",
    )
    parser.add_argument(
        "--clamp-negative",
        action="store_true",
        help="Clamp negative predicted depths to 0.0 during postprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = PINOPredictor(
        model=args.checkpoint,
        device=args.device,
        clamp_negative=args.clamp_negative,
    )
    summary = evaluate_dataset(
        predictor=predictor,
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        manning_n=args.manning_n,
    )
    print(json.dumps(summary["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
