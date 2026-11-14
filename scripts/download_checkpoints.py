"""Checkpoint inventory & Google Drive download instructions for PINO-Crestix-Probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

CHECKPOINT_REGISTRY = {
    "pino_ukea_32x32_best": {
        "architecture": "PINOFloodModel (FNO3d + latent velocity heads)",
        "drive_path": "GoogleDrive://Crestix-Probe/PINO/checkpoints/best.pt",
        "local_path": "artifacts/ukea_32x32/checkpoints/best.pt",
        "config": "configs/ukea_32x32.yaml",
    },
    "pino_common_eval_best": {
        "architecture": "PINOFloodModel common-evaluation release weight alias",
        "drive_path": "GoogleDrive://Crestix-Probe/PINO/checkpoints/best_pino_model.pt",
        "local_path": "artifacts/ukea_32x32/best_pino_model.pt",
        "config": "configs/ukea_32x32.yaml",
    },
    "pino_ukea_32x32_last": {
        "architecture": "PINOFloodModel resume state",
        "drive_path": "GoogleDrive://Crestix-Probe/PINO/checkpoints/last.pt",
        "local_path": "artifacts/ukea_32x32/checkpoints/last.pt",
        "config": "configs/ukea_32x32.yaml",
    },
    "pino_smoke_test": {
        "architecture": "PINOFloodModel local CPU smoke checkpoint",
        "drive_path": "N/A (generated locally via scripts/train_pino.py --smoke-test)",
        "local_path": "artifacts/smoke_test/checkpoints/best.pt",
        "config": "configs/ukea_32x32.yaml",
    },
    "larno_raw_dataset": {
        "architecture": "LarNO 100-Event Benchmark (80 Futian 20m + 20 UKEA 8m)",
        "drive_path": "GoogleDrive://Crestix-Probe/Datasets/LarNO-UKEA-100Events/",
        "local_path": "data/ukea_8m_5min",
        "config": "configs/larno_data_pipeline.yaml",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect shared Google Drive checkpoint paths and verify local availability."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether expected local checkpoint/data paths exist.",
    )
    parser.add_argument(
        "--drive-url",
        type=str,
        default=None,
        help="Optional public Google Drive file/folder ID or URL to fetch via gdown if installed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ukea_32x32/checkpoints/best.pt"),
        help="Local destination file when using --drive-url.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.drive_url:
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Optional dependency `gdown` is required for --drive-url. "
                "Install it with `pip install gdown` or download manually from the shared Crestix-Probe Google Drive."
            ) from exc
        args.output.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(args.drive_url, str(args.output), fuzzy=True)
        return

    report = {}
    for key, entry in CHECKPOINT_REGISTRY.items():
        local = Path(entry["local_path"])
        report[key] = {
            **entry,
            "present_locally": local.exists(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
