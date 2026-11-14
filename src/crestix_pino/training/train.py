"""CLI and programmatic entrypoint for PINO training on UKEA 32x32 flood data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..data.dataloader import build_dataloaders
from ..data.dataset import download_ukea_dataset
from .config import TrainConfig, load_config
from .trainer import PINOTrainer


def run_training(cfg: TrainConfig) -> Dict[str, Any]:
    """Run data preparation (if requested) and execute PINOTrainer.fit()."""
    if cfg.data.auto_download:
        download_ukea_dataset(cfg.data.data_root)

    pin_memory = cfg.device == "cuda" or (
        cfg.device == "auto" and __import__("torch").cuda.is_available()
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data, seed=cfg.seed, pin_memory=pin_memory
    )

    trainer = PINOTrainer(cfg)
    summary = trainer.fit(train_loader, val_loader)
    summary["test_batches"] = len(test_loader)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PINO on LarNO UKEA 32x32 flood data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ukea_32x32.yaml",
        help="Path to YAML training configuration.",
    )
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output/artifact directory.")
    parser.add_argument("--data-root", type=str, default=None, help="Override UKEA dataset root.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download ukea_8m_5min from Hugging Face if missing.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 2-epoch CPU smoke test.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.optim.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.checkpoint.output_dir = args.output_dir
    if args.data_root is not None:
        cfg.data.data_root = args.data_root
    if args.resume is not None:
        cfg.checkpoint.resume_path = args.resume
    if args.seed is not None:
        cfg.seed = args.seed
    if args.download_data:
        cfg.data.auto_download = True
    if args.smoke_test:
        cfg.device = "cpu"
        cfg.optim.epochs = 2
        cfg.data.batch_size = 2
        cfg.logging.tensorboard = False

    summary = run_training(cfg)
    logging.getLogger(__name__).info("Training complete: %s", summary)
    return summary


if __name__ == "__main__":
    main()
