"""CPU smoke tests for PINO training loop, checkpointing, resume, and early stopping."""

from __future__ import annotations

import json
from pathlib import Path

from crestix_pino.training.config import (
    CheckpointConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from crestix_pino.training.train import run_training
from tests.test_dataset import _create_synthetic_ukea


def test_cpu_smoke_train_checkpoint_and_resume(tmp_path: Path) -> None:
    data_root = _create_synthetic_ukea(tmp_path / "data")
    out_dir = tmp_path / "artifacts"

    cfg = TrainConfig(
        seed=123,
        device="cpu",
        data=DataConfig(
            data_root=str(data_root),
            val_event_count=2,
            window_t=8,
            stride_t=16,
            batch_size=2,
        ),
        model=ModelConfig(width=8, modes_t=2, modes_y=4, modes_x=4, n_layers=2),
        optim=OptimConfig(epochs=2, lr=1e-3, scheduler="cosine", early_stopping_patience=5),
        checkpoint=CheckpointConfig(output_dir=str(out_dir), save_every_epochs=1),
        logging=LoggingConfig(tensorboard=False, plot_losses=True),
    )

    summary = run_training(cfg)
    assert summary["epochs_completed"] == 2
    assert (out_dir / "checkpoints" / "best.pt").exists()
    assert (out_dir / "checkpoints" / "last.pt").exists()
    assert (out_dir / "checkpoints" / "last.metadata.json").exists()
    assert (out_dir / "history.json").exists()
    assert (out_dir / "loss_curves.png").exists()

    with open(out_dir / "checkpoints" / "last.metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["epoch"] == 1
    assert meta["num_parameters"] > 0

    # Resume training for 2 more epochs
    cfg.optim.epochs = 4
    cfg.checkpoint.resume_path = str(out_dir / "checkpoints" / "last.pt")
    resumed_summary = run_training(cfg)
    assert resumed_summary["epochs_completed"] == 4
