"""Structured configuration dataclasses and YAML serialization for PINO training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    data_root: str = "data/ukea_8m_5min"
    auto_download: bool = False
    val_event_count: int = 2
    crop_y: int = 9
    crop_x: int = 44
    region_height: int = 32
    region_width: int = 32
    window_t: Optional[int] = 12
    stride_t: int = 4
    wall_height: float = 50.0
    manning_n: float = 0.035
    rainfall_unit_scale: float = 1.0 / (300.0 * 1000.0)
    batch_size: int = 4
    num_workers: int = 0


@dataclass
class ModelConfig:
    in_channels: int = 4
    append_grid_coords: bool = True
    width: int = 32
    modes_t: int = 4
    modes_y: int = 12
    modes_x: int = 12
    n_layers: int = 4
    padding_t: int = 2
    padding_y: int = 4
    padding_x: int = 4
    predict_latent_velocity: bool = True
    nonnegative_depth_mode: str = "softplus"


@dataclass
class PhysicsConfig:
    gravity: float = 9.81
    dx: float = 8.0
    dy: float = 8.0
    dt: float = 300.0
    wet_dry_eps: float = 1.0e-3
    default_manning_n: float = 0.035
    data_loss_type: str = "mse"
    data_weight: float = 1.0
    physics_weight: float = 0.1
    pde_continuity_weight: float = 1.0
    pde_momentum_x_weight: float = 0.25
    pde_momentum_y_weight: float = 0.25
    mass_conservation_weight: float = 0.5
    boundary_condition_weight: float = 0.5
    initial_condition_weight: float = 1.0
    nonnegative_depth_weight: float = 1.0


@dataclass
class OptimConfig:
    epochs: int = 50
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    scheduler: str = "cosine"
    min_lr: float = 1.0e-5
    step_size: int = 15
    gamma: float = 0.5
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1.0e-6


@dataclass
class CheckpointConfig:
    output_dir: str = "artifacts/ukea_32x32"
    save_every_epochs: int = 5
    resume_path: Optional[str] = None


@dataclass
class LoggingConfig:
    tensorboard: bool = True
    log_every_steps: int = 1
    plot_losses: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "auto"
    deterministic: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(dc_cls: Any, values: Optional[Dict[str, Any]]) -> Any:
    if not values:
        return dc_cls()
    valid_keys = {f.name for f in dc_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in values.items() if k in valid_keys}
    return dc_cls(**filtered)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    """Load a YAML configuration file into a TrainConfig instance."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        for key, val in overrides.items():
            if val is not None:
                raw[key] = val

    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        deterministic=bool(raw.get("deterministic", True)),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        physics=_merge_dataclass(PhysicsConfig, raw.get("physics")),
        optim=_merge_dataclass(OptimConfig, raw.get("optim")),
        checkpoint=_merge_dataclass(CheckpointConfig, raw.get("checkpoint")),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging")),
    )


def save_config(config: TrainConfig, path: str | Path) -> Path:
    """Persist TrainConfig to a YAML file."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    return dst

