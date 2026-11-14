"""Reproducible PINO training and validation loop with checkpointing, LR scheduling, and logging."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainConfig, save_config

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto', 'cpu', or 'cuda' into a torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but unavailable; falling back to CPU.", device_str)
        return torch.device("cpu")
    return torch.device(device_str)


@dataclass
class _FallbackPINOOutput:
    h: torch.Tensor
    u: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    h_raw: Optional[torch.Tensor] = None


class _SpectralConv3d(nn.Module):
    """Lightweight 3D Fourier layer used when full models package is not yet merged."""

    def __init__(self, in_ch: int, out_ch: int, mt: int, my: int, mx: int) -> None:
        super().__init__()
        self.mt, self.my, self.mx = mt, my, mx
        scale = 1.0 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, mt, my, mx, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t_dim, h_dim, w_dim = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            b,
            self.weights.shape[1],
            t_dim,
            h_dim,
            w_dim // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mt = min(self.mt, t_dim)
        my = min(self.my, h_dim)
        mx = min(self.mx, w_dim // 2 + 1)
        out_ft[:, :, :mt, :my, :mx] = torch.einsum(
            "bctyx,coyx->botyx",
            x_ft[:, :, :mt, :my, :mx],
            self.weights[:, :, :mt, :my, :mx].permute(0, 1, 3, 4, 2)[..., 0],
        )
        return torch.fft.irfftn(out_ft, s=(t_dim, h_dim, w_dim), dim=(-3, -2, -1))


class _FallbackPINO3d(nn.Module):
    """Self-contained 3D FNO/PINO fallback model matching PINOFloodModel's I/O contract."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        m = cfg.model
        in_ch = m.in_channels + (3 if m.append_grid_coords else 0)
        self.append_grid_coords = m.append_grid_coords
        self.lift = nn.Conv3d(in_ch, m.width, kernel_size=1)
        self.convs = nn.ModuleList(
            [
                nn.Conv3d(m.width, m.width, kernel_size=3, padding=1)
                for _ in range(m.n_layers)
            ]
        )
        self.skips = nn.ModuleList(
            [nn.Conv3d(m.width, m.width, kernel_size=1) for _ in range(m.n_layers)]
        )
        self.depth_head = nn.Sequential(
            nn.Conv3d(m.width, m.width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(m.width * 2, 1, kernel_size=1),
        )
        self.vel_head = (
            nn.Sequential(
                nn.Conv3d(m.width, m.width * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(m.width * 2, 2, kernel_size=1),
            )
            if m.predict_latent_velocity
            else None
        )
        self.nonneg_mode = m.nonnegative_depth_mode

    def _append_coords(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t_dim, h_dim, w_dim = x.shape
        t_c = torch.linspace(0.0, 1.0, t_dim, device=x.device, dtype=x.dtype)
        y_c = torch.linspace(0.0, 1.0, h_dim, device=x.device, dtype=x.dtype)
        x_c = torch.linspace(0.0, 1.0, w_dim, device=x.device, dtype=x.dtype)
        gt, gy, gx = torch.meshgrid(t_c, y_c, x_c, indexing="ij")
        coords = torch.stack([gt, gy, gx], dim=0).unsqueeze(0).expand(b, -1, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, x: torch.Tensor) -> _FallbackPINOOutput:
        if self.append_grid_coords:
            x = self._append_coords(x)
        feat = self.lift(x)
        for conv, skip in zip(self.convs, self.skips):
            feat = F.gelu(conv(feat) + skip(feat))
        h_raw = self.depth_head(feat).squeeze(1)
        if self.nonneg_mode == "softplus":
            h = F.softplus(h_raw, beta=5.0)
        elif self.nonneg_mode == "relu":
            h = F.relu(h_raw)
        else:
            h = h_raw
        u, v = None, None
        if self.vel_head is not None:
            vel = self.vel_head(feat)
            u, v = vel[:, 0], vel[:, 1]
        return _FallbackPINOOutput(h=h, u=u, v=v, h_raw=h_raw)


class _FallbackPINOLoss(nn.Module):
    """Masked data MSE + initial condition + continuity residual fallback loss."""

    def __init__(self, dt: float = 300.0, dx: float = 8.0, dy: float = 8.0, physics_weight: float = 0.1) -> None:
        super().__init__()
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.physics_weight = physics_weight

    def forward(
        self,
        output: Any,
        h_true: torch.Tensor,
        dem: torch.Tensor,
        rainfall_m_s: torch.Tensor,
        h_init: torch.Tensor,
        manning_n: Optional[torch.Tensor | float] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_pred = output.h
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).expand_as(h_pred).float()
            denom = mask.sum().clamp_min(1.0)
            l_data = (((h_pred - h_true) ** 2) * mask).sum() / denom
        else:
            l_data = F.mse_loss(h_pred, h_true)

        l_ic = F.mse_loss(h_pred[:, 0], h_init)
        l_phys = l_ic
        if output.u is not None and output.v is not None and h_pred.shape[1] > 1:
            dh_dt = (h_pred[:, 1:] - h_pred[:, :-1]) / self.dt
            hu = h_pred[:, :-1] * output.u[:, :-1]
            hv = h_pred[:, :-1] * output.v[:, :-1]
            dhu_dx = (torch.roll(hu, -1, dims=-1) - torch.roll(hu, 1, dims=-1)) / (2.0 * self.dx)
            dhv_dy = (torch.roll(hv, -1, dims=-2) - torch.roll(hv, 1, dims=-2)) / (2.0 * self.dy)
            cont_res = dh_dt + dhu_dx + dhv_dy - rainfall_m_s[:, :-1]
            l_phys = l_ic + (cont_res[:, :, 1:-1, 1:-1] ** 2).mean()

        total = l_data + self.physics_weight * l_phys
        return total, {
            "data_loss": l_data.detach(),
            "total_physics": l_phys.detach(),
            "total_loss": total.detach(),
        }


def build_model_and_criterion(cfg: TrainConfig) -> Tuple[nn.Module, nn.Module, str]:
    """Instantiate PINOFloodModel + PINOCompositeLoss if available, else fallback."""
    try:
        from ..losses.data_loss import DataLoss
        from ..losses.physics_loss import (
            PINOCompositeLoss,
            PhysicsLoss,
            PhysicsLossWeights,
        )
        from ..models.pino import PINOFloodModel
        from ..physics.boundary_conditions import BoundaryConfig
        from ..physics.shallow_water import ShallowWaterConfig

        m = cfg.model
        p = cfg.physics
        model = PINOFloodModel(
            in_channels=m.in_channels,
            width=m.width,
            modes_t=m.modes_t,
            modes_y=m.modes_y,
            modes_x=m.modes_x,
            n_layers=m.n_layers,
            padding_t=m.padding_t,
            padding_y=m.padding_y,
            padding_x=m.padding_x,
            append_grid_coords=m.append_grid_coords,
            predict_latent_velocity=m.predict_latent_velocity,
            nonnegative_depth_mode=m.nonnegative_depth_mode,
        )
        sw_cfg = ShallowWaterConfig(
            gravity=p.gravity,
            dx=p.dx,
            dy=p.dy,
            dt=p.dt,
            wet_dry_eps=p.wet_dry_eps,
            default_manning_n=p.default_manning_n,
        )
        bc_cfg = BoundaryConfig()
        weights = PhysicsLossWeights(
            pde_continuity=p.pde_continuity_weight,
            pde_momentum_x=p.pde_momentum_x_weight,
            pde_momentum_y=p.pde_momentum_y_weight,
            mass_conservation=p.mass_conservation_weight,
            boundary_condition=p.boundary_condition_weight,
            initial_condition=p.initial_condition_weight,
            nonnegative_depth=p.nonnegative_depth_weight,
        )
        criterion = PINOCompositeLoss(
            data_loss=DataLoss(loss_type=p.data_loss_type),
            physics_loss=PhysicsLoss(sw_config=sw_cfg, boundary_config=bc_cfg, weights=weights),
            data_weight=p.data_weight,
            physics_weight=p.physics_weight,
        )
        return model, criterion, "PINOFloodModel+PINOCompositeLoss"
    except ImportError:
        logger.info("crestix_pino.models/losses not found; using built-in _FallbackPINO3d.")
        model = _FallbackPINO3d(cfg)
        criterion = _FallbackPINOLoss(
            dt=cfg.physics.dt,
            dx=cfg.physics.dx,
            dy=cfg.physics.dy,
            physics_weight=cfg.physics.physics_weight,
        )
        return model, criterion, "FallbackPINO3d+FallbackPINOLoss"


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> Optional[Any]:
    opt = cfg.optim
    if opt.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, opt.epochs), eta_min=opt.min_lr
        )
    if opt.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=opt.gamma, patience=max(1, opt.early_stopping_patience // 2)
        )
    if opt.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.step_size, gamma=opt.gamma
        )
    if opt.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {opt.scheduler}")


class PINOTrainer:
    """End-to-end trainer for PINO flood models with checkpointing and telemetry."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed, deterministic=cfg.deterministic)
        self.device = resolve_device(cfg.device)
        self.output_dir = Path(cfg.checkpoint.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.tb_dir = self.output_dir / "tensorboard"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_config(cfg, self.output_dir / "resolved_config.yaml")

        self.model, self.criterion, self.arch_name = build_model_and_criterion(cfg)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
        self.scheduler = build_scheduler(self.optimizer, cfg)

        self.writer = None
        if cfg.logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.tb_dir))
            except Exception as exc:
                logger.warning("TensorBoard unavailable (%s); skipping SummaryWriter.", exc)

        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_data_loss": [],
            "val_loss": [],
            "val_data_loss": [],
            "val_rmse": [],
            "lr": [],
            "data_time_s": [],
            "epoch_time_s": [],
            "gpu_max_mem_mb": [],
        }

        if cfg.checkpoint.resume_path:
            self._resume(cfg.checkpoint.resume_path)

    def _resume(self, resume_path: str) -> None:
        payload = load_checkpoint(
            resume_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.start_epoch = int(payload["epoch"]) + 1
        self.global_step = int(payload.get("global_step", 0))
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        if payload.get("history"):
            self.history = payload["history"]
        logger.info(
            "Resumed from %s at epoch %d (best_val_loss=%.6f)",
            resume_path,
            self.start_epoch,
            self.best_val_loss,
        )

    def _step_batch(self, batch: Dict[str, Any], train: bool) -> Tuple[torch.Tensor, Dict[str, float]]:
        inputs = batch["inputs"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        dem = batch["dem"].to(self.device, non_blocking=True)
        rainfall = batch["rainfall_m_s"].to(self.device, non_blocking=True)
        h_init = batch["h_init"].to(self.device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(self.device, non_blocking=True)
        manning_n = float(batch["manning_n"][0].item())

        if train:
            self.optimizer.zero_grad(set_to_none=True)

        output = self.model(inputs)
        loss, breakdown = self.criterion(
            output=output,
            h_true=targets,
            dem=dem,
            rainfall_m_s=rainfall,
            h_init=h_init,
            manning_n=manning_n,
            valid_mask=valid_mask,
        )

        if train:
            loss.backward()
            if self.cfg.optim.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.optim.grad_clip_norm
                )
            self.optimizer.step()

        with torch.no_grad():
            mask_4d = valid_mask.unsqueeze(1).expand_as(output.h).float()
            mse_masked = (((output.h - targets) ** 2) * mask_4d).sum() / mask_4d.sum().clamp_min(1.0)
            rmse = torch.sqrt(mse_masked)

        metrics = {
            "total_loss": float(loss.detach().item()),
            "data_loss": float(breakdown["data_loss"].item()),
            "rmse": float(rmse.item()),
        }
        return loss, metrics

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        total_loss = 0.0
        total_data_loss = 0.0
        total_rmse = 0.0
        data_time = 0.0
        n_batches = 0

        t_end = time.perf_counter()
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                data_time += time.perf_counter() - t_end
                _, metrics = self._step_batch(batch, train=train)
                total_loss += metrics["total_loss"]
                total_data_loss += metrics["data_loss"]
                total_rmse += metrics["rmse"]
                n_batches += 1
                if train:
                    self.global_step += 1
                t_end = time.perf_counter()

        denom = max(1, n_batches)
        return {
            "loss": total_loss / denom,
            "data_loss": total_data_loss / denom,
            "rmse": total_rmse / denom,
            "data_time_s": data_time,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Run full training and validation loop with early stopping and plotting."""
        logger.info(
            "Starting training (%s) on device=%s | train_batches=%d | val_batches=%d",
            self.arch_name,
            self.device,
            len(train_loader),
            len(val_loader),
        )

        no_improve = 0
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            t0 = time.perf_counter()
            train_stats = self._run_epoch(train_loader, train=True)
            val_stats = self._run_epoch(val_loader, train=False)
            epoch_time = time.perf_counter() - t0

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()

            lr_now = float(self.optimizer.param_groups[0]["lr"])
            gpu_mem_mb = (
                float(torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2))
                if self.device.type == "cuda"
                else 0.0
            )

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_stats["loss"])
            self.history["train_data_loss"].append(train_stats["data_loss"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["val_data_loss"].append(val_stats["data_loss"])
            self.history["val_rmse"].append(val_stats["rmse"])
            self.history["lr"].append(lr_now)
            self.history["data_time_s"].append(train_stats["data_time_s"])
            self.history["epoch_time_s"].append(epoch_time)
            self.history["gpu_max_mem_mb"].append(gpu_mem_mb)

            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_stats["loss"], epoch)
                self.writer.add_scalar("loss/val", val_stats["loss"], epoch)
                self.writer.add_scalar("rmse/val", val_stats["rmse"], epoch)
                self.writer.add_scalar("perf/data_time_s", train_stats["data_time_s"], epoch)
                self.writer.add_scalar("perf/lr", lr_now, epoch)
                if self.device.type == "cuda":
                    self.writer.add_scalar("perf/gpu_max_mem_mb", gpu_mem_mb, epoch)

            logger.info(
                "Epoch %03d/%03d | train_loss=%.6f | val_loss=%.6f | val_rmse=%.5f m | "
                "data_time=%.3fs | epoch_time=%.3fs | lr=%.2e | gpu_mem=%.1f MB",
                epoch + 1,
                self.cfg.optim.epochs,
                train_stats["loss"],
                val_stats["loss"],
                val_stats["rmse"],
                train_stats["data_time_s"],
                epoch_time,
                lr_now,
                gpu_mem_mb,
            )

            improved = val_stats["loss"] < (
                self.best_val_loss - self.cfg.optim.early_stopping_min_delta
            )
            if improved:
                self.best_val_loss = val_stats["loss"]
                no_improve = 0
                save_checkpoint(
                    self.ckpt_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    history=self.history,
                    config=self.cfg.to_dict(),
                    metadata_extra={
                        "arch": self.arch_name,
                        "device": str(self.device),
                        "val_loss": val_stats["loss"],
                        "val_rmse": val_stats["rmse"],
                    },
                )
            else:
                no_improve += 1

            if (epoch + 1) % max(1, self.cfg.checkpoint.save_every_epochs) == 0 or (
                epoch + 1 == self.cfg.optim.epochs
            ):
                save_checkpoint(
                    self.ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    best_val_loss=self.best_val_loss,
                    history=self.history,
                    config=self.cfg.to_dict(),
                    metadata_extra={
                        "arch": self.arch_name,
                        "device": str(self.device),
                        "val_loss": val_stats["loss"],
                        "val_rmse": val_stats["rmse"],
                    },
                )

            if (
                self.cfg.optim.early_stopping_patience > 0
                and no_improve >= self.cfg.optim.early_stopping_patience
            ):
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        history_path = self.output_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        if self.cfg.logging.plot_losses:
            self._save_loss_plot(self.output_dir / "loss_curves.png")

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return {
            "best_val_loss": self.best_val_loss,
            "epochs_completed": len(self.history["epoch"]),
            "output_dir": str(self.output_dir),
            "history_path": str(history_path),
            "best_checkpoint": str(self.ckpt_dir / "best.pt"),
        }

    def _save_loss_plot(self, dst: Path) -> None:
        if not self.history["epoch"]:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [e + 1 for e in self.history["epoch"]]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            axes[0].plot(epochs, self.history["train_loss"], label="Train Total Loss", marker="o", markersize=3)
            axes[0].plot(epochs, self.history["val_loss"], label="Val Total Loss", marker="s", markersize=3)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training vs Validation Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(epochs, self.history["val_rmse"], label="Val Water-Depth RMSE (m)", color="tab:green", marker="^", markersize=3)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("RMSE (m)")
            axes[1].set_title("Validation Water-Depth RMSE")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(dst, dpi=150)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Failed to generate loss plot (%s).", exc)
