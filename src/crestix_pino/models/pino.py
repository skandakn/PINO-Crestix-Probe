"""Physics-Informed Fourier Neural Operator (PINO) for 2D Flood Prediction.

Input Channels (C_in = 4 by default):
    0: dem (m)                  - Digital Elevation Model bed topography z_b(x, y)
    1: rainfall (m/s)           - Meteorological rainfall rate R(x, y, t) in SI [m/s]
    2: initial_depth (m)        - Initial water depth h_0(x, y)
    3: manning_n (s*m^(-1/3))   - Surface roughness coefficient n(x, y)

Outputs (PINOOutput):
    - h: (B, T, H, W) [m]       - Primary supervised output: water depth >= 0
    - u: (B, T, H, W) [m/s]     - Unsupervised latent x-velocity for SWE residuals
    - v: (B, T, H, W) [m/s]     - Unsupervised latent y-velocity for SWE residuals

Note on Label Limitations:
    LarNO UKEA core datasets provide terrain (DEM), rainfall, and water-depth (h)
    fields, NOT direct ground-truth velocity (u, v) labels. Therefore, u and v are
    latent physical variables constrained solely via Shallow Water PDE residuals,
    mass conservation, and boundary conditions. No supervised velocity accuracy is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import FNO3d

NonNegativeMode = Literal["softplus", "relu", "none"]


@dataclass(frozen=True)
class GridMetadata:
    """Native spatial-temporal grid metadata for PINO flood forecasting (SI units)."""

    dx: float = 8.0
    dy: float = 8.0
    dt: float = 300.0
    grid_height: int = 32
    grid_width: int = 32
    crs_or_domain: str = "LarNO-UKEA-32x32"

    def __post_init__(self) -> None:
        if self.dx <= 0.0 or self.dy <= 0.0 or self.dt <= 0.0:
            raise ValueError(
                f"Grid spacings dx={self.dx}, dy={self.dy}, dt={self.dt} must be positive."
            )
        if self.grid_height < 2 or self.grid_width < 2:
            raise ValueError(
                f"Grid dimensions ({self.grid_height}, {self.grid_width}) must be >= 2."
            )


@dataclass(frozen=True)
class PINOModelConfig:
    """Reproducible architecture and horizon configuration for PINOFloodModel."""

    in_channels: int = 4
    width: int = 32
    modes_t: int = 4
    modes_y: int = 12
    modes_x: int = 12
    n_layers: int = 4
    padding_t: int = 2
    padding_y: int = 4
    padding_x: int = 4
    append_grid_coords: bool = True
    predict_latent_velocity: bool = True
    nonnegative_depth_mode: NonNegativeMode = "softplus"
    input_steps: int = 1
    forecast_steps: int = 12
    wall_height: float = 50.0
    default_manning_n: float = 0.035
    grid_metadata: GridMetadata = GridMetadata()


@dataclass
class PINOOutput:
    """Structured output of PINOFloodModel."""

    h: torch.Tensor
    u: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    h_raw: Optional[torch.Tensor] = None
    valid_mask: Optional[torch.Tensor] = None


class PINOFloodModel(nn.Module):
    """Physics-Informed Fourier Neural Operator predicting water depth and latent velocities."""

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        modes_t: int = 4,
        modes_y: int = 12,
        modes_x: int = 12,
        n_layers: int = 4,
        padding_t: int = 2,
        padding_y: int = 4,
        padding_x: int = 4,
        append_grid_coords: bool = True,
        predict_latent_velocity: bool = True,
        nonnegative_depth_mode: NonNegativeMode = "softplus",
        grid_metadata: Optional[GridMetadata] = None,
        input_steps: int = 1,
        forecast_steps: int = 12,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.predict_latent_velocity = predict_latent_velocity
        self.nonnegative_depth_mode = nonnegative_depth_mode
        self.grid_metadata = grid_metadata or GridMetadata()
        self.input_steps = int(input_steps)
        self.forecast_steps = int(forecast_steps)

        self.backbone = FNO3d(
            in_channels=in_channels,
            out_channels=1,
            width=width,
            modes_t=modes_t,
            modes_y=modes_y,
            modes_x=modes_x,
            n_layers=n_layers,
            padding_t=padding_t,
            padding_y=padding_y,
            padding_x=padding_x,
            append_grid_coords=append_grid_coords,
        )

        # Primary depth head
        self.depth_head = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, 1, kernel_size=1),
        )

        # Latent velocity head (u, v) for physics residual evaluation
        if predict_latent_velocity:
            self.velocity_head = nn.Sequential(
                nn.Conv3d(width, width * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(width * 2, 2, kernel_size=1),
            )
        else:
            self.velocity_head = None

    @staticmethod
    def build_input_tensor(
        dem_m: torch.Tensor,
        rainfall_m_s: torch.Tensor,
        initial_depth_m: torch.Tensor,
        manning_n: Optional[torch.Tensor | float] = None,
        default_manning_n: float = 0.035,
    ) -> torch.Tensor:
        """Assemble standard 4-channel input tensor (B, 4, T, H, W) from physical fields.

        Args:
            dem_m: Elevation grid (B, H, W) or (B, T, H, W) in [m].
            rainfall_m_s: Rainfall rate (B, T, H, W) in [m/s].
            initial_depth_m: Initial water depth (B, H, W) or (B, T, H, W) in [m].
            manning_n: Roughness field (B, H, W), (B, T, H, W), or float in [s*m^(-1/3)].
            default_manning_n: Fallback scalar roughness if manning_n is None.
        """
        if rainfall_m_s.ndim != 4:
            raise ValueError(f"Expected rainfall_m_s shape (B, T, H, W), got {tuple(rainfall_m_s.shape)}")

        b, t_dim, h_dim, w_dim = rainfall_m_s.shape

        def _to_4d(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 3:
                return tensor.unsqueeze(1).expand(b, t_dim, h_dim, w_dim)
            if tensor.ndim == 4:
                return tensor
            raise ValueError(f"Expected 3D or 4D tensor, got {tuple(tensor.shape)}")

        dem_4d = _to_4d(dem_m)
        h0_4d = _to_4d(initial_depth_m)

        if manning_n is None:
            n_4d = torch.full_like(rainfall_m_s, float(default_manning_n))
        elif isinstance(manning_n, (int, float)):
            n_4d = torch.full_like(rainfall_m_s, float(manning_n))
        else:
            n_4d = _to_4d(manning_n)

        return torch.stack([dem_4d, rainfall_m_s, h0_4d, n_4d], dim=1)

    def _apply_nonnegative(self, h_raw: torch.Tensor) -> torch.Tensor:
        if self.nonnegative_depth_mode == "softplus":
            return F.softplus(h_raw, beta=5.0)
        if self.nonnegative_depth_mode == "relu":
            return F.relu(h_raw)
        if self.nonnegative_depth_mode == "none":
            return h_raw
        raise ValueError(f"Unknown nonnegative_depth_mode: {self.nonnegative_depth_mode}")

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> PINOOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, T, H, W).
            valid_mask: Optional boolean/float mask of shape (B, H, W), (B, 1, H, W),
                or (B, T, H, W) where True/1.0 denotes valid terrain cells and
                False/0.0 denotes invalid/building DEM cells.

        Returns:
            PINOOutput containing:
                h: (B, T, H, W) predicted water depth [m] (0.0 on masked invalid cells)
                u: (B, T, H, W) latent x-velocity [m/s] (if predict_latent_velocity=True)
                v: (B, T, H, W) latent y-velocity [m/s] (if predict_latent_velocity=True)
                h_raw: (B, T, H, W) pre-activation water depth [m]
                valid_mask: (B, T, H, W) boolean mask if provided
        """
        features = self.backbone.forward_features(x)  # (B, width, T, H, W)

        h_raw = self.depth_head(features).squeeze(1)  # (B, T, H, W)
        h = self._apply_nonnegative(h_raw)

        u = None
        v = None
        if self.velocity_head is not None:
            vel = self.velocity_head(features)        # (B, 2, T, H, W)
            u = vel[:, 0, :, :, :]
            v = vel[:, 1, :, :, :]

        mask_4d: Optional[torch.Tensor] = None
        if valid_mask is not None:
            m = valid_mask.to(device=h.device, dtype=torch.bool)
            if m.ndim == 3:
                mask_4d = m.unsqueeze(1).expand_as(h)
            elif m.ndim == 4 and m.shape[1] == 1:
                mask_4d = m.expand_as(h)
            elif m.ndim == 4 and m.shape == h.shape:
                mask_4d = m
            else:
                raise ValueError(
                    f"Incompatible valid_mask shape {tuple(valid_mask.shape)} for h {tuple(h.shape)}"
                )
            m_float = mask_4d.to(dtype=h.dtype)
            h = h * m_float
            if u is not None and v is not None:
                u = u * m_float
                v = v * m_float

        return PINOOutput(h=h, u=u, v=v, h_raw=h_raw, valid_mask=mask_4d)

    @classmethod
    def from_config(cls, cfg: PINOModelConfig) -> "PINOFloodModel":
        """Instantiate PINOFloodModel from a reproducible PINOModelConfig."""
        return cls(
            in_channels=cfg.in_channels,
            width=cfg.width,
            modes_t=cfg.modes_t,
            modes_y=cfg.modes_y,
            modes_x=cfg.modes_x,
            n_layers=cfg.n_layers,
            padding_t=cfg.padding_t,
            padding_y=cfg.padding_y,
            padding_x=cfg.padding_x,
            append_grid_coords=cfg.append_grid_coords,
            predict_latent_velocity=cfg.predict_latent_velocity,
            nonnegative_depth_mode=cfg.nonnegative_depth_mode,
            grid_metadata=cfg.grid_metadata,
            input_steps=cfg.input_steps,
            forecast_steps=cfg.forecast_steps,
        )

    def forecast(
        self,
        dem_m: torch.Tensor,
        rainfall_m_s: torch.Tensor,
        initial_depth_m: torch.Tensor,
        forecast_steps: Optional[int] = None,
        valid_mask: Optional[torch.Tensor] = None,
        manning_n: Optional[torch.Tensor | float] = None,
        wall_height: float = 50.0,
        default_manning_n: float = 0.035,
    ) -> PINOOutput:
        """Run masked spatio-temporal forecast for configurable input_steps and forecast_steps.

        Handles NaN/invalid DEM cells by constructing `valid_mask = ~torch.isnan(dem_m)`,
        imputing invalid DEM cells with `max(valid_dem) + wall_height`, zeroing rainfall
        and initial depth on invalid cells, and supporting either (B, H, W) initial depth
        or (B, T_in, H, W) historical depth context.
        """
        steps = int(forecast_steps if forecast_steps is not None else self.forecast_steps)
        if steps < 1:
            raise ValueError(f"forecast_steps must be >= 1, got {steps}.")

        if dem_m.ndim == 2:
            dem_b = dem_m.unsqueeze(0)
        elif dem_m.ndim == 3:
            dem_b = dem_m
        else:
            raise ValueError(f"Expected dem_m shape (H, W) or (B, H, W), got {tuple(dem_m.shape)}")

        b, h_dim, w_dim = dem_b.shape
        if valid_mask is None:
            mask_3d = torch.isfinite(dem_b)
        else:
            mask_3d = valid_mask.to(device=dem_b.device, dtype=torch.bool)
            if mask_3d.ndim == 2:
                mask_3d = mask_3d.unsqueeze(0).expand(b, h_dim, w_dim)

        # Impute invalid DEM cells with elevated wall height while keeping mask_3d for output masking
        dem_clean = dem_b.clone()
        for idx in range(b):
            valid_vals = dem_b[idx][mask_3d[idx]]
            fill_val = (float(valid_vals.max().item()) + wall_height) if valid_vals.numel() > 0 else wall_height
            dem_clean[idx] = torch.where(mask_3d[idx], dem_b[idx], torch.full_like(dem_b[idx], fill_val))

        # Align rainfall to (B, steps, H, W)
        if rainfall_m_s.ndim == 1 and rainfall_m_s.shape[0] == steps:
            rain_4d = rainfall_m_s.view(1, steps, 1, 1).expand(b, steps, h_dim, w_dim)
        elif rainfall_m_s.ndim == 3 and rainfall_m_s.shape == (steps, h_dim, w_dim):
            rain_4d = rainfall_m_s.unsqueeze(0).expand(b, steps, h_dim, w_dim)
        elif rainfall_m_s.ndim == 4 and rainfall_m_s.shape == (b, steps, h_dim, w_dim):
            rain_4d = rainfall_m_s
        else:
            raise ValueError(
                f"Incompatible rainfall_m_s shape {tuple(rainfall_m_s.shape)} for (B={b}, T={steps}, H={h_dim}, W={w_dim})."
            )
        rain_4d = rain_4d * mask_3d.unsqueeze(1).to(dtype=rain_4d.dtype)

        # Align initial_depth_m: (B, H, W) or (B, T_in, H, W) -> (B, steps, H, W)
        if initial_depth_m.ndim == 2:
            h0_b = initial_depth_m.unsqueeze(0).expand(b, h_dim, w_dim)
            h0_4d = h0_b.unsqueeze(1).expand(b, steps, h_dim, w_dim)
        elif initial_depth_m.ndim == 3 and initial_depth_m.shape == (b, h_dim, w_dim):
            h0_4d = initial_depth_m.unsqueeze(1).expand(b, steps, h_dim, w_dim)
        elif initial_depth_m.ndim == 4 and initial_depth_m.shape[0] == b and initial_depth_m.shape[2:] == (h_dim, w_dim):
            t_in = initial_depth_m.shape[1]
            if t_in >= steps:
                h0_4d = initial_depth_m[:, -steps:, :, :]
            else:
                pad_last = initial_depth_m[:, -1:, :, :].expand(b, steps - t_in, h_dim, w_dim)
                h0_4d = torch.cat([initial_depth_m, pad_last], dim=1)
        else:
            raise ValueError(
                f"Incompatible initial_depth_m shape {tuple(initial_depth_m.shape)} for (B={b}, H={h_dim}, W={w_dim})."
            )
        h0_4d = h0_4d * mask_3d.unsqueeze(1).to(dtype=h0_4d.dtype)

        x = self.build_input_tensor(
            dem_m=dem_clean,
            rainfall_m_s=rain_4d,
            initial_depth_m=h0_4d,
            manning_n=manning_n,
            default_manning_n=default_manning_n,
        )
        return self.forward(x, valid_mask=mask_3d)
