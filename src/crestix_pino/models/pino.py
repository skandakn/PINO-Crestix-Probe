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


@dataclass
class PINOOutput:
    """Structured output of PINOFloodModel."""

    h: torch.Tensor
    u: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    h_raw: Optional[torch.Tensor] = None


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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.predict_latent_velocity = predict_latent_velocity
        self.nonnegative_depth_mode = nonnegative_depth_mode

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

    def forward(self, x: torch.Tensor) -> PINOOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, T, H, W).

        Returns:
            PINOOutput containing:
                h: (B, T, H, W) predicted water depth [m]
                u: (B, T, H, W) latent x-velocity [m/s] (if predict_latent_velocity=True)
                v: (B, T, H, W) latent y-velocity [m/s] (if predict_latent_velocity=True)
                h_raw: (B, T, H, W) pre-activation water depth [m]
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

        return PINOOutput(h=h, u=u, v=v, h_raw=h_raw)
