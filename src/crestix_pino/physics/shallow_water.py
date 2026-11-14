"""2D Shallow Water Equations (SWE) physics utilities and unit conversions.

All PDE calculations operate strictly in SI units:
- Water depth h, bed elevation z_b, spatial spacing dx, dy: meters (m)
- Horizontal velocities u, v: meters per second (m/s)
- Time step dt: seconds (s)
- Rainfall rate R, infiltration rate I: meters per second (m/s)
- Manning roughness coefficient n: s * m^(-1/3)
- Gravitational acceleration g: 9.81 m/s^2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

GRAVITY_M_S2: float = 9.81
MM_PER_HOUR_TO_M_PER_SEC: float = 1.0 / 3_600_000.0


@dataclass(frozen=True)
class ShallowWaterConfig:
    """Configuration for 2D Shallow Water Equations in SI units."""

    dx: float = 2.0
    dy: float = 2.0
    dt: float = 60.0
    gravity: float = GRAVITY_M_S2
    wet_dry_eps: float = 1e-3
    default_manning_n: float = 0.035

    def __post_init__(self) -> None:
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(f"Spatial resolutions dx={self.dx}, dy={self.dy} must be > 0 m.")
        if self.dt <= 0.0:
            raise ValueError(f"Time interval dt={self.dt} must be > 0 s.")
        if self.gravity <= 0.0:
            raise ValueError(f"Gravity={self.gravity} must be > 0 m/s^2.")
        if self.wet_dry_eps <= 0.0:
            raise ValueError(f"wet_dry_eps={self.wet_dry_eps} must be > 0 m.")


def mm_per_hour_to_m_per_sec(rainfall_mm_h: torch.Tensor | float) -> torch.Tensor | float:
    """Convert rainfall intensity from mm/h to m/s.

    1 mm/h = 1e-3 m / 3600 s = 1 / 3,600,000 m/s.
    """
    return rainfall_mm_h * MM_PER_HOUR_TO_M_PER_SEC


def m_per_sec_to_mm_per_hour(rainfall_m_s: torch.Tensor | float) -> torch.Tensor | float:
    """Convert rainfall intensity from m/s to mm/h."""
    return rainfall_m_s * 3_600_000.0


def mm_to_meters(depth_mm: torch.Tensor | float) -> torch.Tensor | float:
    """Convert water depth from millimeters (mm) to meters (m)."""
    return depth_mm * 1e-3


def compute_wet_mask(h: torch.Tensor, wet_dry_eps: float = 1e-3) -> torch.Tensor:
    """Return a floating-point mask (1.0 for wet cells h > wet_dry_eps, 0.0 for dry).

    Used to prevent unphysical momentum/Manning singularities on dry ground.
    """
    return (h > wet_dry_eps).to(dtype=h.dtype)


def compute_manning_friction_slope(
    h: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    manning_n: torch.Tensor | float = 0.035,
    wet_dry_eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Manning bed friction slopes (S_fx, S_fy) [dimensionless].

    Formula:
        S_fx = (n^2 * u * sqrt(u^2 + v^2)) / h_safe^(4/3)
        S_fy = (n^2 * v * sqrt(u^2 + v^2)) / h_safe^(4/3)

    where h_safe = clamp(h, min=wet_dry_eps) avoids division by zero on dry cells,
    and wet_mask zeros out friction on dry regions where h <= wet_dry_eps.
    """
    h_safe = torch.clamp(h, min=wet_dry_eps)
    speed = torch.sqrt(u * u + v * v + 1e-12)
    denom = torch.pow(h_safe, 4.0 / 3.0)
    n_sq = manning_n * manning_n
    wet_mask = compute_wet_mask(h, wet_dry_eps)

    s_fx = wet_mask * (n_sq * u * speed) / denom
    s_fy = wet_mask * (n_sq * v * speed) / denom
    return s_fx, s_fy


def time_derivative(field: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute partial derivative d(field)/dt along the time axis (dim=-3) in 1/s.

    Expects shape (..., T, H, W) with T >= 2.
    Uses second-order central differences for interior steps t in [1, T-2]
    and first-order one-sided differences at t=0 and t=T-1.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0 s, got {dt}")
    t_steps = field.shape[-3]
    if t_steps < 2:
        raise ValueError(f"Need at least 2 time steps to compute d/dt, got T={t_steps}")

    df_dt = torch.empty_like(field)
    df_dt[..., 0, :, :] = (field[..., 1, :, :] - field[..., 0, :, :]) / dt
    if t_steps > 2:
        df_dt[..., 1:-1, :, :] = (field[..., 2:, :, :] - field[..., :-2, :, :]) / (2.0 * dt)
    df_dt[..., -1, :, :] = (field[..., -1, :, :] - field[..., -2, :, :]) / dt
    return df_dt


def spatial_derivatives_2d(
    field: torch.Tensor,
    dx: float,
    dy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute partial derivatives (df/dx, df/dy) along (W, H) in 1/m.

    Expects shape (..., H, W) with H >= 2, W >= 2.
    Note: dim=-2 corresponds to y (rows, spacing dy), dim=-1 corresponds to x (cols, spacing dx).
    Uses second-order central differences in interior and first-order one-sided at edges.
    """
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must be > 0 m, got dx={dx}, dy={dy}")
    h_dim, w_dim = field.shape[-2], field.shape[-1]
    if h_dim < 2 or w_dim < 2:
        raise ValueError(f"Need H >= 2 and W >= 2 for spatial derivatives, got ({h_dim}, {w_dim})")

    df_dx = torch.empty_like(field)
    df_dx[..., :, 0] = (field[..., :, 1] - field[..., :, 0]) / dx
    if w_dim > 2:
        df_dx[..., :, 1:-1] = (field[..., :, 2:] - field[..., :, :-2]) / (2.0 * dx)
    df_dx[..., :, -1] = (field[..., :, -1] - field[..., :, -2]) / dx

    df_dy = torch.empty_like(field)
    df_dy[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dy
    if h_dim > 2:
        df_dy[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dy)
    df_dy[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dy

    return df_dx, df_dy
