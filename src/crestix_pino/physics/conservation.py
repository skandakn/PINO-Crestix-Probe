"""Global and step-wise mass conservation equations for 2D flood domains."""

from __future__ import annotations

from typing import Optional

import torch

from .boundary_conditions import BoundaryConfig
from .shallow_water import ShallowWaterConfig


def compute_domain_volume(
    h: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute total water volume [m^3] over spatial domain (H, W) at each time step.

    Args:
        h: Water depth tensor of shape (B, T, H, W) in meters [m].
        dx: Cell width in meters [m].
        dy: Cell height in meters [m].

    Returns:
        Volume tensor of shape (B, T) in cubic meters [m^3].
    """
    h_pos = torch.clamp(h, min=0.0)
    cell_area = dx * dy
    return h_pos.sum(dim=(-2, -1)) * cell_area


def compute_boundary_outflow_flux(
    h: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    boundary_config: Optional[BoundaryConfig] = None,
) -> torch.Tensor:
    """Compute net outward volumetric flux [m^3/s] across domain boundaries.

    Conventions:
    - West boundary (col 0): outward normal is -x, so outward flux = -sum(h * u[:, :, :, 0]) * dy
    - East boundary (col -1): outward normal is +x, so outward flux = +sum(h * u[:, :, :, -1]) * dy
    - North boundary (row 0): outward normal is -y, so outward flux = -sum(h * v[:, :, 0, :]) * dx
    - South boundary (row -1): outward normal is +y, so outward flux = +sum(h * v[:, :, -1, :]) * dx
    Closed boundaries contribute zero net outflow.
    """
    bc = boundary_config or BoundaryConfig()
    h_pos = torch.clamp(h, min=0.0)
    hu = h_pos * u
    hv = h_pos * v

    flux = torch.zeros(h.shape[:2], device=h.device, dtype=h.dtype)

    if bc.west != "closed":
        flux = flux - hu[..., :, 0].sum(dim=-1) * dy
    if bc.east != "closed":
        flux = flux + hu[..., :, -1].sum(dim=-1) * dy
    if bc.north != "closed":
        flux = flux - hv[..., 0, :].sum(dim=-1) * dx
    if bc.south != "closed":
        flux = flux + hv[..., -1, :].sum(dim=-1) * dx

    return flux


def compute_mass_conservation_residual(
    h: torch.Tensor,
    rainfall_m_s: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    infiltration_m_s: Optional[torch.Tensor] = None,
    config: Optional[ShallowWaterConfig] = None,
    boundary_config: Optional[BoundaryConfig] = None,
) -> torch.Tensor:
    """Compute normalized step-wise mass balance residual [m/s] of shape (B, T-1).

    For each time interval [t_k, t_{k+1}]:
        (V_{k+1} - V_k) / dt = Q_rain_avg - Q_outflow_avg
    Normalized by domain area A = H * W * dx * dy so units are [m/s].
    """
    if h.ndim != 4 or h.shape[1] < 2:
        raise ValueError(f"Expected h shape (B, T, H, W) with T >= 2, got {tuple(h.shape)}")

    cfg = config or ShallowWaterConfig()
    inf_val = torch.zeros_like(rainfall_m_s) if infiltration_m_s is None else infiltration_m_s

    volume = compute_domain_volume(h, dx=cfg.dx, dy=cfg.dy)  # (B, T)
    dv_dt = (volume[:, 1:] - volume[:, :-1]) / cfg.dt        # (B, T-1) [m^3/s]

    net_rain_rate = (rainfall_m_s - inf_val).sum(dim=(-2, -1)) * (cfg.dx * cfg.dy)  # (B, T)
    q_rain_avg = 0.5 * (net_rain_rate[:, 1:] + net_rain_rate[:, :-1])

    if u is not None and v is not None:
        outflow = compute_boundary_outflow_flux(
            h=h,
            u=u,
            v=v,
            dx=cfg.dx,
            dy=cfg.dy,
            boundary_config=boundary_config,
        )
        q_out_avg = 0.5 * (outflow[:, 1:] + outflow[:, :-1])
    else:
        q_out_avg = torch.zeros_like(q_rain_avg)

    domain_area = float(h.shape[-2] * h.shape[-1]) * cfg.dx * cfg.dy
    residual_m_s = (dv_dt - (q_rain_avg - q_out_avg)) / domain_area
    return residual_m_s
