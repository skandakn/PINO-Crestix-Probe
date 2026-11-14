"""2D Shallow Water Equations PDE residual computation in SI units.

Computes:
1. Continuity (mass conservation) residual [m/s]:
   r_mass = dh/dt + d(hu)/dx + d(hv)/dy - (R - I)

2. x-Momentum residual [m^2/s^2]:
   r_mom_x = d(hu)/dt + d(h*u^2 + 0.5*g*h^2)/dx + d(h*u*v)/dy + g*h*(dz_b/dx + S_fx)

3. y-Momentum residual [m^2/s^2]:
   r_mom_y = d(hv)/dt + d(h*u*v)/dx + d(h*v^2 + 0.5*g*h^2)/dy + g*h*(dz_b/dy + S_fy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .shallow_water import (
    ShallowWaterConfig,
    compute_manning_friction_slope,
    compute_wet_mask,
    spatial_derivatives_2d,
    time_derivative,
)


@dataclass
class SWEResiduals:
    """Container for 2D Shallow Water Equation residuals."""

    continuity: torch.Tensor
    momentum_x: torch.Tensor
    momentum_y: torch.Tensor


def compute_swe_residuals(
    h: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dem: torch.Tensor,
    rainfall_m_s: torch.Tensor,
    manning_n: Optional[torch.Tensor | float] = None,
    infiltration_m_s: Optional[torch.Tensor] = None,
    config: Optional[ShallowWaterConfig] = None,
    mask_dry_momentum: bool = True,
) -> SWEResiduals:
    """Evaluate 2D Shallow Water Equation PDE residuals across (B, T, H, W).

    Args:
        h: Water depth tensor of shape (B, T, H, W) in meters [m].
        u: Depth-averaged x-velocity of shape (B, T, H, W) in [m/s].
        v: Depth-averaged y-velocity of shape (B, T, H, W) in [m/s].
        dem: Bed elevation z_b of shape (B, T, H, W) or (B, 1, H, W) or (B, H, W) in [m].
        rainfall_m_s: Rainfall rate R of shape (B, T, H, W) in [m/s].
        manning_n: Manning's roughness coefficient [s * m^(-1/3)], scalar or tensor.
        infiltration_m_s: Optional infiltration rate I of shape (B, T, H, W) in [m/s].
        config: ShallowWaterConfig with physical dx, dy, dt, gravity, wet_dry_eps.
        mask_dry_momentum: If True, zeroes momentum residuals on dry cells (h <= wet_dry_eps)
            where velocity is physically undefined.

    Returns:
        SWEResiduals with continuity [m/s], momentum_x [m^2/s^2], momentum_y [m^2/s^2].
    """
    if h.ndim != 4:
        raise ValueError(f"Expected h shape (B, T, H, W), got {tuple(h.shape)}")
    if u.shape != h.shape or v.shape != h.shape:
        raise ValueError(
            f"Velocity shapes u={tuple(u.shape)}, v={tuple(v.shape)} must match h={tuple(h.shape)}"
        )

    cfg = config or ShallowWaterConfig()
    n_val = cfg.default_manning_n if manning_n is None else manning_n
    inf_val = torch.zeros_like(rainfall_m_s) if infiltration_m_s is None else infiltration_m_s

    if dem.ndim == 3:
        dem_4d = dem.unsqueeze(1).expand_as(h)
    elif dem.ndim == 4 and dem.shape[1] == 1:
        dem_4d = dem.expand_as(h)
    elif dem.ndim == 4 and dem.shape == h.shape:
        dem_4d = dem
    else:
        raise ValueError(f"Incompatible dem shape {tuple(dem.shape)} for h shape {tuple(h.shape)}")

    # Ensure non-negative depth for hydrostatic pressure and friction stability
    h_pos = torch.clamp(h, min=0.0)
    hu = h_pos * u
    hv = h_pos * v

    # 1. Continuity equation: dh/dt + d(hu)/dx + d(hv)/dy - (R - I) = 0
    dh_dt = time_derivative(h_pos, dt=cfg.dt)
    dhu_dx, _ = spatial_derivatives_2d(hu, dx=cfg.dx, dy=cfg.dy)
    _, dhv_dy = spatial_derivatives_2d(hv, dx=cfg.dx, dy=cfg.dy)
    net_source = rainfall_m_s - inf_val
    r_mass = dh_dt + dhu_dx + dhv_dy - net_source

    # 2. Momentum fluxes
    half_g_h2 = 0.5 * cfg.gravity * (h_pos * h_pos)
    flux_xx = h_pos * (u * u) + half_g_h2
    flux_xy = h_pos * (u * v)
    flux_yy = h_pos * (v * v) + half_g_h2

    dhu_dt = time_derivative(hu, dt=cfg.dt)
    dhv_dt = time_derivative(hv, dt=cfg.dt)

    dflux_xx_dx, _ = spatial_derivatives_2d(flux_xx, dx=cfg.dx, dy=cfg.dy)
    _, dflux_xy_dy = spatial_derivatives_2d(flux_xy, dx=cfg.dx, dy=cfg.dy)

    dflux_xy_dx, _ = spatial_derivatives_2d(flux_xy, dx=cfg.dx, dy=cfg.dy)
    _, dflux_yy_dy = spatial_derivatives_2d(flux_yy, dx=cfg.dx, dy=cfg.dy)

    # Bed slope terms: g * h * dz_b/dx, g * h * dz_b/dy
    dz_dx, dz_dy = spatial_derivatives_2d(dem_4d, dx=cfg.dx, dy=cfg.dy)

    # Manning friction slope terms: S_fx, S_fy
    s_fx, s_fy = compute_manning_friction_slope(
        h=h_pos,
        u=u,
        v=v,
        manning_n=n_val,
        wet_dry_eps=cfg.wet_dry_eps,
    )

    r_mom_x = dhu_dt + dflux_xx_dx + dflux_xy_dy + cfg.gravity * h_pos * (dz_dx + s_fx)
    r_mom_y = dhv_dt + dflux_xy_dx + dflux_yy_dy + cfg.gravity * h_pos * (dz_dy + s_fy)

    if mask_dry_momentum:
        wet_mask = compute_wet_mask(h_pos, wet_dry_eps=cfg.wet_dry_eps)
        r_mom_x = r_mom_x * wet_mask
        r_mom_y = r_mom_y * wet_mask

    return SWEResiduals(
        continuity=r_mass,
        momentum_x=r_mom_x,
        momentum_y=r_mom_y,
    )
