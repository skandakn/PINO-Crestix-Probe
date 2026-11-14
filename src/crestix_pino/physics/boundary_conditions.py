"""Boundary condition definitions and consistency residuals for 2D flood domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

BoundaryType = Literal["closed", "open", "periodic"]


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary condition configuration for the 4 edges of a 2D terrain grid."""

    north: BoundaryType = "closed"
    south: BoundaryType = "open"
    east: BoundaryType = "closed"
    west: BoundaryType = "closed"

    def __post_init__(self) -> None:
        valid = {"closed", "open", "periodic"}
        for side, val in [
            ("north", self.north),
            ("south", self.south),
            ("east", self.east),
            ("west", self.west),
        ]:
            if val not in valid:
                raise ValueError(f"Invalid boundary condition '{val}' for {side}; expected {valid}.")


def compute_boundary_consistency_loss(
    h: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    dx: float = 2.0,
    dy: float = 2.0,
    boundary_config: Optional[BoundaryConfig] = None,
) -> torch.Tensor:
    """Compute physical boundary-condition consistency loss across all 4 domain edges.

    - 'closed' (impermeable wall):
        * Normal velocity = 0 (v=0 on north/south, u=0 on east/west)
        * Zero normal depth gradient (dh/dn = 0)
    - 'open' (free outflow / Neumann zero-gradient):
        * Zero normal depth gradient (dh/dn = 0)
        * Zero normal velocity gradient (du_n/dn = 0)
    - 'periodic':
        * Opposite edges match: f[:, :, 0, :] == f[:, :, -1, :] or col 0 == col -1
    """
    if h.ndim != 4 or h.shape[-2] < 2 or h.shape[-1] < 2:
        raise ValueError(f"Expected h shape (B, T, H, W) with H,W >= 2, got {tuple(h.shape)}")

    bc = boundary_config or BoundaryConfig()
    terms = []

    # North edge (row 0, normal along y)
    if bc.north == "closed":
        dh_dn = (h[..., 1, :] - h[..., 0, :]) / dy
        terms.append((dh_dn * dh_dn).mean())
        if v is not None:
            terms.append((v[..., 0, :] * v[..., 0, :]).mean())
    elif bc.north == "open":
        dh_dn = (h[..., 1, :] - h[..., 0, :]) / dy
        terms.append((dh_dn * dh_dn).mean())
        if v is not None:
            dv_dn = (v[..., 1, :] - v[..., 0, :]) / dy
            terms.append((dv_dn * dv_dn).mean())
    elif bc.north == "periodic":
        diff_h = h[..., 0, :] - h[..., -1, :]
        terms.append((diff_h * diff_h).mean())
        if v is not None:
            diff_v = v[..., 0, :] - v[..., -1, :]
            terms.append((diff_v * diff_v).mean())

    # South edge (row -1, normal along y)
    if bc.south == "closed":
        dh_dn = (h[..., -1, :] - h[..., -2, :]) / dy
        terms.append((dh_dn * dh_dn).mean())
        if v is not None:
            terms.append((v[..., -1, :] - 0.0).pow(2).mean())
    elif bc.south == "open":
        dh_dn = (h[..., -1, :] - h[..., -2, :]) / dy
        terms.append((dh_dn * dh_dn).mean())
        if v is not None:
            dv_dn = (v[..., -1, :] - v[..., -2, :]) / dy
            terms.append((dv_dn * dv_dn).mean())

    # West edge (col 0, normal along x)
    if bc.west == "closed":
        dh_dn = (h[..., :, 1] - h[..., :, 0]) / dx
        terms.append((dh_dn * dh_dn).mean())
        if u is not None:
            terms.append((u[..., :, 0] * u[..., :, 0]).mean())
    elif bc.west == "open":
        dh_dn = (h[..., :, 1] - h[..., :, 0]) / dx
        terms.append((dh_dn * dh_dn).mean())
        if u is not None:
            du_dn = (u[..., :, 1] - u[..., :, 0]) / dx
            terms.append((du_dn * du_dn).mean())
    elif bc.west == "periodic":
        diff_h = h[..., :, 0] - h[..., :, -1]
        terms.append((diff_h * diff_h).mean())
        if u is not None:
            diff_u = u[..., :, 0] - u[..., :, -1]
            terms.append((diff_u * diff_u).mean())

    # East edge (col -1, normal along x)
    if bc.east == "closed":
        dh_dn = (h[..., :, -1] - h[..., :, -2]) / dx
        terms.append((dh_dn * dh_dn).mean())
        if u is not None:
            terms.append((u[..., :, -1] * u[..., :, -1]).mean())
    elif bc.east == "open":
        dh_dn = (h[..., :, -1] - h[..., :, -2]) / dx
        terms.append((dh_dn * dh_dn).mean())
        if u is not None:
            du_dn = (u[..., :, -1] - u[..., :, -2]) / dx
            terms.append((du_dn * du_dn).mean())

    if not terms:
        return torch.zeros((), device=h.device, dtype=h.dtype)

    return torch.stack(terms).mean()
