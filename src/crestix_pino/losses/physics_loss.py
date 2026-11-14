"""Physics-informed loss functions for 2D Shallow Water Equations (PINO).

Combines:
1. Shallow-water PDE residuals (continuity, x-momentum, y-momentum)
2. Global/step-wise mass conservation balance
3. Boundary-condition consistency
4. Nonnegative water depth penalty
5. Initial-condition constraint h(t=0) == h_init
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.pino import PINOOutput
from ..physics.boundary_conditions import BoundaryConfig, compute_boundary_consistency_loss
from ..physics.conservation import compute_mass_conservation_residual
from ..physics.residuals import compute_swe_residuals
from ..physics.shallow_water import ShallowWaterConfig
from .data_loss import DataLoss


@dataclass
class PhysicsLossWeights:
    """Weights for individual physics-informed loss components."""

    pde_continuity: float = 1.0
    pde_momentum_x: float = 0.25
    pde_momentum_y: float = 0.25
    mass_conservation: float = 0.5
    boundary_condition: float = 0.5
    initial_condition: float = 1.0
    nonnegative_depth: float = 1.0


def compute_nonnegative_depth_loss(h: torch.Tensor) -> torch.Tensor:
    """Penalize negative predicted water depths: mean(ReLU(-h)^2) [m^2]."""
    neg = F.relu(-h)
    return (neg * neg).mean()


def compute_initial_condition_loss(
    h_pred: torch.Tensor,
    h_init: torch.Tensor,
) -> torch.Tensor:
    """Enforce initial water-depth condition at t=0: MSE(h_pred[:, 0], h_init) [m^2].

    Args:
        h_pred: Predicted water depth of shape (B, T, H, W).
        h_init: Initial depth of shape (B, H, W) or (B, T, H, W) (uses t=0 slice if 4D).
    """
    h0_pred = h_pred[:, 0, :, :]
    if h_init.ndim == 4:
        h0_true = h_init[:, 0, :, :]
    elif h_init.ndim == 3:
        h0_true = h_init
    else:
        raise ValueError(f"Expected h_init shape (B, H, W) or (B, T, H, W), got {tuple(h_init.shape)}")
    return F.mse_loss(h0_pred, h0_true)


class PhysicsLoss(nn.Module):
    """Composite Shallow-Water Physics-Informed Loss."""

    def __init__(
        self,
        sw_config: Optional[ShallowWaterConfig] = None,
        boundary_config: Optional[BoundaryConfig] = None,
        weights: Optional[PhysicsLossWeights] = None,
    ) -> None:
        super().__init__()
        self.sw_config = sw_config or ShallowWaterConfig()
        self.boundary_config = boundary_config or BoundaryConfig()
        self.weights = weights or PhysicsLossWeights()

    def forward(
        self,
        output: PINOOutput,
        dem: torch.Tensor,
        rainfall_m_s: torch.Tensor,
        h_init: torch.Tensor,
        manning_n: Optional[torch.Tensor | float] = None,
        infiltration_m_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate all physics loss components and return (total_physics_loss, breakdown)."""
        h = output.h
        u = output.u
        v = output.v
        if u is None or v is None:
            raise ValueError(
                "PhysicsLoss requires latent velocity components (u, v) to evaluate "
                "2D Shallow Water PDE residuals. Enable predict_latent_velocity=True."
            )

        # 1. Shallow-Water PDE Residuals (Continuity + x/y Momentum)
        residuals = compute_swe_residuals(
            h=h,
            u=u,
            v=v,
            dem=dem,
            rainfall_m_s=rainfall_m_s,
            manning_n=manning_n,
            infiltration_m_s=infiltration_m_s,
            config=self.sw_config,
        )
        loss_cont = (residuals.continuity ** 2).mean()
        loss_mom_x = (residuals.momentum_x ** 2).mean()
        loss_mom_y = (residuals.momentum_y ** 2).mean()

        # 2. Mass Conservation Loss
        mass_res = compute_mass_conservation_residual(
            h=h,
            rainfall_m_s=rainfall_m_s,
            u=u,
            v=v,
            infiltration_m_s=infiltration_m_s,
            config=self.sw_config,
            boundary_config=self.boundary_config,
        )
        loss_mass = (mass_res ** 2).mean()

        # 3. Boundary-Condition Consistency Loss
        loss_bc = compute_boundary_consistency_loss(
            h=h,
            u=u,
            v=v,
            dx=self.sw_config.dx,
            dy=self.sw_config.dy,
            boundary_config=self.boundary_config,
        )

        # 4. Initial-Condition Constraint Loss
        loss_ic = compute_initial_condition_loss(h_pred=h, h_init=h_init)

        # 5. Nonnegative Water Depth Penalty (evaluated on h_raw if present, else h)
        depth_for_neg = output.h_raw if output.h_raw is not None else h
        loss_nonneg = compute_nonnegative_depth_loss(depth_for_neg)

        w = self.weights
        total = (
            w.pde_continuity * loss_cont
            + w.pde_momentum_x * loss_mom_x
            + w.pde_momentum_y * loss_mom_y
            + w.mass_conservation * loss_mass
            + w.boundary_condition * loss_bc
            + w.initial_condition * loss_ic
            + w.nonnegative_depth * loss_nonneg
        )

        breakdown = {
            "pde_continuity": loss_cont.detach(),
            "pde_momentum_x": loss_mom_x.detach(),
            "pde_momentum_y": loss_mom_y.detach(),
            "mass_conservation": loss_mass.detach(),
            "boundary_condition": loss_bc.detach(),
            "initial_condition": loss_ic.detach(),
            "nonnegative_depth": loss_nonneg.detach(),
            "total_physics": total.detach(),
        }
        return total, breakdown


class PINOCompositeLoss(nn.Module):
    """Combined Supervised Data Loss + Physics-Informed Shallow-Water Loss."""

    def __init__(
        self,
        data_loss: Optional[DataLoss] = None,
        physics_loss: Optional[PhysicsLoss] = None,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_loss = data_loss or DataLoss(loss_type="mse")
        self.physics_loss = physics_loss or PhysicsLoss()
        self.data_weight = data_weight
        self.physics_weight = physics_weight

    def forward(
        self,
        output: PINOOutput,
        h_true: torch.Tensor,
        dem: torch.Tensor,
        rainfall_m_s: torch.Tensor,
        h_init: torch.Tensor,
        manning_n: Optional[torch.Tensor | float] = None,
        infiltration_m_s: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_data = self.data_loss(output.h, h_true, valid_mask=valid_mask)
        l_phys, phys_breakdown = self.physics_loss(
            output=output,
            dem=dem,
            rainfall_m_s=rainfall_m_s,
            h_init=h_init,
            manning_n=manning_n,
            infiltration_m_s=infiltration_m_s,
        )
        total = self.data_weight * l_data + self.physics_weight * l_phys
        breakdown = {
            "data_loss": l_data.detach(),
            **phys_breakdown,
            "total_loss": total.detach(),
        }
        return total, breakdown
