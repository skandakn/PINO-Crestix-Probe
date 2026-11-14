"""Supervised data loss for water-depth predictions against LarNO UKEA labels."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DataLossType = Literal["mse", "l1", "rel_l2"]


class DataLoss(nn.Module):
    """Supervised loss on predicted water depth h [m] vs observed/target depth h_true [m].

    Supports optional spatial-temporal valid masks for handling nodata/outside-catchment cells.
    """

    def __init__(self, loss_type: DataLossType = "mse", eps: float = 1e-8) -> None:
        super().__init__()
        if loss_type not in {"mse", "l1", "rel_l2"}:
            raise ValueError(f"Unsupported loss_type='{loss_type}'. Expected 'mse', 'l1', or 'rel_l2'.")
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        h_pred: torch.Tensor,
        h_true: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h_pred.shape != h_true.shape:
            raise ValueError(
                f"Shape mismatch: h_pred={tuple(h_pred.shape)} vs h_true={tuple(h_true.shape)}"
            )

        if valid_mask is not None:
            mask = valid_mask.to(dtype=h_pred.dtype, device=h_pred.device)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1).expand_as(h_pred)
            elif mask.shape != h_pred.shape:
                mask = mask.expand_as(h_pred)
            denom = torch.clamp(mask.sum(), min=1.0)
        else:
            mask = None
            denom = None

        if self.loss_type == "mse":
            sq_err = (h_pred - h_true) ** 2
            if mask is not None and denom is not None:
                return (sq_err * mask).sum() / denom
            return sq_err.mean()

        if self.loss_type == "l1":
            abs_err = torch.abs(h_pred - h_true)
            if mask is not None and denom is not None:
                return (abs_err * mask).sum() / denom
            return abs_err.mean()

        # Relative L2 loss averaged across batch
        diff = (h_pred - h_true) * (mask if mask is not None else 1.0)
        ref = h_true * (mask if mask is not None else 1.0)
        flat_diff = diff.reshape(diff.shape[0], -1)
        flat_ref = ref.reshape(ref.shape[0], -1)
        rel = torch.linalg.norm(flat_diff, ord=2, dim=1) / (
            torch.linalg.norm(flat_ref, ord=2, dim=1) + self.eps
        )
        return rel.mean()
