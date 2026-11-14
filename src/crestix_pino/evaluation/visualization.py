"""Visualization utilities for PINO spatial error maps, time series, and horizon plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crestix_pino.evaluation.metrics import _broadcast_rainfall_m_per_s


def _select_2d_slice(arr: Any, step_idx: int, sample_idx: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 4:
        return data[sample_idx, step_idx]
    if data.ndim == 3:
        return data[step_idx]
    raise ValueError(f"Expected 3D (T, H, W) or 4D (N, T, H, W) array, got shape {data.shape}.")


def plot_spatial_error_map(
    pred: Any,
    target: Any,
    output_path: str | Path,
    step_idx: int = -1,
    sample_idx: int = 0,
    title_prefix: str = "PINO vs LarNO",
) -> Path:
    """Save a 4-panel spatial comparison: LarNO, PINO, signed error, and absolute error."""
    pred_2d = _select_2d_slice(pred, step_idx, sample_idx)
    gt_2d = _select_2d_slice(target, step_idx, sample_idx)
    if pred_2d.shape != gt_2d.shape:
        raise ValueError(f"Shape mismatch: pred {pred_2d.shape} vs target {gt_2d.shape}.")

    signed_err = pred_2d - gt_2d
    abs_err = np.abs(signed_err)

    vmax_depth = float(max(np.max(gt_2d), np.max(pred_2d), 1e-6))
    vmax_signed = float(max(np.max(np.abs(signed_err)), 1e-6))
    vmax_abs = float(max(np.max(abs_err), 1e-6))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)

    panels = [
        (gt_2d, "LarNO Ground Truth (m)", "Blues", 0.0, vmax_depth),
        (pred_2d, "PINO Prediction (m)", "Blues", 0.0, vmax_depth),
        (signed_err, "Signed Error (PINO - LarNO, m)", "RdBu_r", -vmax_signed, vmax_signed),
        (abs_err, "Absolute Error (m)", "inferno", 0.0, vmax_abs),
    ]
    for ax, (grid, label, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("X cell")
        ax.set_ylabel("Y cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    step_label = step_idx if step_idx >= 0 else f"final ({step_idx})"
    fig.suptitle(f"{title_prefix} — Step {step_label}", fontsize=12, fontweight="bold")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_time_series_comparison(
    pred: Any,
    target: Any,
    output_path: str | Path,
    sample_idx: int = 0,
    dt_seconds: float = 30.0,
    cell_size_m: float = 10.0,
    initial_water_depth: Any | None = None,
    rainfall_mm_per_hr: Any | None = None,
) -> Path:
    """Save time-series comparison of mean depth, max depth, and total water volume."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    gt_arr = np.asarray(target, dtype=np.float64)
    if pred_arr.ndim == 4:
        pred_arr = pred_arr[sample_idx]
        gt_arr = gt_arr[sample_idx]
    if pred_arr.shape != gt_arr.shape or pred_arr.ndim != 3:
        raise ValueError(
            f"Expected matching 3D (T, H, W) arrays after sample selection, "
            f"got {pred_arr.shape} and {gt_arr.shape}."
        )

    steps, h, w = pred_arr.shape
    times_min = np.arange(1, steps + 1, dtype=np.float64) * (dt_seconds / 60.0)
    cell_area = float(cell_size_m) ** 2

    pred_mean = np.mean(pred_arr, axis=(1, 2))
    gt_mean = np.mean(gt_arr, axis=(1, 2))
    pred_max = np.max(pred_arr, axis=(1, 2))
    gt_max = np.max(gt_arr, axis=(1, 2))
    pred_vol = np.sum(pred_arr, axis=(1, 2)) * cell_area
    gt_vol = np.sum(gt_arr, axis=(1, 2)) * cell_area

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    axes[0].plot(times_min, gt_mean, label="LarNO Ground Truth", linewidth=2)
    axes[0].plot(times_min, pred_mean, "--", label="PINO Prediction", linewidth=2)
    axes[0].set_title("Domain Mean Depth (m)")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(times_min, gt_max, label="LarNO Ground Truth", linewidth=2)
    axes[1].plot(times_min, pred_max, "--", label="PINO Prediction", linewidth=2)
    axes[1].set_title("Domain Max Depth (m)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Depth (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(times_min, gt_vol, label="LarNO Ground Truth", linewidth=2)
    axes[2].plot(times_min, pred_vol, "--", label="PINO Prediction", linewidth=2)
    if initial_water_depth is not None and rainfall_mm_per_hr is not None:
        h0 = np.asarray(initial_water_depth, dtype=np.float64)
        if h0.ndim == 3:
            h0 = h0[sample_idx]
        rain_m_s = _broadcast_rainfall_m_per_s(rainfall_mm_per_hr, 1, steps, h, w)[0]
        v0 = float(np.sum(h0)) * cell_area
        rain_vol = np.sum(rain_m_s, axis=(1, 2)) * dt_seconds * cell_area
        expected_vol = v0 + np.cumsum(rain_vol)
        axes[2].plot(times_min, expected_vol, ":", label="Closed-Domain Mass Balance", linewidth=1.8)
    axes[2].set_title("Total Water Volume (m^3)")
    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Volume (m^3)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_horizon_metrics(
    horizon_metrics: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot MAE, RMSE, R^2, and Relative L2 Error across forecast horizon steps."""
    if not horizon_metrics:
        raise ValueError("horizon_metrics must be a non-empty list.")

    steps = [m["step"] for m in horizon_metrics]
    mae = [m["mae_m"] for m in horizon_metrics]
    rmse = [m["rmse_m"] for m in horizon_metrics]
    r2 = [m["r2"] for m in horizon_metrics]
    rel_l2 = [m["relative_l2_error"] for m in horizon_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    series = [
        (axes[0, 0], mae, "MAE vs Horizon", "MAE (m)"),
        (axes[0, 1], rmse, "RMSE vs Horizon", "RMSE (m)"),
        (axes[1, 0], r2, "R^2 vs Horizon", "R^2"),
        (axes[1, 1], rel_l2, "Relative L2 Error vs Horizon", "Relative L2 Error"),
    ]
    for ax, vals, title, ylabel in series:
        ax.plot(steps, vals, marker="o", linewidth=1.8, markersize=4)
        ax.set_title(title)
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
