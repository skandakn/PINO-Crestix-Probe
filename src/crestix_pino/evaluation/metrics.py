"""Evaluation metrics for PINO flood water-depth predictions against LarNO ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np

MM_PER_HOUR_TO_M_PER_S = 1e-3 / 3600.0


def _as_float_array(arr: Any, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
    return out


def _validate_pred_target(pred: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = _as_float_array(pred, "pred")
    target_arr = _as_float_array(target, "target")
    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"Shape mismatch between pred {pred_arr.shape} and target {target_arr.shape}."
        )
    if pred_arr.ndim not in (3, 4):
        raise ValueError(
            f"Expected 3D (T, H, W) or 4D (N, T, H, W) arrays, got shape {pred_arr.shape}."
        )
    return pred_arr, target_arr


def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Normalize (T, H, W) to (1, T, H, W) for uniform batch processing."""
    if arr.ndim == 3:
        return arr[np.newaxis, ...]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}.")


def _broadcast_rainfall_m_per_s(
    rainfall_mm_per_hr: Any,
    batch_size: int,
    steps: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Convert rainfall in mm/h to (N, T, H, W) in m/s without silent unit assumptions."""
    rain = _as_float_array(rainfall_mm_per_hr, "rainfall_mm_per_hr")
    if np.any(rain < 0):
        raise ValueError("rainfall_mm_per_hr must be nonnegative.")

    target_shape = (batch_size, steps, height, width)
    if rain.ndim == 0:
        rain_4d = np.broadcast_to(rain, target_shape)
    elif rain.ndim == 1 and rain.shape == (steps,):
        rain_4d = np.broadcast_to(rain[np.newaxis, :, np.newaxis, np.newaxis], target_shape)
    elif rain.ndim == 2 and rain.shape == (height, width):
        rain_4d = np.broadcast_to(rain[np.newaxis, np.newaxis, :, :], target_shape)
    elif rain.ndim == 3 and rain.shape == (steps, height, width):
        rain_4d = np.broadcast_to(rain[np.newaxis, :, :, :], target_shape)
    elif rain.ndim == 4 and rain.shape == target_shape:
        rain_4d = rain
    else:
        raise ValueError(
            f"Incompatible rainfall shape {rain.shape} for target shape {target_shape}."
        )
    return rain_4d * MM_PER_HOUR_TO_M_PER_S


def compute_mae(pred: Any, target: Any) -> float:
    """Compute Mean Absolute Error (meters) between predicted and target water depth."""
    pred_arr, target_arr = _validate_pred_target(pred, target)
    return float(np.mean(np.abs(pred_arr - target_arr)))


def compute_rmse(pred: Any, target: Any) -> float:
    """Compute Root Mean Squared Error (meters) between predicted and target water depth."""
    pred_arr, target_arr = _validate_pred_target(pred, target)
    return float(np.sqrt(np.mean((pred_arr - target_arr) ** 2)))


def compute_r2(pred: Any, target: Any) -> float:
    """Compute coefficient of determination (R^2) over all grid cells and timesteps."""
    pred_arr, target_arr = _validate_pred_target(pred, target)
    ss_res = float(np.sum((target_arr - pred_arr) ** 2))
    target_mean = float(np.mean(target_arr))
    ss_tot = float(np.sum((target_arr - target_mean) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("nan")
    return 1.0 - (ss_res / ss_tot)


def compute_relative_depth_error(
    pred: Any,
    target: Any,
    wet_threshold: float = 1e-3,
    eps: float = 1e-8,
) -> dict[str, float | int | None]:
    """Compute global relative L2 error and mean relative error on wet ground-truth cells."""
    pred_arr, target_arr = _validate_pred_target(pred, target)
    diff = pred_arr - target_arr
    target_norm = float(np.linalg.norm(target_arr))
    rel_l2 = float(np.linalg.norm(diff) / max(target_norm, eps))

    wet_mask = target_arr > wet_threshold
    wet_count = int(np.sum(wet_mask))
    if wet_count > 0:
        wet_mre: float | None = float(
            np.mean(np.abs(diff[wet_mask]) / np.maximum(target_arr[wet_mask], eps))
        )
    else:
        wet_mre = None

    return {
        "relative_l2_error": rel_l2,
        "wet_mean_relative_error": wet_mre,
        "wet_cell_count": wet_count,
        "wet_threshold_m": float(wet_threshold),
    }


def compute_nonnegative_violations(pred: Any, tol: float = 0.0) -> dict[str, float | int]:
    """Count and measure negative water-depth predictions below -tol (meters)."""
    pred_arr = _as_float_array(pred, "pred")
    viol_mask = pred_arr < -tol
    violation_count = int(np.sum(viol_mask))
    total_cells = int(pred_arr.size)
    min_depth = float(np.min(pred_arr))
    mean_neg = float(np.mean(-pred_arr[viol_mask])) if violation_count > 0 else 0.0
    return {
        "violation_count": violation_count,
        "total_cells": total_cells,
        "violation_fraction": float(violation_count / total_cells) if total_cells > 0 else 0.0,
        "min_depth_m": min_depth,
        "mean_negative_magnitude_m": mean_neg,
    }


def compute_mass_conservation_error(
    pred: Any,
    initial_water_depth: Any,
    rainfall_mm_per_hr: Any,
    dt_seconds: float,
    cell_size_m: float,
    target: Any | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute closed-domain water volume conservation error across forecast steps.

    Compares predicted domain volume V_pred(t) = sum(h_pred(t)) * cell_area against
    expected volume V_expected(t) = V_0 + cumulative_rainfall_volume(t).
    """
    if dt_seconds <= 0 or cell_size_m <= 0:
        raise ValueError("dt_seconds and cell_size_m must be positive.")

    pred_4d = _ensure_4d(_as_float_array(pred, "pred"))
    n, t_steps, h, w = pred_4d.shape
    cell_area = float(cell_size_m) ** 2

    h0 = _as_float_array(initial_water_depth, "initial_water_depth")
    if h0.ndim == 2 and h0.shape == (h, w):
        h0_4d = np.broadcast_to(h0[np.newaxis, :, :], (n, h, w))
    elif h0.ndim == 3 and h0.shape == (n, h, w):
        h0_4d = h0
    else:
        raise ValueError(
            f"Incompatible initial_water_depth shape {h0.shape} for pred shape {pred_4d.shape}."
        )

    rain_m_s = _broadcast_rainfall_m_per_s(rainfall_mm_per_hr, n, t_steps, h, w)
    v0 = np.sum(h0_4d, axis=(1, 2)) * cell_area  # (N,)
    step_rain_vol = np.sum(rain_m_s, axis=(2, 3)) * dt_seconds * cell_area  # (N, T)
    expected_vol = v0[:, np.newaxis] + np.cumsum(step_rain_vol, axis=1)  # (N, T)
    pred_vol = np.sum(pred_4d, axis=(2, 3)) * cell_area  # (N, T)

    abs_err = np.abs(pred_vol - expected_vol)
    rel_err = abs_err / np.maximum(np.abs(expected_vol), eps)

    result: dict[str, Any] = {
        "mean_abs_volume_error_m3": float(np.mean(abs_err)),
        "max_abs_volume_error_m3": float(np.max(abs_err)),
        "mean_relative_volume_error": float(np.mean(rel_err)),
        "final_relative_volume_error": float(np.mean(rel_err[:, -1])),
        "per_step_relative_volume_error": [float(x) for x in np.mean(rel_err, axis=0)],
    }

    if target is not None:
        target_4d = _ensure_4d(_as_float_array(target, "target"))
        if target_4d.shape != pred_4d.shape:
            raise ValueError(
                f"target shape {target_4d.shape} does not match pred shape {pred_4d.shape}."
            )
        target_vol = np.sum(target_4d, axis=(2, 3)) * cell_area
        vs_target_rel = np.abs(pred_vol - target_vol) / np.maximum(np.abs(target_vol), eps)
        result["vs_target_mean_relative_volume_error"] = float(np.mean(vs_target_rel))
        result["vs_target_final_relative_volume_error"] = float(np.mean(vs_target_rel[:, -1]))

    return result


def compute_physics_residual(
    pred: Any,
    dem: Any | None = None,
    rainfall_mm_per_hr: Any | None = None,
    dt_seconds: float = 30.0,
    cell_size_m: float = 10.0,
    manning_n: float = 0.035,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Compute depth-only shallow-water continuity / diffusive-wave PDE residual (m/s).

    Does not fabricate velocity outputs. When `dem` is provided, computes the
    diffusive-wave continuity residual:
        R_pde = dh/dt + div(q) - rain
    with q = -(1/n) * h_pos^(5/3) * grad(eta) / sqrt(max(|grad(eta)|, eps)),
    where eta = dem + h_pos on interior spatial cells.
    When `dem` is None, computes the storage-rainfall residual dh/dt - rain.
    """
    if dt_seconds <= 0 or cell_size_m <= 0 or manning_n <= 0:
        raise ValueError("dt_seconds, cell_size_m, and manning_n must be positive.")

    pred_4d = _ensure_4d(_as_float_array(pred, "pred"))
    n, t_steps, h, w = pred_4d.shape
    if t_steps < 2:
        raise ValueError("Need at least 2 forecast steps to compute temporal derivative dh/dt.")

    dh_dt = (pred_4d[:, 1:] - pred_4d[:, :-1]) / float(dt_seconds)  # (N, T-1, H, W)
    if rainfall_mm_per_hr is not None:
        rain_m_s = _broadcast_rainfall_m_per_s(rainfall_mm_per_hr, n, t_steps, h, w)[:, 1:]
    else:
        rain_m_s = np.zeros_like(dh_dt)

    if dem is not None and h >= 3 and w >= 3:
        dem_arr = _as_float_array(dem, "dem")
        if dem_arr.ndim == 2 and dem_arr.shape == (h, w):
            dem_4d = dem_arr[np.newaxis, np.newaxis, :, :]
        elif dem_arr.ndim == 3 and dem_arr.shape == (n, h, w):
            dem_4d = dem_arr[:, np.newaxis, :, :]
        else:
            raise ValueError(f"Incompatible dem shape {dem_arr.shape} for pred shape {pred_4d.shape}.")

        h_mid = np.maximum(0.5 * (pred_4d[:, 1:] + pred_4d[:, :-1]), 0.0)
        eta = dem_4d + h_mid
        d_eta_dy, d_eta_dx = np.gradient(eta, float(cell_size_m), axis=(2, 3))
        slope_mag = np.sqrt(d_eta_dx ** 2 + d_eta_dy ** 2 + eps ** 2)
        conveyance = (1.0 / float(manning_n)) * (h_mid ** (5.0 / 3.0)) / np.sqrt(slope_mag)
        qx = -conveyance * d_eta_dx
        qy = -conveyance * d_eta_dy
        dqx_dx = np.gradient(qx, float(cell_size_m), axis=3)
        dqy_dy = np.gradient(qy, float(cell_size_m), axis=2)
        residual = (dh_dt + dqx_dx + dqy_dy - rain_m_s)[:, :, 1:-1, 1:-1]
        formulation = "diffusive_wave_continuity_interior"
    else:
        residual = dh_dt - rain_m_s
        formulation = "storage_rainfall_continuity"

    return {
        "mean_abs_residual_m_per_s": float(np.mean(np.abs(residual))),
        "rmse_residual_m_per_s": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_residual_m_per_s": float(np.max(np.abs(residual))),
        "formulation": formulation,
    }


def compute_horizon_metrics(
    pred: Any,
    target: Any,
    wet_threshold: float = 1e-3,
) -> list[dict[str, Any]]:
    """Compute evaluation metrics at each forecast horizon step t = 0..T-1."""
    pred_arr, target_arr = _validate_pred_target(pred, target)
    pred_4d = _ensure_4d(pred_arr)
    target_4d = _ensure_4d(target_arr)

    steps = pred_4d.shape[1]
    horizon: list[dict[str, Any]] = []
    for t in range(steps):
        p_t = pred_4d[:, t : t + 1, :, :]
        g_t = target_4d[:, t : t + 1, :, :]
        rel = compute_relative_depth_error(p_t, g_t, wet_threshold=wet_threshold)
        viol = compute_nonnegative_violations(p_t)
        horizon.append(
            {
                "step": t + 1,
                "mae_m": compute_mae(p_t, g_t),
                "rmse_m": compute_rmse(p_t, g_t),
                "r2": compute_r2(p_t, g_t),
                "relative_l2_error": rel["relative_l2_error"],
                "wet_mean_relative_error": rel["wet_mean_relative_error"],
                "nonnegative_violation_fraction": viol["violation_fraction"],
            }
        )
    return horizon


def evaluate_predictions(
    pred: Any,
    target: Any,
    dem: Any | None = None,
    rainfall_mm_per_hr: Any | None = None,
    initial_water_depth: Any | None = None,
    dt_seconds: float = 30.0,
    cell_size_m: float = 10.0,
    manning_n: float = 0.035,
    wet_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Compute the full Prompt 6 metric suite for predicted vs. LarNO ground-truth grids."""
    pred_arr, target_arr = _validate_pred_target(pred, target)

    summary: dict[str, Any] = {
        "mae_m": compute_mae(pred_arr, target_arr),
        "rmse_m": compute_rmse(pred_arr, target_arr),
        "r2": compute_r2(pred_arr, target_arr),
        "relative_depth_error": compute_relative_depth_error(
            pred_arr, target_arr, wet_threshold=wet_threshold
        ),
        "nonnegative_violations": compute_nonnegative_violations(pred_arr),
        "horizon_metrics": compute_horizon_metrics(
            pred_arr, target_arr, wet_threshold=wet_threshold
        ),
    }

    if initial_water_depth is not None and rainfall_mm_per_hr is not None:
        summary["mass_conservation"] = compute_mass_conservation_error(
            pred=pred_arr,
            initial_water_depth=initial_water_depth,
            rainfall_mm_per_hr=rainfall_mm_per_hr,
            dt_seconds=dt_seconds,
            cell_size_m=cell_size_m,
            target=target_arr,
        )
    else:
        summary["mass_conservation"] = None

    if pred_arr.shape[-3] >= 2:
        summary["physics_residual"] = compute_physics_residual(
            pred=pred_arr,
            dem=dem,
            rainfall_mm_per_hr=rainfall_mm_per_hr,
            dt_seconds=dt_seconds,
            cell_size_m=cell_size_m,
            manning_n=manning_n,
        )
    else:
        summary["physics_residual"] = None

    return summary
