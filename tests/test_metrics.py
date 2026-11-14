"""Unit tests for PINO evaluation metrics."""

import math
import numpy as np
import pytest

from crestix_pino.evaluation.metrics import (
    MM_PER_HOUR_TO_M_PER_S,
    compute_horizon_metrics,
    compute_mae,
    compute_mass_conservation_error,
    compute_nonnegative_violations,
    compute_physics_residual,
    compute_r2,
    compute_relative_depth_error,
    compute_rmse,
    evaluate_predictions,
)


def test_basic_error_metrics_exact_and_perturbed():
    target = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float64)
    pred = target + 0.5
    assert math.isclose(compute_mae(pred, target), 0.5)
    assert math.isclose(compute_rmse(pred, target), 0.5)
    assert math.isclose(compute_r2(target, target), 1.0)
    assert compute_r2(pred, target) < 1.0


def test_relative_depth_error_handles_dry_cells():
    target = np.zeros((2, 3, 3), dtype=np.float64)
    pred = np.zeros((2, 3, 3), dtype=np.float64)
    rel_dry = compute_relative_depth_error(pred, target)
    assert rel_dry["wet_cell_count"] == 0
    assert rel_dry["wet_mean_relative_error"] is None

    target[:, 0, 0] = 2.0
    pred[:, 0, 0] = 1.5
    rel_wet = compute_relative_depth_error(pred, target, wet_threshold=0.1)
    assert rel_wet["wet_cell_count"] == 2
    assert math.isclose(rel_wet["wet_mean_relative_error"], 0.25)


def test_nonnegative_violations():
    pred = np.array([[[-0.2, 0.0], [0.5, -0.1]]], dtype=np.float64)
    stats = compute_nonnegative_violations(pred)
    assert stats["violation_count"] == 2
    assert stats["total_cells"] == 4
    assert math.isclose(stats["violation_fraction"], 0.5)
    assert math.isclose(stats["min_depth_m"], -0.2)
    assert math.isclose(stats["mean_negative_magnitude_m"], 0.15)


def test_mass_conservation_exact_closed_domain():
    steps, h, w = 4, 5, 5
    dt = 60.0
    cell_size = 10.0
    rain_mm_hr = 36.0  # 0.036 m/hr = 1e-5 m/s -> 6e-4 m per 60s step
    dh_per_step = rain_mm_hr * MM_PER_HOUR_TO_M_PER_S * dt

    h0 = np.full((h, w), 0.1, dtype=np.float64)
    pred = np.stack([h0 + (t + 1) * dh_per_step for t in range(steps)], axis=0)

    mass = compute_mass_conservation_error(
        pred=pred,
        initial_water_depth=h0,
        rainfall_mm_per_hr=rain_mm_hr,
        dt_seconds=dt,
        cell_size_m=cell_size,
        target=pred,
    )
    assert mass["mean_abs_volume_error_m3"] < 1e-9
    assert mass["mean_relative_volume_error"] < 1e-9
    assert mass["vs_target_mean_relative_volume_error"] < 1e-9


def test_physics_residual_with_and_without_dem():
    steps, h, w = 3, 6, 6
    dt = 30.0
    rain_mm_hr = 36.0
    dh = rain_mm_hr * MM_PER_HOUR_TO_M_PER_S * dt
    pred = np.stack([np.full((h, w), (t + 1) * dh) for t in range(steps)], axis=0)

    res_no_dem = compute_physics_residual(pred, dem=None, rainfall_mm_per_hr=rain_mm_hr, dt_seconds=dt)
    assert res_no_dem["formulation"] == "storage_rainfall_continuity"
    assert res_no_dem["mean_abs_residual_m_per_s"] < 1e-12

    dem = np.zeros((h, w), dtype=np.float64)
    res_with_dem = compute_physics_residual(pred, dem=dem, rainfall_mm_per_hr=rain_mm_hr, dt_seconds=dt)
    assert res_with_dem["formulation"] == "diffusive_wave_continuity_interior"
    assert res_with_dem["mean_abs_residual_m_per_s"] < 1e-12


def test_horizon_and_aggregate_evaluation():
    target = np.ones((3, 4, 4), dtype=np.float64)
    pred = target.copy()
    pred[0] += 0.1
    pred[1] += 0.2
    pred[2] += 0.3

    horizon = compute_horizon_metrics(pred, target)
    assert len(horizon) == 3
    assert horizon[0]["mae_m"] < horizon[1]["mae_m"] < horizon[2]["mae_m"]

    summary = evaluate_predictions(
        pred=pred,
        target=target,
        dem=np.zeros((4, 4)),
        rainfall_mm_per_hr=10.0,
        initial_water_depth=np.ones((4, 4)),
    )
    assert "mass_conservation" in summary and summary["mass_conservation"] is not None
    assert "physics_residual" in summary and summary["physics_residual"] is not None


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_mae(np.zeros((2, 4, 4)), np.zeros((3, 4, 4)))
