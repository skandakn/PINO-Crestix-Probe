"""Unit tests for PINO physics validation, dimensional consistency, and diagnostic tools."""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics_validation.pino_physics_validator import (
    GRAVITY_M_S2,
    MM_PER_HOUR_TO_M_PER_SEC,
    BoundaryConfig,
    PhysicsLossWeights,
    ShallowWaterConfig,
    compute_boundary_consistency_loss,
    compute_mass_conservation_residual,
    compute_nonnegative_depth_loss,
    compute_swe_residuals,
    compute_swe_residuals_masked,
    load_evaluation_metrics_module,
    m_per_sec_to_mm_per_hour,
    mm_per_hour_to_m_per_sec,
    mm_to_meters,
    preprocess_dem_numpy,
    run_loss_and_gradient_analysis,
    run_synthetic_benchmarks,
    softplus_beta5,
    spatial_derivatives_2d,
    time_derivative,
)
from physics_validation.svg_plots import (
    plot_loss_and_gradient_dominance,
    plot_softplus_negative_paradox,
    plot_synthetic_benchmarks,
)


def test_si_unit_conversions_and_roundtrip() -> None:
    assert pytest.approx(1e-5, rel=1e-12) == mm_per_hour_to_m_per_sec(36.0)
    assert pytest.approx(MM_PER_HOUR_TO_M_PER_SEC, rel=1e-12) == mm_per_hour_to_m_per_sec(1.0)
    assert pytest.approx(72.5, rel=1e-12) == m_per_sec_to_mm_per_hour(mm_per_hour_to_m_per_sec(72.5))
    assert pytest.approx(0.5, rel=1e-12) == mm_to_meters(500.0)


def test_derivative_scale_accuracy_and_mismatch_effect() -> None:
    t = np.arange(6, dtype=np.float64) * 300.0  # UKEA 5-min step
    y = np.arange(8, dtype=np.float64) * 8.0    # UKEA 8m cell size
    x = np.arange(8, dtype=np.float64) * 8.0
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    field = 0.002 * tt + 0.05 * yy - 0.03 * xx

    dt_true = time_derivative(field, dt=300.0)
    assert np.allclose(dt_true[1:-1], 0.002, atol=1e-12)

    # Evaluating UKEA data with default ShallowWaterConfig dt=60s amplifies dh/dt by 5x
    dt_wrong = time_derivative(field, dt=60.0)
    assert pytest.approx(5.0, rel=1e-9) == float(np.mean(dt_wrong[1:-1]) / np.mean(dt_true[1:-1]))

    dx_field, dy_field = spatial_derivatives_2d(field[np.newaxis, ...], dx=8.0, dy=8.0)
    assert np.allclose(dy_field[..., 1:-1, 1:-1], 0.05, atol=1e-12)
    assert np.allclose(dx_field[..., 1:-1, 1:-1], -0.03, atol=1e-12)

    # Verify evaluation/metrics.py diffusive-wave scaling when ModelContract defaults (dx=10, dt=30)
    # are mistakenly used instead of UKEA (dx=8, dt=300): div(q) scales by (8/10)^1.5
    eval_metrics = load_evaluation_metrics_module(REPO_ROOT)
    dem = 0.02 * yy[0] + 0.01 * xx[0] + 0.03 * np.sin(yy[0] / 16.0)
    h = np.broadcast_to((0.15 + 0.05 * np.sin(xx[0] / 12.0))[np.newaxis, np.newaxis, :, :], (1, 4, 8, 8)).copy()
    rain = np.zeros_like(h)
    res_ukea = eval_metrics.compute_physics_residual(h, dem=dem, rainfall_mm_per_hr=rain, dt_seconds=300.0, cell_size_m=8.0)
    res_default = eval_metrics.compute_physics_residual(h, dem=dem, rainfall_mm_per_hr=rain, dt_seconds=30.0, cell_size_m=10.0)
    expected_ratio = (8.0 / 10.0) ** 1.5
    actual_ratio = res_default["rmse_residual_m_per_s"] / res_ukea["rmse_residual_m_per_s"]
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-2)


def test_s1_through_s7_synthetic_benchmarks() -> None:
    synth = run_synthetic_benchmarks()

    # S1: Lake at rest exact on linear bed, first-order edge / second-order quadratic truncation
    s1 = synth["S1_lake_at_rest"]
    assert s1["linear_continuity_max_abs"] < 1e-12
    assert s1["linear_mom_x_interior_max_abs"] < 1e-11
    assert s1["linear_mom_x_edge_max_abs"] > 1e-2
    assert s1["quadratic_mom_x_interior_max_abs"] > 1e-2

    # S2: Still pond under uniform rainfall in closed basin
    s2 = synth["S2_uniform_rainfall_closed_basin"]
    assert s2["continuity_max_abs_m_s"] < 1e-15
    assert s2["mass_balance_max_abs_m_s"] < 1e-15

    # S3: Steady uniform Manning equilibrium
    s3 = synth["S3_steady_manning_channel"]
    assert s3["continuity_max_abs_m_s"] < 1e-12
    assert s3["momentum_x_max_abs_m2_s2"] < 1e-11

    # S4: Wet-dry epsilon stability & jump discontinuity near 1e-3 m
    s4 = synth["S4_dry_bed_and_stiffness"]
    assert s4["dry_all_finite"] is True
    assert s4["dry_momentum_zero"] is True
    assert s4["drag_at_threshold_plus_m2_s2"] > 0.05
    assert s4["max_sensitivity_near_threshold"] > 20.0

    # S5: softplus(beta=5) + ReLU(-h_raw)^2 dry-cell positive trapping paradox
    s5 = synth["S5_negative_depth_and_softplus_paradox"]
    assert s5["softplus_at_h_raw_0_m"] == pytest.approx(np.log(2.0) / 5.0, rel=1e-6)
    assert s5["trapped_dry_depth_m_w1"] > 0.10
    assert s5["pde_continuity_grad_norm_when_h_negative"] == 0.0
    assert s5["nonneg_penalty_grad_norm_when_h_negative"] > 0.01

    # S6: +50m building NaN fill wall momentum explosion & adjacent street gradient leakage
    s6 = synth["S6_building_50m_wall_spike"]
    assert s6["building_cells"] == 64
    assert s6["unmasked_momentum_sq_mean"] > 1.0
    assert s6["adjacent_street_unmasked_grad_mom_sq_mean"] > 0.4
    assert s6["masked_momentum_sq_mean"] == pytest.approx(0.0, abs=1e-15)

    # S7: Open-boundary latent velocity gauge freedom can cheat mass balance with unphysical h
    s7 = synth["S7_open_boundary_gauge_freedom"]
    assert s7["true_volume_gain_expected_m3"] > 0.0
    assert s7["wrong_volume_change_m3"] < 0.0
    assert s7["can_fake_zero_residual_with_wrong_depth"] is True


def test_boundary_loss_dimensional_mismatch_and_eval_closed_domain_failure() -> None:
    cfg = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    bc = BoundaryConfig(north="closed", south="open", east="closed", west="closed")
    h = np.full((1, 4, 8, 8), 0.2, dtype=np.float64)
    u = np.full((1, 4, 8, 8), 0.1, dtype=np.float64)
    v = np.zeros_like(h)
    mean_loss, breakdown = compute_boundary_consistency_loss(h, u=u, v=v, dx=cfg.dx, dy=cfg.dy, boundary_config=bc)
    assert mean_loss > 0.0
    assert breakdown["east_u_sq"][1] == "m^2/s^2"
    assert breakdown["east_dh_dn_sq"][1] == "dimensionless [1]"
    assert breakdown["south_dv_dn_sq"][1] == "1/s^2"
    assert breakdown["east_u_sq"][0] > 0.0 and breakdown["west_u_sq"][0] > 0.0

    # Closed-domain metric in evaluation/metrics.py flags true open sub-catchment runoff as huge error
    eval_metrics = load_evaluation_metrics_module(REPO_ROOT)
    # 50% of incoming rain leaves the open boundary
    rain_mm_h = np.full((1, 5, 8, 8), 36.0, dtype=np.float64)
    rain_m_s = mm_per_hour_to_m_per_sec(rain_mm_h)
    h_open = np.zeros_like(rain_m_s)
    for k in range(1, 5):
        h_open[:, k] = h_open[:, k - 1] + 0.5 * rain_m_s[:, k] * 300.0
    mc = eval_metrics.compute_mass_conservation_error(
        pred=h_open,
        initial_water_depth=h_open[:, 0],
        rainfall_mm_per_hr=rain_mm_h,
        dt_seconds=300.0,
        cell_size_m=8.0,
    )
    assert mc["final_relative_volume_error"] > 0.35


def test_loss_and_gradient_dominance_and_svg_exports(tmp_path: Path) -> None:
    # Offline synthetic run ensures deterministic execution without network
    loss_grad = run_loss_and_gradient_analysis()
    gu = loss_grad["grad_norms_unmasked_ukea"]
    # Continuity gradient is orders of magnitude weaker than momentum & data
    assert gu["pde_continuity"] < 1e-2 * gu["data_loss"]

    # On building-NaN DEM, unmasked physics gradient vastly exceeds masked physics gradient
    assert (
        loss_grad["gradient_dominance_ratio_bldg_unmasked_phys_over_data"]
        > 5.0 * loss_grad["gradient_dominance_ratio_bldg_masked_phys_over_data"]
    )

    # Corrected optimization drives dry-cell depth far lower than unmasked h_raw trapping
    tu = loss_grad["optimization_trajectory_unmasked"]
    tc = loss_grad["optimization_trajectory_corrected"]
    assert tc["dry_cell_mean_h"][-1] < 0.02
    assert tc["dry_cell_mean_h"][-1] < 0.25 * tu["dry_cell_mean_h"][-1]

    # Verify standalone SVG generation produces valid SVG XML
    synth = run_synthetic_benchmarks()
    p1 = plot_synthetic_benchmarks(synth, tmp_path / "synth.svg")
    p2 = plot_loss_and_gradient_dominance(loss_grad, tmp_path / "loss.svg")
    p3 = plot_softplus_negative_paradox(synth, loss_grad, tmp_path / "paradox.svg")
    for p in (p1, p2, p3):
        assert p.exists() and p.stat().st_size > 500
        root = ET.fromstring(p.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")
