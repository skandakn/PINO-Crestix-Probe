import math

import torch

from crestix_pino.physics import (
    BoundaryConfig,
    ShallowWaterConfig,
    compute_boundary_consistency_loss,
    compute_mass_conservation_residual,
    compute_swe_residuals,
    m_per_sec_to_mm_per_hour,
    mm_per_hour_to_m_per_sec,
    mm_to_meters,
)


def test_unit_conversions() -> None:
    # 36 mm/h = 0.036 m / 3600 s = 1e-5 m/s
    r_ms = mm_per_hour_to_m_per_sec(36.0)
    assert math.isclose(r_ms, 1e-5, rel_tol=1e-9)
    assert math.isclose(m_per_sec_to_mm_per_hour(r_ms), 36.0, rel_tol=1e-9)
    assert math.isclose(mm_to_meters(250.0), 0.25, rel_tol=1e-9)


def test_lake_at_rest_well_balanced_residuals() -> None:
    """For a still lake h + z_b = H0 = const with u=v=0, R=0, PDE residuals must vanish in the interior."""
    b, t_dim, h_dim, w_dim = 1, 4, 10, 10
    cfg = ShallowWaterConfig(dx=2.0, dy=2.0, dt=10.0)

    # Smooth sloping bed elevation z_b(y, x)
    y = torch.linspace(0.0, 1.0, h_dim).view(1, 1, h_dim, 1)
    x = torch.linspace(0.0, 1.0, w_dim).view(1, 1, 1, w_dim)
    dem = (1.5 * x + 0.8 * y).expand(b, t_dim, h_dim, w_dim).clone()

    # Constant free-surface elevation H0 = 5.0 m => h = H0 - dem > 0
    h = 5.0 - dem
    u = torch.zeros_like(h)
    v = torch.zeros_like(h)
    rain = torch.zeros_like(h)

    res = compute_swe_residuals(h=h, u=u, v=v, dem=dem, rainfall_m_s=rain, config=cfg)

    assert torch.allclose(res.continuity, torch.zeros_like(res.continuity), atol=1e-6)
    # Interior linear bed slope + hydrostatic pressure gradient cancels exactly
    assert torch.allclose(res.momentum_x[:, :, 1:-1, 1:-1], torch.zeros_like(res.momentum_x[:, :, 1:-1, 1:-1]), atol=1e-4)
    assert torch.allclose(res.momentum_y[:, :, 1:-1, 1:-1], torch.zeros_like(res.momentum_y[:, :, 1:-1, 1:-1]), atol=1e-4)


def test_uniform_rainfall_mass_conservation() -> None:
    """On a flat closed basin with u=v=0 and uniform rainfall R, h(t)=h0+R*t satisfies continuity & mass balance."""
    b, t_dim, h_dim, w_dim = 1, 6, 8, 8
    dt = 60.0
    cfg = ShallowWaterConfig(dx=5.0, dy=5.0, dt=dt)
    bc = BoundaryConfig(north="closed", south="closed", east="closed", west="closed")

    rain_rate = mm_per_hour_to_m_per_sec(72.0)  # 2e-5 m/s
    rain = torch.full((b, t_dim, h_dim, w_dim), rain_rate)
    t_vals = torch.arange(t_dim, dtype=torch.float32).view(1, t_dim, 1, 1) * dt
    h = 0.2 + rain_rate * t_vals.expand(b, t_dim, h_dim, w_dim)
    u = torch.zeros_like(h)
    v = torch.zeros_like(h)
    dem = torch.zeros_like(h)

    res = compute_swe_residuals(h=h, u=u, v=v, dem=dem, rainfall_m_s=rain, config=cfg)
    assert torch.allclose(res.continuity, torch.zeros_like(res.continuity), atol=1e-7)

    mass_res = compute_mass_conservation_residual(
        h=h, rainfall_m_s=rain, u=u, v=v, config=cfg, boundary_config=bc
    )
    assert torch.allclose(mass_res, torch.zeros_like(mass_res), atol=1e-7)


def test_dry_cells_no_nan_singularities() -> None:
    """Zero water depth (dry cells) with non-zero velocity must not yield NaN/Inf in friction or residuals."""
    b, t_dim, h_dim, w_dim = 1, 3, 6, 6
    h = torch.zeros(b, t_dim, h_dim, w_dim)
    u = torch.ones(b, t_dim, h_dim, w_dim) * 0.5
    v = torch.ones(b, t_dim, h_dim, w_dim) * -0.3
    dem = torch.randn(b, h_dim, w_dim)
    rain = torch.zeros(b, t_dim, h_dim, w_dim)

    res = compute_swe_residuals(h=h, u=u, v=v, dem=dem, rainfall_m_s=rain)
    assert torch.isfinite(res.continuity).all()
    assert torch.isfinite(res.momentum_x).all()
    assert torch.isfinite(res.momentum_y).all()
    assert torch.all(res.momentum_x == 0.0)
    assert torch.all(res.momentum_y == 0.0)


def test_boundary_consistency_penalizes_normal_flow_at_closed_wall() -> None:
    b, t_dim, h_dim, w_dim = 1, 3, 6, 6
    bc = BoundaryConfig(north="closed", south="closed", east="closed", west="closed")
    h = torch.ones(b, t_dim, h_dim, w_dim)
    u_zero = torch.zeros_like(h)
    v_zero = torch.zeros_like(h)

    loss_consistent = compute_boundary_consistency_loss(h, u_zero, v_zero, dx=2.0, dy=2.0, boundary_config=bc)
    assert loss_consistent.item() == 0.0

    u_violating = torch.ones_like(h) * 2.0
    loss_violating = compute_boundary_consistency_loss(h, u_violating, v_zero, dx=2.0, dy=2.0, boundary_config=bc)
    assert loss_violating.item() > 0.0
