"""PINO Physics Validation & Diagnostic Analysis Core Module (Crestix Probe).

Faithfully mirrors the finite-difference Shallow Water Equations (SWE) physics engine,
boundary conditions, mass conservation, and composite physics loss from
`src/crestix_pino/` in pure NumPy (with optional PyTorch cross-check when available),
without altering shared model files.

Includes:
- SI unit conversions & scale audits
- Analytical & masked SWE residuals, Manning friction, mass balance, and boundary losses
- 7 analytical synthetic verification & edge-case stress benchmarks
- Loss-scale & gradient-dominance diagnostics (including training curve simulation)
- Official UKEA 32x32 validation-event diagnostics (strictly excluding test_events)
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple
import urllib.request

import numpy as np

GRAVITY_M_S2: float = 9.81
MM_PER_HOUR_TO_M_PER_SEC: float = 1.0 / 3_600_000.0
HF_DATASET_BASE_URL: str = "https://huggingface.co/datasets/holmescao/LarNO-dataset/resolve/main"
UKEA_VALIDATION_EVENTS: Tuple[str, ...] = ("r200y_p0.5_d3h_1", "r300y_p0.9_d3h_1")

BoundaryType = Literal["closed", "open", "periodic"]


@dataclass(frozen=True)
class ShallowWaterConfig:
    """SI configuration mirroring `src/crestix_pino/physics/shallow_water.py`."""

    dx: float = 2.0
    dy: float = 2.0
    dt: float = 60.0
    gravity: float = GRAVITY_M_S2
    wet_dry_eps: float = 1e-3
    default_manning_n: float = 0.035

    def __post_init__(self) -> None:
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(f"Spatial resolutions dx={self.dx}, dy={self.dy} must be > 0 m.")
        if self.dt <= 0.0:
            raise ValueError(f"Time interval dt={self.dt} must be > 0 s.")
        if self.gravity <= 0.0:
            raise ValueError(f"Gravity={self.gravity} must be > 0 m/s^2.")
        if self.wet_dry_eps <= 0.0:
            raise ValueError(f"wet_dry_eps={self.wet_dry_eps} must be > 0 m.")


@dataclass(frozen=True)
class BoundaryConfig:
    """Per-edge boundary configuration mirroring `boundary_conditions.py`."""

    north: BoundaryType = "closed"
    south: BoundaryType = "open"
    east: BoundaryType = "closed"
    west: BoundaryType = "closed"

    def __post_init__(self) -> None:
        valid = {"closed", "open", "periodic"}
        for side, val in (
            ("north", self.north),
            ("south", self.south),
            ("east", self.east),
            ("west", self.west),
        ):
            if val not in valid:
                raise ValueError(f"Invalid boundary condition '{val}' for {side}.")


@dataclass
class PhysicsLossWeights:
    """Loss component weights mirroring `src/crestix_pino/losses/physics_loss.py`."""

    pde_continuity: float = 1.0
    pde_momentum_x: float = 0.25
    pde_momentum_y: float = 0.25
    mass_conservation: float = 0.5
    boundary_condition: float = 0.5
    initial_condition: float = 1.0
    nonnegative_depth: float = 1.0


@dataclass
class SWEResiduals:
    """Container for 2D Shallow Water Equation residuals."""

    continuity: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray


def load_evaluation_metrics_module(repo_root: str | Path) -> Any:
    """Directly import `src/crestix_pino/evaluation/metrics.py` without triggering torch."""
    metrics_path = Path(repo_root) / "src" / "crestix_pino" / "evaluation" / "metrics.py"
    spec = importlib.util.spec_from_file_location("crestix_pino_eval_metrics_direct", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation metrics module from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mm_per_hour_to_m_per_sec(rainfall_mm_h: np.ndarray | float) -> np.ndarray | float:
    return rainfall_mm_h * MM_PER_HOUR_TO_M_PER_SEC


def m_per_sec_to_mm_per_hour(rainfall_m_s: np.ndarray | float) -> np.ndarray | float:
    return rainfall_m_s * 3_600_000.0


def mm_to_meters(depth_mm: np.ndarray | float) -> np.ndarray | float:
    return depth_mm * 1e-3


def softplus_beta5(h_raw: np.ndarray) -> np.ndarray:
    """Numerically stable softplus(beta=5.0) mirroring `PINOFloodModel._apply_nonnegative`."""
    beta = 5.0
    scaled = beta * np.asarray(h_raw, dtype=np.float64)
    return (np.maximum(scaled, 0.0) + np.log1p(np.exp(-np.abs(scaled)))) / beta


def d_softplus_beta5(h_raw: np.ndarray) -> np.ndarray:
    """Exact derivative sigmoid(5.0 * h_raw)."""
    scaled = 5.0 * np.asarray(h_raw, dtype=np.float64)
    return np.where(scaled >= 0, 1.0 / (1.0 + np.exp(-scaled)), np.exp(scaled) / (1.0 + np.exp(scaled)))


def preprocess_dem_numpy(
    dem: np.ndarray,
    wall_height: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror `src/crestix_pino/data/preprocessing.py:preprocess_dem`."""
    building_mask = np.isnan(dem)
    valid_mask = ~building_mask
    fill_val = float(np.nanmax(dem)) + float(wall_height) if np.any(valid_mask) else float(wall_height)
    dem_clean = np.where(building_mask, fill_val, dem).astype(np.float64)
    return dem_clean, building_mask, valid_mask


def compute_wet_mask(h: np.ndarray, wet_dry_eps: float = 1e-3) -> np.ndarray:
    return (h > wet_dry_eps).astype(h.dtype)


def time_derivative(field: np.ndarray, dt: float) -> np.ndarray:
    """Compute d(field)/dt along axis=-3 in [1/s] matching `shallow_water.time_derivative`."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0 s, got {dt}")
    t_steps = field.shape[-3]
    if t_steps < 2:
        raise ValueError(f"Need at least 2 time steps to compute d/dt, got T={t_steps}")
    df_dt = np.empty_like(field, dtype=np.float64)
    df_dt[..., 0, :, :] = (field[..., 1, :, :] - field[..., 0, :, :]) / dt
    if t_steps > 2:
        df_dt[..., 1:-1, :, :] = (field[..., 2:, :, :] - field[..., :-2, :, :]) / (2.0 * dt)
    df_dt[..., -1, :, :] = (field[..., -1, :, :] - field[..., -2, :, :]) / dt
    return df_dt


def spatial_derivatives_2d(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (df/dx, df/dy) along (W, H) matching `shallow_water.spatial_derivatives_2d`."""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"dx and dy must be > 0 m, got dx={dx}, dy={dy}")
    h_dim, w_dim = field.shape[-2], field.shape[-1]
    if h_dim < 2 or w_dim < 2:
        raise ValueError(f"Need H >= 2 and W >= 2, got ({h_dim}, {w_dim})")

    arr = np.asarray(field, dtype=np.float64)
    df_dx = np.empty_like(arr)
    df_dx[..., :, 0] = (arr[..., :, 1] - arr[..., :, 0]) / dx
    if w_dim > 2:
        df_dx[..., :, 1:-1] = (arr[..., :, 2:] - arr[..., :, :-2]) / (2.0 * dx)
    df_dx[..., :, -1] = (arr[..., :, -1] - arr[..., :, -2]) / dx

    df_dy = np.empty_like(arr)
    df_dy[..., 0, :] = (arr[..., 1, :] - arr[..., 0, :]) / dy
    if h_dim > 2:
        df_dy[..., 1:-1, :] = (arr[..., 2:, :] - arr[..., :-2, :]) / (2.0 * dy)
    df_dy[..., -1, :] = (arr[..., -1, :] - arr[..., -2, :]) / dy
    return df_dx, df_dy


def spatial_derivatives_2d_masked(
    field: np.ndarray,
    valid_mask: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided/central spatial derivatives using ONLY valid (non-building) neighbors.

    Avoids differencing across $+50$ m synthetic building fill walls.
    Cells with zero valid adjacent neighbors receive zero derivative.
    """
    arr = np.asarray(field, dtype=np.float64)
    mask = np.broadcast_to(valid_mask.astype(bool), arr.shape)
    h_dim, w_dim = arr.shape[-2], arr.shape[-1]

    df_dx = np.zeros_like(arr)
    for x in range(w_dim):
        has_l = (x > 0) & mask[..., :, x - 1] if x > 0 else np.zeros(mask[..., :, 0].shape, dtype=bool)
        has_r = (
            (x < w_dim - 1) & mask[..., :, x + 1]
            if x < w_dim - 1
            else np.zeros(mask[..., :, 0].shape, dtype=bool)
        )
        cur = mask[..., :, x]
        both = cur & has_l & has_r
        only_r = cur & (~has_l) & has_r
        only_l = cur & has_l & (~has_r)
        if x > 0 and x < w_dim - 1:
            df_dx[..., :, x] = np.where(
                both,
                (arr[..., :, x + 1] - arr[..., :, x - 1]) / (2.0 * dx),
                df_dx[..., :, x],
            )
        if x < w_dim - 1:
            df_dx[..., :, x] = np.where(
                only_r, (arr[..., :, x + 1] - arr[..., :, x]) / dx, df_dx[..., :, x]
            )
        if x > 0:
            df_dx[..., :, x] = np.where(
                only_l, (arr[..., :, x] - arr[..., :, x - 1]) / dx, df_dx[..., :, x]
            )

    df_dy = np.zeros_like(arr)
    for y in range(h_dim):
        has_u = (y > 0) & mask[..., y - 1, :] if y > 0 else np.zeros(mask[..., 0, :].shape, dtype=bool)
        has_d = (
            (y < h_dim - 1) & mask[..., y + 1, :]
            if y < h_dim - 1
            else np.zeros(mask[..., 0, :].shape, dtype=bool)
        )
        cur = mask[..., y, :]
        both = cur & has_u & has_d
        only_d = cur & (~has_u) & has_d
        only_u = cur & has_u & (~has_d)
        if y > 0 and y < h_dim - 1:
            df_dy[..., y, :] = np.where(
                both,
                (arr[..., y + 1, :] - arr[..., y - 1, :]) / (2.0 * dy),
                df_dy[..., y, :],
            )
        if y < h_dim - 1:
            df_dy[..., y, :] = np.where(
                only_d, (arr[..., y + 1, :] - arr[..., y, :]) / dy, df_dy[..., y, :]
            )
        if y > 0:
            df_dy[..., y, :] = np.where(
                only_u, (arr[..., y, :] - arr[..., y - 1, :]) / dy, df_dy[..., y, :]
            )

    return df_dx, df_dy


def compute_manning_friction_slope(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    manning_n: np.ndarray | float = 0.035,
    wet_dry_eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dimensionless Manning bed friction slopes (S_fx, S_fy)."""
    h_safe = np.maximum(h, wet_dry_eps)
    speed = np.sqrt(u * u + v * v + 1e-12)
    denom = np.power(h_safe, 4.0 / 3.0)
    n_sq = np.asarray(manning_n, dtype=np.float64) ** 2
    wet_mask = compute_wet_mask(h, wet_dry_eps)
    s_fx = wet_mask * (n_sq * u * speed) / denom
    s_fy = wet_mask * (n_sq * v * speed) / denom
    return s_fx, s_fy


def _expand_dem(dem: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    if dem.ndim == 2:
        return np.broadcast_to(dem[np.newaxis, np.newaxis, :, :], target_shape)
    if dem.ndim == 3:
        return np.broadcast_to(dem[:, np.newaxis, :, :], target_shape)
    if dem.ndim == 4 and dem.shape[1] == 1:
        return np.broadcast_to(dem, target_shape)
    if dem.ndim == 4 and dem.shape == target_shape:
        return np.asarray(dem, dtype=np.float64)
    raise ValueError(f"Incompatible dem shape {dem.shape} for target shape {target_shape}")


def compute_swe_residuals(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dem: np.ndarray,
    rainfall_m_s: np.ndarray,
    manning_n: Optional[np.ndarray | float] = None,
    infiltration_m_s: Optional[np.ndarray] = None,
    config: Optional[ShallowWaterConfig] = None,
    mask_dry_momentum: bool = True,
) -> SWEResiduals:
    """Evaluate 2D SWE residuals mirroring `src/crestix_pino/physics/residuals.py`."""
    cfg = config or ShallowWaterConfig()
    n_val = cfg.default_manning_n if manning_n is None else manning_n
    inf_val = np.zeros_like(rainfall_m_s, dtype=np.float64) if infiltration_m_s is None else infiltration_m_s
    dem_4d = _expand_dem(dem, h.shape)

    h_pos = np.maximum(h, 0.0)
    hu = h_pos * u
    hv = h_pos * v

    dh_dt = time_derivative(h_pos, dt=cfg.dt)
    dhu_dx, _ = spatial_derivatives_2d(hu, dx=cfg.dx, dy=cfg.dy)
    _, dhv_dy = spatial_derivatives_2d(hv, dx=cfg.dx, dy=cfg.dy)
    r_mass = dh_dt + dhu_dx + dhv_dy - (rainfall_m_s - inf_val)

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

    dz_dx, dz_dy = spatial_derivatives_2d(dem_4d, dx=cfg.dx, dy=cfg.dy)
    s_fx, s_fy = compute_manning_friction_slope(h_pos, u, v, manning_n=n_val, wet_dry_eps=cfg.wet_dry_eps)

    r_mom_x = dhu_dt + dflux_xx_dx + dflux_xy_dy + cfg.gravity * h_pos * (dz_dx + s_fx)
    r_mom_y = dhv_dt + dflux_xy_dx + dflux_yy_dy + cfg.gravity * h_pos * (dz_dy + s_fy)

    if mask_dry_momentum:
        wet_mask = compute_wet_mask(h_pos, wet_dry_eps=cfg.wet_dry_eps)
        r_mom_x = r_mom_x * wet_mask
        r_mom_y = r_mom_y * wet_mask

    return SWEResiduals(continuity=r_mass, momentum_x=r_mom_x, momentum_y=r_mom_y)


def compute_swe_residuals_masked(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dem: np.ndarray,
    rainfall_m_s: np.ndarray,
    valid_mask: np.ndarray,
    manning_n: Optional[np.ndarray | float] = None,
    infiltration_m_s: Optional[np.ndarray] = None,
    config: Optional[ShallowWaterConfig] = None,
) -> SWEResiduals:
    """Corrected building-aware SWE residuals using masked one-sided derivatives and building zeroing."""
    cfg = config or ShallowWaterConfig()
    n_val = cfg.default_manning_n if manning_n is None else manning_n
    inf_val = np.zeros_like(rainfall_m_s, dtype=np.float64) if infiltration_m_s is None else infiltration_m_s
    dem_4d = _expand_dem(dem, h.shape)
    mask_4d = np.broadcast_to(valid_mask.astype(bool), h.shape)

    h_pos = np.maximum(h, 0.0) * mask_4d
    u_m = u * mask_4d
    v_m = v * mask_4d
    hu = h_pos * u_m
    hv = h_pos * v_m

    dh_dt = time_derivative(h_pos, dt=cfg.dt)
    dhu_dx, _ = spatial_derivatives_2d_masked(hu, mask_4d, dx=cfg.dx, dy=cfg.dy)
    _, dhv_dy = spatial_derivatives_2d_masked(hv, mask_4d, dx=cfg.dx, dy=cfg.dy)
    r_mass = (dh_dt + dhu_dx + dhv_dy - (rainfall_m_s - inf_val)) * mask_4d

    half_g_h2 = 0.5 * cfg.gravity * (h_pos * h_pos)
    flux_xx = h_pos * (u_m * u_m) + half_g_h2
    flux_xy = h_pos * (u_m * v_m)
    flux_yy = h_pos * (v_m * v_m) + half_g_h2

    dhu_dt = time_derivative(hu, dt=cfg.dt)
    dhv_dt = time_derivative(hv, dt=cfg.dt)
    dflux_xx_dx, _ = spatial_derivatives_2d_masked(flux_xx, mask_4d, dx=cfg.dx, dy=cfg.dy)
    _, dflux_xy_dy = spatial_derivatives_2d_masked(flux_xy, mask_4d, dx=cfg.dx, dy=cfg.dy)
    dflux_xy_dx, _ = spatial_derivatives_2d_masked(flux_xy, mask_4d, dx=cfg.dx, dy=cfg.dy)
    _, dflux_yy_dy = spatial_derivatives_2d_masked(flux_yy, mask_4d, dx=cfg.dx, dy=cfg.dy)

    dz_dx, dz_dy = spatial_derivatives_2d_masked(dem_4d, mask_4d, dx=cfg.dx, dy=cfg.dy)
    s_fx, s_fy = compute_manning_friction_slope(h_pos, u_m, v_m, manning_n=n_val, wet_dry_eps=cfg.wet_dry_eps)

    wet_and_valid = compute_wet_mask(h_pos, wet_dry_eps=cfg.wet_dry_eps) * mask_4d
    r_mom_x = (dhu_dt + dflux_xx_dx + dflux_xy_dy + cfg.gravity * h_pos * (dz_dx + s_fx)) * wet_and_valid
    r_mom_y = (dhv_dt + dflux_xy_dx + dflux_yy_dy + cfg.gravity * h_pos * (dz_dy + s_fy)) * wet_and_valid
    return SWEResiduals(continuity=r_mass, momentum_x=r_mom_x, momentum_y=r_mom_y)


def compute_domain_volume(h: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.maximum(h, 0.0).sum(axis=(-2, -1)) * (dx * dy)


def compute_boundary_outflow_flux(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
    boundary_config: Optional[BoundaryConfig] = None,
) -> np.ndarray:
    bc = boundary_config or BoundaryConfig()
    h_pos = np.maximum(h, 0.0)
    hu = h_pos * u
    hv = h_pos * v
    flux = np.zeros(h.shape[:2], dtype=np.float64)
    if bc.west != "closed":
        flux = flux - hu[..., :, 0].sum(axis=-1) * dy
    if bc.east != "closed":
        flux = flux + hu[..., :, -1].sum(axis=-1) * dy
    if bc.north != "closed":
        flux = flux - hv[..., 0, :].sum(axis=-1) * dx
    if bc.south != "closed":
        flux = flux + hv[..., -1, :].sum(axis=-1) * dx
    return flux


def compute_mass_conservation_residual(
    h: np.ndarray,
    rainfall_m_s: np.ndarray,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    infiltration_m_s: Optional[np.ndarray] = None,
    config: Optional[ShallowWaterConfig] = None,
    boundary_config: Optional[BoundaryConfig] = None,
) -> np.ndarray:
    cfg = config or ShallowWaterConfig()
    inf_val = np.zeros_like(rainfall_m_s, dtype=np.float64) if infiltration_m_s is None else infiltration_m_s
    volume = compute_domain_volume(h, dx=cfg.dx, dy=cfg.dy)
    dv_dt = (volume[:, 1:] - volume[:, :-1]) / cfg.dt
    net_rain_rate = (rainfall_m_s - inf_val).sum(axis=(-2, -1)) * (cfg.dx * cfg.dy)
    q_rain_avg = 0.5 * (net_rain_rate[:, 1:] + net_rain_rate[:, :-1])
    if u is not None and v is not None:
        outflow = compute_boundary_outflow_flux(h, u, v, dx=cfg.dx, dy=cfg.dy, boundary_config=boundary_config)
        q_out_avg = 0.5 * (outflow[:, 1:] + outflow[:, :-1])
    else:
        q_out_avg = np.zeros_like(q_rain_avg)
    domain_area = float(h.shape[-2] * h.shape[-1]) * cfg.dx * cfg.dy
    return (dv_dt - (q_rain_avg - q_out_avg)) / domain_area


def compute_boundary_consistency_loss(
    h: np.ndarray,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    dx: float = 2.0,
    dy: float = 2.0,
    boundary_config: Optional[BoundaryConfig] = None,
) -> Tuple[float, Dict[str, Tuple[float, str]]]:
    """Mirror `boundary_conditions.compute_boundary_consistency_loss` and report each term's unit."""
    bc = boundary_config or BoundaryConfig()
    terms: List[float] = []
    breakdown_units: Dict[str, Tuple[float, str]] = {}

    if bc.north == "closed":
        val = float(np.mean(((h[..., 1, :] - h[..., 0, :]) / dy) ** 2))
        terms.append(val)
        breakdown_units["north_dh_dn_sq"] = (val, "dimensionless [1]")
        if v is not None:
            v_val = float(np.mean(v[..., 0, :] ** 2))
            terms.append(v_val)
            breakdown_units["north_v_sq"] = (v_val, "m^2/s^2")
    elif bc.north == "open":
        val = float(np.mean(((h[..., 1, :] - h[..., 0, :]) / dy) ** 2))
        terms.append(val)
        breakdown_units["north_dh_dn_sq"] = (val, "dimensionless [1]")
        if v is not None:
            dv_val = float(np.mean(((v[..., 1, :] - v[..., 0, :]) / dy) ** 2))
            terms.append(dv_val)
            breakdown_units["north_dv_dn_sq"] = (dv_val, "1/s^2")
    elif bc.north == "periodic":
        val = float(np.mean((h[..., 0, :] - h[..., -1, :]) ** 2))
        terms.append(val)
        breakdown_units["north_periodic_h_sq"] = (val, "m^2")
        if v is not None:
            v_val = float(np.mean((v[..., 0, :] - v[..., -1, :]) ** 2))
            terms.append(v_val)
            breakdown_units["north_periodic_v_sq"] = (v_val, "m^2/s^2")

    if bc.south == "closed":
        val = float(np.mean(((h[..., -1, :] - h[..., -2, :]) / dy) ** 2))
        terms.append(val)
        breakdown_units["south_dh_dn_sq"] = (val, "dimensionless [1]")
        if v is not None:
            v_val = float(np.mean(v[..., -1, :] ** 2))
            terms.append(v_val)
            breakdown_units["south_v_sq"] = (v_val, "m^2/s^2")
    elif bc.south == "open":
        val = float(np.mean(((h[..., -1, :] - h[..., -2, :]) / dy) ** 2))
        terms.append(val)
        breakdown_units["south_dh_dn_sq"] = (val, "dimensionless [1]")
        if v is not None:
            dv_val = float(np.mean(((v[..., -1, :] - v[..., -2, :]) / dy) ** 2))
            terms.append(dv_val)
            breakdown_units["south_dv_dn_sq"] = (dv_val, "1/s^2")

    if bc.west == "closed":
        val = float(np.mean(((h[..., :, 1] - h[..., :, 0]) / dx) ** 2))
        terms.append(val)
        breakdown_units["west_dh_dn_sq"] = (val, "dimensionless [1]")
        if u is not None:
            u_val = float(np.mean(u[..., :, 0] ** 2))
            terms.append(u_val)
            breakdown_units["west_u_sq"] = (u_val, "m^2/s^2")
    elif bc.west == "open":
        val = float(np.mean(((h[..., :, 1] - h[..., :, 0]) / dx) ** 2))
        terms.append(val)
        breakdown_units["west_dh_dn_sq"] = (val, "dimensionless [1]")
        if u is not None:
            du_val = float(np.mean(((u[..., :, 1] - u[..., :, 0]) / dx) ** 2))
            terms.append(du_val)
            breakdown_units["west_du_dn_sq"] = (du_val, "1/s^2")
    elif bc.west == "periodic":
        val = float(np.mean((h[..., :, 0] - h[..., :, -1]) ** 2))
        terms.append(val)
        breakdown_units["west_periodic_h_sq"] = (val, "m^2")
        if u is not None:
            u_val = float(np.mean((u[..., :, 0] - u[..., :, -1]) ** 2))
            terms.append(u_val)
            breakdown_units["west_periodic_u_sq"] = (u_val, "m^2/s^2")

    if bc.east == "closed":
        val = float(np.mean(((h[..., :, -1] - h[..., :, -2]) / dx) ** 2))
        terms.append(val)
        breakdown_units["east_dh_dn_sq"] = (val, "dimensionless [1]")
        if u is not None:
            u_val = float(np.mean(u[..., :, -1] ** 2))
            terms.append(u_val)
            breakdown_units["east_u_sq"] = (u_val, "m^2/s^2")
    elif bc.east == "open":
        val = float(np.mean(((h[..., :, -1] - h[..., :, -2]) / dx) ** 2))
        terms.append(val)
        breakdown_units["east_dh_dn_sq"] = (val, "dimensionless [1]")
        if u is not None:
            du_val = float(np.mean(((u[..., :, -1] - u[..., :, -2]) / dx) ** 2))
            terms.append(du_val)
            breakdown_units["east_du_dn_sq"] = (du_val, "1/s^2")

    mean_loss = float(np.mean(terms)) if terms else 0.0
    return mean_loss, breakdown_units


def compute_nonnegative_depth_loss(h: np.ndarray) -> float:
    neg = np.maximum(-np.asarray(h, dtype=np.float64), 0.0)
    return float(np.mean(neg * neg))


def compute_initial_condition_loss(h_pred: np.ndarray, h_init: np.ndarray) -> float:
    h0_true = h_init[:, 0, :, :] if h_init.ndim == 4 else h_init
    return float(np.mean((h_pred[:, 0, :, :] - h0_true) ** 2))


def compute_physics_loss_components(
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dem: np.ndarray,
    rainfall_m_s: np.ndarray,
    h_init: np.ndarray,
    h_true: np.ndarray,
    h_raw: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    manning_n: Optional[np.ndarray | float] = None,
    sw_config: Optional[ShallowWaterConfig] = None,
    bc_config: Optional[BoundaryConfig] = None,
    weights: Optional[PhysicsLossWeights] = None,
    data_weight: float = 1.0,
    physics_weight: float = 1.0,
    use_masked_swe: bool = False,
) -> Dict[str, float]:
    """Compute exact PINOCompositeLoss breakdown matching `src/crestix_pino/losses/physics_loss.py`."""
    cfg = sw_config or ShallowWaterConfig()
    bc = bc_config or BoundaryConfig()
    w = weights or PhysicsLossWeights()

    if valid_mask is not None:
        m4 = np.broadcast_to(valid_mask.astype(np.float64), h.shape)
        l_data = float(np.sum(((h - h_true) ** 2) * m4) / max(float(np.sum(m4)), 1.0))
    else:
        l_data = float(np.mean((h - h_true) ** 2))

    if use_masked_swe and valid_mask is not None:
        res = compute_swe_residuals_masked(h, u, v, dem, rainfall_m_s, valid_mask=valid_mask, manning_n=manning_n, config=cfg)
    else:
        res = compute_swe_residuals(h, u, v, dem, rainfall_m_s, manning_n=manning_n, config=cfg)

    l_cont = float(np.mean(res.continuity ** 2))
    l_mom_x = float(np.mean(res.momentum_x ** 2))
    l_mom_y = float(np.mean(res.momentum_y ** 2))

    mass_res = compute_mass_conservation_residual(h, rainfall_m_s, u=u, v=v, config=cfg, boundary_config=bc)
    l_mass = float(np.mean(mass_res ** 2))
    l_bc, _ = compute_boundary_consistency_loss(h, u=u, v=v, dx=cfg.dx, dy=cfg.dy, boundary_config=bc)
    l_ic = compute_initial_condition_loss(h, h_init)
    depth_for_neg = h_raw if h_raw is not None else h
    l_nonneg = compute_nonnegative_depth_loss(depth_for_neg)

    l_phys = (
        w.pde_continuity * l_cont
        + w.pde_momentum_x * l_mom_x
        + w.pde_momentum_y * l_mom_y
        + w.mass_conservation * l_mass
        + w.boundary_condition * l_bc
        + w.initial_condition * l_ic
        + w.nonnegative_depth * l_nonneg
    )
    total_loss = data_weight * l_data + physics_weight * l_phys
    return {
        "data_loss": l_data,
        "pde_continuity": l_cont,
        "pde_momentum_x": l_mom_x,
        "pde_momentum_y": l_mom_y,
        "mass_conservation": l_mass,
        "boundary_condition": l_bc,
        "initial_condition": l_ic,
        "nonnegative_depth": l_nonneg,
        "total_physics": float(l_phys),
        "weighted_data": float(data_weight * l_data),
        "weighted_physics": float(physics_weight * l_phys),
        "total_loss": float(total_loss),
    }


def _finite_diff_grad_norm(
    arr: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    eps: float = 1e-5,
) -> float:
    """Compute exact L2 gradient norm dL/d(arr) via centered finite differences."""
    flat = arr.ravel()
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        lp = loss_fn(arr)
        flat[i] = orig - eps
        lm = loss_fn(arr)
        flat[i] = orig
        grad[i] = (lp - lm) / (2.0 * eps)
    return float(np.linalg.norm(grad))


def run_synthetic_benchmarks() -> Dict[str, Any]:
    """Run all 7 synthetic analytical benchmarks and stress cases."""
    results: Dict[str, Any] = {}

    # S1: Lake-at-rest (linear bed vs quadratic bed boundary error)
    b, t_dim, h_dim, w_dim = 1, 4, 10, 10
    cfg1 = ShallowWaterConfig(dx=2.0, dy=2.0, dt=10.0)
    y = np.linspace(0.0, 1.0, h_dim).reshape(1, 1, h_dim, 1)
    x = np.linspace(0.0, 1.0, w_dim).reshape(1, 1, 1, w_dim)
    dem_lin = np.broadcast_to(1.5 * x + 0.8 * y, (b, t_dim, h_dim, w_dim)).copy()
    h_lin = 5.0 - dem_lin
    zeros = np.zeros_like(h_lin)
    res_lin = compute_swe_residuals(h_lin, zeros, zeros, dem_lin, zeros, config=cfg1)

    dem_quad = np.broadcast_to(1.2 * (x ** 2) + 0.9 * (y ** 2), (b, t_dim, h_dim, w_dim)).copy()
    h_quad = 5.0 - dem_quad
    res_quad = compute_swe_residuals(h_quad, zeros, zeros, dem_quad, zeros, config=cfg1)
    results["S1_lake_at_rest"] = {
        "linear_continuity_max_abs": float(np.max(np.abs(res_lin.continuity))),
        "linear_mom_x_interior_max_abs": float(np.max(np.abs(res_lin.momentum_x[:, :, 1:-1, 1:-1]))),
        "linear_mom_y_interior_max_abs": float(np.max(np.abs(res_lin.momentum_y[:, :, 1:-1, 1:-1]))),
        "linear_mom_x_edge_max_abs": float(np.max(np.abs(res_lin.momentum_x))),
        "quadratic_mom_x_interior_max_abs": float(np.max(np.abs(res_quad.momentum_x[:, :, 1:-1, 1:-1]))),
        "quadratic_mom_x_edge_max_abs": float(np.max(np.abs(res_quad.momentum_x))),
        "passed": bool(
            np.max(np.abs(res_lin.continuity)) < 1e-6
            and np.max(np.abs(res_lin.momentum_x[:, :, 1:-1, 1:-1])) < 1e-6
            and np.max(np.abs(res_lin.momentum_y[:, :, 1:-1, 1:-1])) < 1e-6
        ),
    }

    # S2: Uniform rainfall over flat closed basin
    cfg2 = ShallowWaterConfig(dx=5.0, dy=5.0, dt=60.0)
    bc_closed = BoundaryConfig(north="closed", south="closed", east="closed", west="closed")
    rain_rate = float(mm_per_hour_to_m_per_sec(72.0))
    rain2 = np.full((1, 6, 8, 8), rain_rate, dtype=np.float64)
    t_vals = np.arange(6, dtype=np.float64).reshape(1, 6, 1, 1) * cfg2.dt
    h2 = 0.2 + rain_rate * np.broadcast_to(t_vals, (1, 6, 8, 8))
    z2 = np.zeros_like(h2)
    res2 = compute_swe_residuals(h2, z2, z2, z2, rain2, config=cfg2)
    mass2 = compute_mass_conservation_residual(h2, rain2, u=z2, v=z2, config=cfg2, boundary_config=bc_closed)
    results["S2_uniform_rainfall_closed_basin"] = {
        "continuity_max_abs_m_s": float(np.max(np.abs(res2.continuity))),
        "mass_balance_max_abs_m_s": float(np.max(np.abs(mass2))),
        "mom_x_max_abs": float(np.max(np.abs(res2.momentum_x))),
        "passed": bool(np.max(np.abs(res2.continuity)) < 1e-12 and np.max(np.abs(mass2)) < 1e-12),
    }

    # S3: Steady uniform Manning open-channel flow (S_fx == S_0 = -dz/dx)
    cfg3 = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    s0 = 0.002
    n_coeff = 0.035
    h0 = 0.5
    u0 = (1.0 / n_coeff) * (h0 ** (2.0 / 3.0)) * (s0 ** 0.5)
    x_m = np.arange(10, dtype=np.float64).reshape(1, 1, 1, 10) * cfg3.dx
    dem3 = np.broadcast_to(-s0 * x_m, (1, 4, 8, 10)).copy()
    h3 = np.full((1, 4, 8, 10), h0, dtype=np.float64)
    u3 = np.full_like(h3, u0)
    v3 = np.zeros_like(h3)
    rain3 = np.zeros_like(h3)
    res3 = compute_swe_residuals(h3, u3, v3, dem3, rain3, manning_n=n_coeff, config=cfg3)
    results["S3_steady_manning_channel"] = {
        "steady_velocity_u0_m_s": float(u0),
        "continuity_max_abs_m_s": float(np.max(np.abs(res3.continuity))),
        "momentum_x_max_abs_m2_s2": float(np.max(np.abs(res3.momentum_x))),
        "momentum_y_max_abs_m2_s2": float(np.max(np.abs(res3.momentum_y))),
        "passed": bool(
            np.max(np.abs(res3.continuity)) < 1e-10
            and np.max(np.abs(res3.momentum_x)) < 1e-10
            and np.max(np.abs(res3.momentum_y)) < 1e-10
        ),
    }

    # S4: Dry-bed zero-depth safety & Manning wet-dry threshold gradient stiffness
    cfg4 = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    h4_dry = np.zeros((1, 3, 6, 6), dtype=np.float64)
    u4_dry = np.full_like(h4_dry, 0.5)
    v4_dry = np.full_like(h4_dry, -0.3)
    dem4 = np.zeros_like(h4_dry)
    res4_dry = compute_swe_residuals(h4_dry, u4_dry, v4_dry, dem4, np.zeros_like(h4_dry), config=cfg4)

    depths_sweep = np.geomspace(5e-4, 0.5, 40)
    manning_drag = []
    manning_sens = []
    eps_fd = 1e-5
    for d in depths_sweep:
        sf, _ = compute_manning_friction_slope(np.array([d]), np.array([1.0]), np.array([0.0]), 0.035, 1e-3)
        sf_p, _ = compute_manning_friction_slope(np.array([d + eps_fd]), np.array([1.0]), np.array([0.0]), 0.035, 1e-3)
        sf_m, _ = compute_manning_friction_slope(np.array([max(1e-7, d - eps_fd)]), np.array([1.0]), np.array([0.0]), 0.035, 1e-3)
        drag = float(GRAVITY_M_S2 * d * sf[0])
        drag_p = float(GRAVITY_M_S2 * (d + eps_fd) * sf_p[0])
        drag_m = float(GRAVITY_M_S2 * max(1e-7, d - eps_fd) * sf_m[0])
        manning_drag.append(drag)
        manning_sens.append(abs(drag_p - drag_m) / (2.0 * eps_fd))

    results["S4_dry_bed_and_stiffness"] = {
        "dry_all_finite": bool(
            np.all(np.isfinite(res4_dry.continuity))
            and np.all(np.isfinite(res4_dry.momentum_x))
            and np.all(np.isfinite(res4_dry.momentum_y))
        ),
        "dry_momentum_zero": bool(np.all(res4_dry.momentum_x == 0.0) and np.all(res4_dry.momentum_y == 0.0)),
        "sweep_depths_m": [float(v) for v in depths_sweep],
        "manning_gravity_drag_m2_s2": manning_drag,
        "manning_sensitivity_abs_d_dh": manning_sens,
        "max_sensitivity_near_threshold": float(np.max(manning_sens)),
        "drag_at_threshold_plus_m2_s2": float(
            GRAVITY_M_S2 * 0.00105 * (0.035 ** 2) * 1.0 / (0.00105 ** (4.0 / 3.0))
        ),
    }

    # S5: Negative-depth handling & softplus(beta=5) + h_raw paradox
    h_raw_grid = np.linspace(-2.0, 0.5, 251)
    h_sp_grid = softplus_beta5(h_raw_grid)
    l_data_curve = h_sp_grid ** 2  # h_true = 0.0 m on dry cell
    l_nonneg_h_raw_curve = np.maximum(-h_raw_grid, 0.0) ** 2
    l_nonneg_h_curve = np.maximum(-h_sp_grid, 0.0) ** 2

    # Solve 1D equilibrium over sweep of weights_nonneg
    opt_idx_w1 = int(np.argmin(l_data_curve + 1.0 * l_nonneg_h_raw_curve))
    opt_idx_w01 = int(np.argmin(l_data_curve + 0.1 * l_nonneg_h_raw_curve))

    # Verify PDE zero gradient for negative h when nonnegative_depth_mode="none"
    h_neg_arr = np.full((1, 3, 4, 4), -0.25, dtype=np.float64)
    u_zero = np.zeros_like(h_neg_arr)
    rain_zero = np.zeros_like(h_neg_arr)
    grad_cont_neg = _finite_diff_grad_norm(
        h_neg_arr.copy(),
        lambda arr: float(np.mean(compute_swe_residuals(arr, u_zero, u_zero, np.zeros_like(arr), rain_zero).continuity ** 2)),
    )
    grad_nonneg_neg = _finite_diff_grad_norm(
        h_neg_arr.copy(),
        lambda arr: compute_nonnegative_depth_loss(arr),
    )

    results["S5_negative_depth_and_softplus_paradox"] = {
        "softplus_at_h_raw_0_m": float(softplus_beta5(np.array([0.0]))[0]),
        "h_raw_grid": [float(v) for v in h_raw_grid],
        "h_softplus_grid": [float(v) for v in h_sp_grid],
        "dry_data_loss_curve": [float(v) for v in l_data_curve],
        "nonneg_h_raw_loss_curve": [float(v) for v in l_nonneg_h_raw_curve],
        "nonneg_h_loss_curve": [float(v) for v in l_nonneg_h_curve],
        "trapped_dry_depth_m_w1": float(h_sp_grid[opt_idx_w1]),
        "trapped_h_raw_w1": float(h_raw_grid[opt_idx_w1]),
        "trapped_dry_depth_m_w01": float(h_sp_grid[opt_idx_w01]),
        "pde_continuity_grad_norm_when_h_negative": float(grad_cont_neg),
        "nonneg_penalty_grad_norm_when_h_negative": float(grad_nonneg_neg),
    }

    # S6: +50m Building NaN Fill Wall Momentum Spike
    raw_dem = np.zeros((32, 32), dtype=np.float64)
    raw_dem[10:18, 12:20] = np.nan
    dem_clean, building_mask, valid_mask = preprocess_dem_numpy(raw_dem, wall_height=50.0)
    h_street = np.full((1, 4, 32, 32), 0.14, dtype=np.float64)  # typical initial softplus(0) ~ 0.1386m
    u_zero_32 = np.zeros_like(h_street)
    v_zero_32 = np.zeros_like(h_street)
    rain_32 = np.zeros_like(h_street)
    cfg_ukea = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)

    res_unmasked = compute_swe_residuals(h_street, u_zero_32, v_zero_32, dem_clean, rain_32, config=cfg_ukea)
    res_masked = compute_swe_residuals_masked(
        h_street, u_zero_32, v_zero_32, dem_clean, rain_32, valid_mask=valid_mask, config=cfg_ukea
    )
    loss_mom_unmasked = float(np.mean(res_unmasked.momentum_x ** 2 + res_unmasked.momentum_y ** 2))
    loss_mom_masked = float(np.mean(res_masked.momentum_x ** 2 + res_masked.momentum_y ** 2))

    # Even if h is zeroed on building cells ONLY (still unmasked DEM gradient on valid adjacent street cells)
    h_street_zeroed_bldg = h_street * valid_mask[np.newaxis, np.newaxis, :, :]
    res_adj_only = compute_swe_residuals(h_street_zeroed_bldg, u_zero_32, v_zero_32, dem_clean, rain_32, config=cfg_ukea)
    loss_mom_adj_unmasked_grad = float(np.mean(res_adj_only.momentum_x ** 2 + res_adj_only.momentum_y ** 2))

    results["S6_building_50m_wall_spike"] = {
        "building_cells": int(np.sum(building_mask)),
        "max_unmasked_dem_slope": float(np.max(np.abs(spatial_derivatives_2d(dem_clean, 8.0, 8.0)[0]))),
        "max_masked_dem_slope": float(np.max(np.abs(spatial_derivatives_2d_masked(dem_clean, valid_mask, 8.0, 8.0)[0]))),
        "unmasked_momentum_sq_mean": loss_mom_unmasked,
        "adjacent_street_unmasked_grad_mom_sq_mean": loss_mom_adj_unmasked_grad,
        "masked_momentum_sq_mean": loss_mom_masked,
        "spike_ratio_unmasked_over_masked": float(loss_mom_unmasked / max(loss_mom_masked, 1e-16)),
    }

    # S7: Open-boundary latent velocity gauge freedom in global mass conservation
    cfg7 = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    bc7 = BoundaryConfig(north="closed", south="open", east="closed", west="closed")
    rain7 = np.full((1, 6, 8, 8), mm_per_hour_to_m_per_sec(50.0), dtype=np.float64)
    # Deliberately WRONG depth trajectory that shrinks 50% instead of increasing under rain!
    h7_wrong = np.zeros((1, 6, 8, 8), dtype=np.float64)
    for k in range(6):
        h7_wrong[:, k] = 0.5 * (0.5 ** k) + 0.1
    v7_wrong_const = np.zeros_like(h7_wrong)
    vol7 = compute_domain_volume(h7_wrong, cfg7.dx, cfg7.dy)
    dv_dt7 = (vol7[:, 1:] - vol7[:, :-1]) / cfg7.dt
    q_rain7 = (rain7.sum(axis=(-2, -1)) * (cfg7.dx * cfg7.dy))
    # Solve south boundary velocity v_south(t) so (dv/dt) == Q_rain - Q_out at every step
    q_rain_avg7 = 0.5 * (q_rain7[:, 1:] + q_rain7[:, :-1])
    # Needed outflow at each interval: Q_out_needed = q_rain_avg - dv_dt
    q_out_needed = q_rain_avg7 - dv_dt7
    # Set instantaneous outflow Q(t_k) using trapezoidal rule constant-per-pair or piecewise solve
    # Here simplest check: set south v[:, k, -1, :] uniformly at interval-mid or exact step match:
    # q_out(t) = sum(h_south * v_south) * dx. Choose v_south(t_k) = q_out(t_k) / (sum(h_south) * dx)
    # where exact discrete q_out at nodes matches trapezoidal average Q_out_needed via constant across [k] by solving triangular system
    q_nodes = np.zeros((1, 6), dtype=np.float64)
    q_nodes[:, 0] = q_out_needed[:, 0]
    for k in range(5):
        q_nodes[:, k + 1] = 2.0 * q_out_needed[:, k] - q_nodes[:, k]
    for k in range(6):
        denom = h7_wrong[:, k, -1, :].sum() * cfg7.dx
        v7_wrong_const[:, k, -1, :] = q_nodes[:, k] / denom
    mass_res_cheated = compute_mass_conservation_residual(
        h7_wrong, rain7, u=np.zeros_like(h7_wrong), v=v7_wrong_const, config=cfg7, boundary_config=bc7
    )
    results["S7_open_boundary_gauge_freedom"] = {
        "true_volume_gain_expected_m3": float(rain7[:, :-1].sum() * cfg7.dt * cfg7.dx * cfg7.dy),
        "wrong_volume_change_m3": float(vol7[0, -1] - vol7[0, 0]),
        "cheated_mass_conservation_residual_max_abs_m_s": float(np.max(np.abs(mass_res_cheated))),
        "latent_v_south_max_abs_m_s": float(np.max(np.abs(v7_wrong_const))),
        "can_fake_zero_residual_with_wrong_depth": bool(np.max(np.abs(mass_res_cheated)) < 1e-10),
    }

    return results


def run_loss_and_gradient_analysis(
    dem_32: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    rainfall_sample: Optional[np.ndarray] = None,
    h_target_sample: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare individual physics loss terms, scale ratios, and gradient norms during optimization."""
    rng = np.random.default_rng(42)
    h_dim, w_dim, t_dim = 6, 6, 4  # compact patch for exact FD gradient norm verification

    if dem_32 is None or valid_mask is None:
        raw_dem = rng.normal(15.0, 0.4, size=(h_dim, w_dim))
        raw_dem[2:4, 2:4] = np.nan
        dem_p, _, mask_p = preprocess_dem_numpy(raw_dem, wall_height=50.0)
    else:
        # Take a representative 8x8 building-containing crop from real UKEA 32x32 for fast exact Jacobian norms
        y0, x0 = 10, 12
        dem_p = dem_32[y0 : y0 + 8, x0 : x0 + 8].astype(np.float64)
        mask_p = valid_mask[y0 : y0 + 8, x0 : x0 + 8]
        h_dim, w_dim = 8, 8

    h_raw = rng.normal(0.0, 0.25, size=(1, t_dim, h_dim, w_dim))
    u = rng.normal(0.0, 0.15, size=(1, t_dim, h_dim, w_dim))
    v = rng.normal(0.0, 0.15, size=(1, t_dim, h_dim, w_dim))
    rain = (
        rainfall_sample[np.newaxis, :t_dim, 10 : 10 + h_dim, 12 : 12 + w_dim]
        if rainfall_sample is not None
        else np.full((1, t_dim, h_dim, w_dim), mm_per_hour_to_m_per_sec(40.0), dtype=np.float64)
    )
    h_true = (
        h_target_sample[np.newaxis, :t_dim, 10 : 10 + h_dim, 12 : 12 + w_dim]
        if h_target_sample is not None
        else np.maximum(rng.uniform(0.0, 0.15, size=(1, t_dim, h_dim, w_dim)), 0.0) * mask_p[np.newaxis, np.newaxis, :, :]
    )
    h_init = h_true[:, 0, :, :]

    cfg_ukea = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    cfg_default = ShallowWaterConfig(dx=2.0, dy=2.0, dt=60.0)
    bc = BoundaryConfig(north="closed", south="open", east="closed", west="closed")
    w = PhysicsLossWeights()

    h_pred = softplus_beta5(h_raw)

    bd_ukea = compute_physics_loss_components(
        h=h_pred, u=u, v=v, dem=dem_p, rainfall_m_s=rain, h_init=h_init, h_true=h_true,
        h_raw=h_raw, valid_mask=mask_p, sw_config=cfg_ukea, bc_config=bc, weights=w,
        data_weight=1.0, physics_weight=0.1, use_masked_swe=False,
    )
    bd_default = compute_physics_loss_components(
        h=h_pred, u=u, v=v, dem=dem_p, rainfall_m_s=rain, h_init=h_init, h_true=h_true,
        h_raw=h_raw, valid_mask=mask_p, sw_config=cfg_default, bc_config=bc, weights=w,
        data_weight=1.0, physics_weight=1.0, use_masked_swe=False,
    )
    bd_masked = compute_physics_loss_components(
        h=h_pred, u=u, v=v, dem=dem_p, rainfall_m_s=rain, h_init=h_init, h_true=h_true,
        h_raw=h_raw, valid_mask=mask_p, sw_config=cfg_ukea, bc_config=bc, weights=w,
        data_weight=1.0, physics_weight=0.1, use_masked_swe=True,
    )

    # Compute exact finite-difference gradient norms w.r.t. h_raw for each term under ukea_32x32 config
    def _term_grad(term_key: str, masked: bool = False) -> float:
        def _loss_only(hr: np.ndarray) -> float:
            hp = softplus_beta5(hr)
            c = compute_physics_loss_components(
                h=hp, u=u, v=v, dem=dem_p, rainfall_m_s=rain, h_init=h_init, h_true=h_true,
                h_raw=hr, valid_mask=mask_p, sw_config=cfg_ukea, bc_config=bc, weights=w,
                data_weight=1.0, physics_weight=0.1, use_masked_swe=masked,
            )
            return c[term_key]

        return _finite_diff_grad_norm(h_raw.copy(), _loss_only, eps=1e-5)

    grad_norms_unmasked = {
        "data_loss": _term_grad("data_loss", False),
        "pde_continuity": _term_grad("pde_continuity", False),
        "pde_momentum_x": _term_grad("pde_momentum_x", False),
        "pde_momentum_y": _term_grad("pde_momentum_y", False),
        "mass_conservation": _term_grad("mass_conservation", False),
        "boundary_condition": _term_grad("boundary_condition", False),
        "initial_condition": _term_grad("initial_condition", False),
        "nonnegative_depth": _term_grad("nonnegative_depth", False),
        "weighted_physics_total": 0.1 * _term_grad("total_physics", False),
    }
    grad_norms_masked = {
        "data_loss": grad_norms_unmasked["data_loss"],
        "pde_continuity": _term_grad("pde_continuity", True),
        "pde_momentum_x": _term_grad("pde_momentum_x", True),
        "pde_momentum_y": _term_grad("pde_momentum_y", True),
        "weighted_physics_total": 0.1 * _term_grad("total_physics", True),
    }

    # Simulate short gradient-descent trajectory (Adam-like normalized or GD steps) showing dominance
    # Compare: (A) Current PINOCompositeLoss (unmasked + h_raw nonneg) vs (B) Masked + on-h nonneg + scaled
    traj_unmasked = {"step": [], "data_loss": [], "physics_loss_weighted": [], "momentum_x": [], "dry_cell_mean_h": []}
    traj_corrected = {"step": [], "data_loss": [], "physics_loss_weighted": [], "momentum_x": [], "dry_cell_mean_h": []}

    # Synthetic dry street beside building with target h_true=0 on dry half and 0.05m on wet half
    h_dry_true = np.zeros((1, 2, 4, 4), dtype=np.float64)
    h_dry_true[:, :, :2, :] = 0.05  # half wet, half dry street
    dem_small = np.zeros((4, 4), dtype=np.float64)
    dem_small[2:4, 2:4] = np.nan
    dem_small_clean, _, mask_small = preprocess_dem_numpy(dem_small, wall_height=50.0)
    h_dry_true *= mask_small[np.newaxis, np.newaxis, :, :]
    rain_small = np.zeros((1, 2, 4, 4), dtype=np.float64)
    u_small = np.zeros((1, 2, 4, 4), dtype=np.float64)
    v_small = np.zeros((1, 2, 4, 4), dtype=np.float64)

    # Measure exact gradient norms on a building-NaN patch (+50m wall) vs masked
    hr_bldg_probe = np.zeros((1, 2, 4, 4), dtype=np.float64)
    def _bldg_term_grad(term_key: str, masked: bool) -> float:
        def _loss_only(hr_arr: np.ndarray) -> float:
            hp = softplus_beta5(hr_arr)
            c = compute_physics_loss_components(
                h=hp, u=u_small, v=v_small, dem=dem_small_clean, rainfall_m_s=rain_small,
                h_init=h_dry_true[:, 0], h_true=h_dry_true, h_raw=hr_arr,
                valid_mask=mask_small, sw_config=cfg_ukea, bc_config=bc, weights=w,
                data_weight=1.0, physics_weight=0.1, use_masked_swe=masked,
            )
            return c[term_key]

        return _finite_diff_grad_norm(hr_bldg_probe.copy(), _loss_only, eps=1e-5)

    grad_norms_bldg_unmasked = {
        "data_loss": _bldg_term_grad("data_loss", False),
        "pde_continuity": _bldg_term_grad("pde_continuity", False),
        "pde_momentum_x": _bldg_term_grad("pde_momentum_x", False),
        "pde_momentum_y": _bldg_term_grad("pde_momentum_y", False),
        "weighted_physics_total": 0.1 * _bldg_term_grad("total_physics", False),
    }
    grad_norms_bldg_masked = {
        "data_loss": grad_norms_bldg_unmasked["data_loss"],
        "pde_continuity": _bldg_term_grad("pde_continuity", True),
        "pde_momentum_x": _bldg_term_grad("pde_momentum_x", True),
        "pde_momentum_y": _bldg_term_grad("pde_momentum_y", True),
        "weighted_physics_total": 0.1 * _bldg_term_grad("total_physics", True),
    }

    def _run_opt_sim(masked: bool, nonneg_on_h_only: bool, lr: float = 0.15, n_steps: int = 40) -> Dict[str, List[float]]:
        hr = np.zeros((1, 2, 4, 4), dtype=np.float64)
        m_adam = np.zeros_like(hr)
        v_adam = np.zeros_like(hr)
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        hist: Dict[str, List[float]] = {
            "step": [], "data_loss": [], "physics_loss_weighted": [], "momentum_x": [], "dry_cell_mean_h": []
        }
        dry_valid_mask = (h_dry_true == 0.0) & mask_small[np.newaxis, np.newaxis, :, :]
        for step in range(n_steps):
            hp = softplus_beta5(hr)
            c = compute_physics_loss_components(
                h=hp, u=u_small, v=v_small, dem=dem_small_clean, rainfall_m_s=rain_small,
                h_init=h_dry_true[:, 0], h_true=h_dry_true,
                h_raw=(hp if nonneg_on_h_only else hr),
                valid_mask=mask_small, sw_config=cfg_ukea, bc_config=bc, weights=w,
                data_weight=1.0, physics_weight=0.1, use_masked_swe=masked,
            )
            hist["step"].append(step)
            hist["data_loss"].append(c["data_loss"])
            hist["physics_loss_weighted"].append(c["weighted_physics"])
            hist["momentum_x"].append(c["pde_momentum_x"])
            hist["dry_cell_mean_h"].append(float(np.mean(hp[dry_valid_mask])))

            def _tot(arr: np.ndarray) -> float:
                hp_i = softplus_beta5(arr)
                ci = compute_physics_loss_components(
                    h=hp_i, u=u_small, v=v_small, dem=dem_small_clean, rainfall_m_s=rain_small,
                    h_init=h_dry_true[:, 0], h_true=h_dry_true,
                    h_raw=(hp_i if nonneg_on_h_only else arr),
                    valid_mask=mask_small, sw_config=cfg_ukea, bc_config=bc, weights=w,
                    data_weight=1.0, physics_weight=0.1, use_masked_swe=masked,
                )
                return ci["total_loss"]

            flat = hr.ravel()
            g = np.zeros_like(flat)
            eps = 1e-4
            for i in range(flat.size):
                orig = flat[i]
                flat[i] = orig + eps
                lp = _tot(hr)
                flat[i] = orig - eps
                lm = _tot(hr)
                flat[i] = orig
                g[i] = (lp - lm) / (2.0 * eps)
            g_arr = g.reshape(hr.shape)
            m_adam = beta1 * m_adam + (1.0 - beta1) * g_arr
            v_adam = beta2 * v_adam + (1.0 - beta2) * (g_arr ** 2)
            m_hat = m_adam / (1.0 - beta1 ** (step + 1))
            v_hat = v_adam / (1.0 - beta2 ** (step + 1))
            hr = hr - lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        return hist

    traj_unmasked = _run_opt_sim(masked=False, nonneg_on_h_only=False)
    traj_corrected = _run_opt_sim(masked=True, nonneg_on_h_only=True)

    return {
        "ukea_32x32_unmasked": bd_ukea,
        "default_pino_unmasked": bd_default,
        "ukea_32x32_masked": bd_masked,
        "grad_norms_unmasked_ukea": grad_norms_unmasked,
        "grad_norms_masked_ukea": grad_norms_masked,
        "grad_norms_bldg_unmasked": grad_norms_bldg_unmasked,
        "grad_norms_bldg_masked": grad_norms_bldg_masked,
        "gradient_dominance_ratio_unmasked_phys_over_data": float(
            grad_norms_unmasked["weighted_physics_total"] / max(grad_norms_unmasked["data_loss"], 1e-12)
        ),
        "gradient_dominance_ratio_masked_phys_over_data": float(
            grad_norms_masked["weighted_physics_total"] / max(grad_norms_masked["data_loss"], 1e-12)
        ),
        "gradient_dominance_ratio_bldg_unmasked_phys_over_data": float(
            grad_norms_bldg_unmasked["weighted_physics_total"] / max(grad_norms_bldg_unmasked["data_loss"], 1e-12)
        ),
        "gradient_dominance_ratio_bldg_masked_phys_over_data": float(
            grad_norms_bldg_masked["weighted_physics_total"] / max(grad_norms_bldg_masked["data_loss"], 1e-12)
        ),
        "optimization_trajectory_unmasked": traj_unmasked,
        "optimization_trajectory_corrected": traj_corrected,
    }


def _download_or_load_ukea_validation(cache_dir: Path) -> Optional[Dict[str, Any]]:
    """Fetch official UKEA DEM and 2 validation events from Hugging Face (never test events)."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dem_dst = cache_dir / "dem.npy"
        if not dem_dst.exists():
            urllib.request.urlretrieve(f"{HF_DATASET_BASE_URL}/geodata/ukea_8m_5min/dem.npy", dem_dst)

        events: Dict[str, Dict[str, np.ndarray]] = {}
        for ev in UKEA_VALIDATION_EVENTS:
            ev_dir = cache_dir / ev
            ev_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("h.npy", "rainfall.npy"):
                dst = ev_dir / fname
                if not dst.exists():
                    urllib.request.urlretrieve(
                        f"{HF_DATASET_BASE_URL}/flood/ukea_8m_5min/{ev}/{fname}", dst
                    )
            events[ev] = {
                "h": np.load(ev_dir / "h.npy"),
                "rainfall": np.load(ev_dir / "rainfall.npy"),
            }
        return {"dem": np.load(dem_dst), "events": events, "source": "huggingface_ukea_8m_5min_validation"}
    except Exception:
        return None


def _align_rainfall_300s_to_m_s(rainfall: np.ndarray, target_steps: int) -> np.ndarray:
    """Mirror `src/crestix_pino/data/preprocessing.py:align_rainfall_to_depth`."""
    t_r, h_dim, w_dim = rainfall.shape
    scale = 1.0 / (300.0 * 1000.0)
    if t_r == target_steps:
        aligned = rainfall.astype(np.float64)
    elif t_r % target_steps == 0:
        factor = t_r // target_steps
        aligned = rainfall.reshape(target_steps, factor, h_dim, w_dim).sum(axis=1).astype(np.float64)
    else:
        idx = np.linspace(0, t_r - 1, target_steps).astype(np.int64)
        aligned = rainfall[idx].astype(np.float64)
    return aligned * scale


def run_validation_event_diagnostics(
    repo_root: str | Path,
    cache_dir: str | Path,
) -> Dict[str, Any]:
    """Run conservation, residual, depth, and negative-violation diagnostics on validation events."""
    cache_path = Path(cache_dir)
    loaded = _download_or_load_ukea_validation(cache_path)
    if loaded is None:
        return {"source": "unavailable", "events": {}}

    eval_metrics = load_evaluation_metrics_module(repo_root)
    raw_dem = loaded["dem"]
    # Exact 32x32 region from configs/ukea_32x32.yaml (crop_y=9, crop_x=44, 32x32)
    dem_crop = raw_dem[9:41, 44:76]
    dem_clean, building_mask, valid_mask = preprocess_dem_numpy(dem_crop, wall_height=50.0)

    cfg_ukea = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    dem_grad_unmasked_x, dem_grad_unmasked_y = spatial_derivatives_2d(dem_clean, 8.0, 8.0)
    dem_grad_masked_x, dem_grad_masked_y = spatial_derivatives_2d_masked(dem_clean, valid_mask, 8.0, 8.0)

    event_reports: Dict[str, Any] = {}
    first_val_arrays: Optional[Dict[str, np.ndarray]] = None

    for ev_id, payload in loaded["events"].items():
        h_full = payload["h"]
        r_full = payload["rainfall"]
        h_crop = h_full[:, 9:41, 44:76].astype(np.float64)
        r_crop = r_full[:, 9:41, 44:76].astype(np.float64)

        r_m_s = _align_rainfall_300s_to_m_s(r_crop, target_steps=h_crop.shape[0])
        r_mm_h = m_per_sec_to_mm_per_hour(r_m_s)

        h_clean = np.nan_to_num(h_crop, nan=0.0)
        h_valid = h_clean[:, valid_mask]
        dh_dt = time_derivative(h_clean, dt=cfg_ukea.dt)

        # True volume evolution on valid (street) cells & all cells (in m^3)
        cell_area = 8.0 * 8.0
        vol_actual_valid = (h_clean * valid_mask[np.newaxis, :, :]).sum(axis=(-2, -1)) * cell_area
        step_rain_vol_valid = (r_m_s * valid_mask[np.newaxis, :, :]).sum(axis=(-2, -1)) * cfg_ukea.dt * cell_area
        closed_expected_from_t0 = np.empty_like(vol_actual_valid)
        closed_expected_from_t0[0] = vol_actual_valid[0]
        closed_expected_from_t0[1:] = vol_actual_valid[0] + np.cumsum(step_rain_vol_valid[1:])

        # Call Member 6's actual evaluation/metrics.py on true ground-truth depth!
        mc_eval_valid = eval_metrics.compute_mass_conservation_error(
            pred=h_clean * valid_mask[np.newaxis, :, :],
            initial_water_depth=h_clean[0] * valid_mask,
            rainfall_mm_per_hr=r_mm_h * valid_mask[np.newaxis, :, :],
            dt_seconds=300.0,
            cell_size_m=8.0,
        )
        neg_stats = eval_metrics.compute_nonnegative_violations(h_clean)

        # Diffusive wave residual under TRUE UKEA scales (dt=300, dx=8) vs DEFAULT ModelContract (dt=30, dx=10)
        res_diff_ukea = eval_metrics.compute_physics_residual(
            pred=h_clean, dem=dem_clean, rainfall_mm_per_hr=r_mm_h, dt_seconds=300.0, cell_size_m=8.0
        )
        res_diff_contract_default = eval_metrics.compute_physics_residual(
            pred=h_clean, dem=dem_clean, rainfall_mm_per_hr=r_mm_h, dt_seconds=30.0, cell_size_m=10.0
        )

        # SWE residual check on true ground truth depth (with u=v=0 reference & mild synthetic velocity perturbation)
        zeros = np.zeros((1, *h_clean.shape), dtype=np.float64)
        swe_unmasked = compute_swe_residuals(
            h=h_clean[np.newaxis, ...],
            u=zeros,
            v=zeros,
            dem=dem_clean,
            rainfall_m_s=r_m_s[np.newaxis, ...],
            config=cfg_ukea,
        )
        swe_masked = compute_swe_residuals_masked(
            h=h_clean[np.newaxis, ...],
            u=zeros,
            v=zeros,
            dem=dem_clean,
            rainfall_m_s=r_m_s[np.newaxis, ...],
            valid_mask=valid_mask,
            config=cfg_ukea,
        )

        if first_val_arrays is None:
            # 2D Diffusive-wave residual map at final interval matching evaluation/metrics.py formula
            dh_dt_last = (h_clean[-1] - h_clean[-2]) / 300.0
            h_mid_last = np.maximum(0.5 * (h_clean[-1] + h_clean[-2]), 0.0)
            eta_last = dem_clean + h_mid_last
            d_eta_dy, d_eta_dx = np.gradient(eta_last, 8.0, axis=(0, 1))
            slope_mag = np.sqrt(d_eta_dx ** 2 + d_eta_dy ** 2 + 1e-12)
            conv = (1.0 / 0.035) * (h_mid_last ** (5.0 / 3.0)) / np.sqrt(slope_mag)
            dqx_dx = np.gradient(-conv * d_eta_dx, 8.0, axis=1)
            dqy_dy = np.gradient(-conv * d_eta_dy, 8.0, axis=0)
            diff_res_last = np.abs(dh_dt_last + dqx_dx + dqy_dy - r_m_s[-1])
            diff_res_last[0, :] = 0.0
            diff_res_last[-1, :] = 0.0
            diff_res_last[:, 0] = 0.0
            diff_res_last[:, -1] = 0.0

            # S6 synthetic +50m building-NaN unmasked momentum magnitude (32x32) for direct visual contrast
            s6_raw_dem = np.zeros((32, 32), dtype=np.float64)
            s6_raw_dem[10:18, 12:20] = np.nan
            s6_dem_clean, _, _ = preprocess_dem_numpy(s6_raw_dem, wall_height=50.0)
            s6_h = np.full((1, 2, 32, 32), 0.14, dtype=np.float64)
            s6_zeros = np.zeros_like(s6_h)
            s6_res = compute_swe_residuals(s6_h, s6_zeros, s6_zeros, s6_dem_clean, s6_zeros, config=cfg_ukea)

            first_val_arrays = {
                "dem_clean": dem_clean,
                "valid_mask": valid_mask,
                "building_mask": building_mask,
                "h_final": h_clean[-1],
                "diffusive_res_ukea_final": diff_res_last,
                "s6_bldg_mom_unmasked": np.sqrt(s6_res.momentum_x[0, -1] ** 2 + s6_res.momentum_y[0, -1] ** 2),
                "mom_unmasked_final": np.sqrt(swe_unmasked.momentum_x[0, -1] ** 2 + swe_unmasked.momentum_y[0, -1] ** 2),
                "mom_masked_final": np.sqrt(swe_masked.momentum_x[0, -1] ** 2 + swe_masked.momentum_y[0, -1] ** 2),
                "cont_masked_final": swe_masked.continuity[0, -1],
                "rainfall_sample": r_m_s,
                "h_sample": h_clean,
            }

        event_reports[ev_id] = {
            "raw_h_shape": list(h_full.shape),
            "raw_rainfall_shape": list(r_full.shape),
            "h_nan_count": int(np.sum(np.isnan(h_crop))),
            "depth_stats_valid_m": {
                "min": float(np.min(h_valid)),
                "mean": float(np.mean(h_valid)),
                "p95": float(np.percentile(h_valid, 95)),
                "max": float(np.max(h_valid)),
                "wet_fraction_gt_1mm": float(np.mean(h_valid > 1e-3)),
            },
            "dh_dt_stats_m_s": {
                "min": float(np.min(dh_dt[:, valid_mask])),
                "mean_abs": float(np.mean(np.abs(dh_dt[:, valid_mask]))),
                "max": float(np.max(dh_dt[:, valid_mask])),
            },
            "negative_depth_violations_gt": neg_stats,
            "volume_series_m3": {
                "actual_valid_volume_m3": [float(v) for v in vol_actual_valid],
                "closed_domain_expected_m3": [float(v) for v in closed_expected_from_t0],
                "cumulative_net_outflow_m3": float(closed_expected_from_t0[-1] - vol_actual_valid[-1]),
                "outflow_fraction_of_rain_input": float(
                    (closed_expected_from_t0[-1] - vol_actual_valid[-1]) / max(float(step_rain_vol_valid[1:].sum()), 1e-9)
                ),
            },
            "eval_metrics_closed_mass_error": mc_eval_valid,
            "diffusive_wave_residual_ukea_scale": res_diff_ukea,
            "diffusive_wave_residual_contract_default_scale": res_diff_contract_default,
            "swe_zero_vel_momentum_mse_unmasked": float(np.mean(swe_unmasked.momentum_x ** 2 + swe_unmasked.momentum_y ** 2)),
            "swe_zero_vel_momentum_mse_masked": float(np.mean(swe_masked.momentum_x ** 2 + swe_masked.momentum_y ** 2)),
            "swe_zero_vel_continuity_rmse_m_s": float(np.sqrt(np.mean(swe_masked.continuity ** 2))),
        }

    return {
        "source": loaded["source"],
        "split_verified_validation_only": list(UKEA_VALIDATION_EVENTS),
        "dem_crop_stats": {
            "shape": list(dem_crop.shape),
            "building_nan_count": int(np.sum(building_mask)),
            "building_fraction": float(np.mean(building_mask)),
            "valid_dem_min_m": float(np.min(dem_clean[valid_mask])),
            "valid_dem_max_m": float(np.max(dem_clean[valid_mask])),
            "wall_fill_value_m": float(np.max(dem_clean[building_mask])) if np.any(building_mask) else None,
            "max_abs_grad_unmasked": float(max(np.max(np.abs(dem_grad_unmasked_x)), np.max(np.abs(dem_grad_unmasked_y)))),
            "max_abs_grad_masked": float(max(np.max(np.abs(dem_grad_masked_x)), np.max(np.abs(dem_grad_masked_y)))),
        },
        "events": event_reports,
        "_arrays": first_val_arrays,
    }


def run_all_diagnostics(
    repo_root: str | Path,
    cache_dir: str | Path,
) -> Dict[str, Any]:
    """Run synthetic verification, validation-event audit, and gradient/loss analysis."""
    synthetic = run_synthetic_benchmarks()
    val_diag = run_validation_event_diagnostics(repo_root=repo_root, cache_dir=cache_dir)
    arrays = val_diag.pop("_arrays", None)
    loss_grad = run_loss_and_gradient_analysis(
        dem_32=(arrays["dem_clean"] if arrays else None),
        valid_mask=(arrays["valid_mask"] if arrays else None),
        rainfall_sample=(arrays["rainfall_sample"] if arrays else None),
        h_target_sample=(arrays["h_sample"] if arrays else None),
    )
    return {
        "synthetic_benchmarks": synthetic,
        "validation_diagnostics": val_diag,
        "loss_and_gradient_analysis": loss_grad,
        "_arrays": arrays,
    }
