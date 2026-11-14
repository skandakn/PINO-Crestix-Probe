"""Zero-dependency standalone SVG plot generators for PINO physics diagnostics."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _svg_open(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" style="background:#ffffff;font-family:Segoe UI,Arial,sans-serif;">\n'
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#111827">'
        f"{html.escape(title)}</text>\n"
    )


def _viridis_like(norm: float) -> str:
    """Map [0, 1] -> smooth perceptual hex color (dark indigo -> teal -> yellow/amber)."""
    t = max(0.0, min(1.0, float(norm)))
    stops = [
        (0.0, (24, 28, 67)),
        (0.33, (33, 113, 141)),
        (0.66, (68, 189, 112)),
        (1.0, (253, 210, 42)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-12)
            r = int(round(c0[0] + u * (c1[0] - c0[0])))
            g = int(round(c0[1] + u * (c1[1] - c0[1])))
            b = int(round(c0[2] + u * (c1[2] - c0[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fdd22a"


def _blues_like(norm: float) -> str:
    t = max(0.0, min(1.0, float(norm)))
    r = int(round(245 - t * 225))
    g = int(round(249 - t * 185))
    b = int(round(255 - t * 105))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_line_chart(
    parts: List[str],
    x0: float,
    y0: float,
    pw: float,
    ph: float,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[Tuple[str, Sequence[float], Sequence[float], str, str]],
    y_min_override: Optional[float] = None,
    y_max_override: Optional[float] = None,
    y_is_log: bool = False,
) -> None:
    """Render a reusable 2D polyline plot inside SVG bounds (x0, y0, pw, ph)."""
    parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#f9fafb" stroke="#d1d5db"/>\n')
    parts.append(
        f'<text x="{x0 + pw / 2:.1f}" y="{y0 - 8:.1f}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">'
        f"{html.escape(title)}</text>\n"
    )

    all_x = [xv for _, xs, _, _, _ in series for xv in xs]
    all_y = [yv for _, _, ys, _, _ in series for yv in ys if math.isfinite(yv)]
    xmin, xmax = min(all_x), max(all_x)
    if xmax <= xmin:
        xmax = xmin + 1.0

    if y_is_log:
        pos_y = [math.log10(max(yv, 1e-16)) for yv in all_y]
        ymin = y_min_override if y_min_override is not None else math.floor(min(pos_y))
        ymax = y_max_override if y_max_override is not None else math.ceil(max(pos_y))
    else:
        ymin = y_min_override if y_min_override is not None else min(all_y)
        ymax = y_max_override if y_max_override is not None else max(all_y)
        if ymax <= ymin:
            ymax = ymin + 1.0

    def _map_x(val: float) -> float:
        return x0 + (val - xmin) / (xmax - xmin) * pw

    def _map_y(val: float) -> float:
        v = math.log10(max(val, 1e-16)) if y_is_log else val
        v = max(ymin, min(ymax, v))
        return y0 + ph - (v - ymin) / (ymax - ymin) * ph

    for i in range(5):
        frac = i / 4.0
        yy = y0 + ph - frac * ph
        yval = ymin + frac * (ymax - ymin)
        lbl = f"1e{yval:.0f}" if y_is_log else f"{yval:.3g}"
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-dasharray="3,3"/>\n')
        parts.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">{lbl}</text>\n')

        xx = x0 + frac * pw
        xval = xmin + frac * (xmax - xmin)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + ph}" stroke="#e5e7eb" stroke-dasharray="3,3"/>\n')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + ph + 15}" text-anchor="middle" font-size="10" fill="#4b5563">{xval:.3g}</text>\n')

    for _, xs, ys, color, dash in series:
        pts = " ".join(f"{_map_x(xv):.1f},{_map_y(yv):.1f}" for xv, yv in zip(xs, ys))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{dash_attr} points="{pts}"/>\n')

    # Legend top-right inside panel
    lx = x0 + 12
    ly = y0 + 16
    for idx, (label, _, _, color, dash) in enumerate(series):
        yy = ly + idx * 16
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx}" y1="{yy - 4}" x2="{lx + 20}" y2="{yy - 4}" stroke="{color}" stroke-width="2.5"{dash_attr}/>\n')
        parts.append(f'<text x="{lx + 26}" y="{yy}" font-size="10.5" fill="#111827">{html.escape(label)}</text>\n')

    parts.append(f'<text x="{x0 + pw / 2:.1f}" y="{y0 + ph + 33}" text-anchor="middle" font-size="11" fill="#374151">{html.escape(xlabel)}</text>\n')
    parts.append(
        f'<text x="{x0 - 44}" y="{y0 + ph / 2:.1f}" text-anchor="middle" font-size="11" fill="#374151" '
        f'transform="rotate(-90 {x0 - 44} {y0 + ph / 2:.1f})">{html.escape(ylabel)}</text>\n'
    )


def plot_synthetic_benchmarks(synth: Dict[str, Any], dst: Path) -> Path:
    """Plot analytical residual verification and wet-dry Manning stiffness."""
    w, h = 1080, 460
    parts = [_svg_open(w, h, "PINO Synthetic Analytical Benchmarks & Wet-Dry Manning Stiffness")]

    # Left: Bar chart of log10 residual metrics
    bx0, by0, bpw, bph = 80, 65, 430, 320
    parts.append(f'<rect x="{bx0}" y="{by0}" width="{bpw}" height="{bph}" fill="#f9fafb" stroke="#d1d5db"/>\n')
    parts.append(f'<text x="{bx0 + bpw / 2:.1f}" y="{by0 - 8}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">Synthetic Benchmark Residual Magnitudes (log10)</text>\n')

    bars = [
        ("S1 Lake Cont", max(synth["S1_lake_at_rest"]["linear_continuity_max_abs"], 1e-16), "#2563eb"),
        ("S1 Quad Edge", max(synth["S1_lake_at_rest"]["quadratic_mom_x_edge_max_abs"], 1e-16), "#f59e0b"),
        ("S2 Rain Mass", max(synth["S2_uniform_rainfall_closed_basin"]["mass_balance_max_abs_m_s"], 1e-16), "#2563eb"),
        ("S3 Manning", max(synth["S3_steady_manning_channel"]["momentum_x_max_abs_m2_s2"], 1e-16), "#2563eb"),
        ("S6 Wall Unmask", max(synth["S6_building_50m_wall_spike"]["unmasked_momentum_sq_mean"], 1e-16), "#dc2626"),
        ("S6 Wall Mask", max(synth["S6_building_50m_wall_spike"]["masked_momentum_sq_mean"], 1e-16), "#16a34a"),
        ("S7 Cheat Mass", max(synth["S7_open_boundary_gauge_freedom"]["cheated_mass_conservation_residual_max_abs_m_s"], 1e-16), "#7c3aed"),
    ]
    log_vals = [math.log10(v) for _, v, _ in bars]
    lmin, lmax = -16.5, max(2.5, math.ceil(max(log_vals)) + 0.5)
    for g in range(-15, int(lmax) + 1, 3):
        yy = by0 + bph - (g - lmin) / (lmax - lmin) * bph
        if by0 <= yy <= by0 + bph:
            parts.append(f'<line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bpw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-dasharray="3,3"/>\n')
            parts.append(f'<text x="{bx0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">1e{g}</text>\n')

    slot = bpw / len(bars)
    zero_y = by0 + bph - (-16.0 - lmin) / (lmax - lmin) * bph
    for idx, (label, val, color) in enumerate(bars):
        lv = max(lmin, math.log10(val))
        top_y = by0 + bph - (lv - lmin) / (lmax - lmin) * bph
        bar_h = max(2.0, by0 + bph - top_y)
        cx = bx0 + idx * slot + slot * 0.18
        bw = slot * 0.64
        parts.append(f'<rect x="{cx:.1f}" y="{top_y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.88"/>\n')
        parts.append(f'<text x="{cx + bw / 2:.1f}" y="{top_y - 4:.1f}" text-anchor="middle" font-size="9" fill="#111827">{val:.1e}</text>\n')
        parts.append(f'<text x="{cx + bw / 2:.1f}" y="{by0 + bph + 15}" text-anchor="middle" font-size="9.5" fill="#374151">{html.escape(label)}</text>\n')
    parts.append(f'<text x="{bx0 - 44}" y="{by0 + bph / 2:.1f}" text-anchor="middle" font-size="11" fill="#374151" transform="rotate(-90 {bx0 - 44} {by0 + bph / 2:.1f})">Residual Magnitude (SI)</text>\n')

    # Right: Manning wet-dry sensitivity vs depth (log10 y)
    s4 = synth["S4_dry_bed_and_stiffness"]
    depths_log = [math.log10(d) for d in s4["sweep_depths_m"]]
    sens = s4["manning_sensitivity_abs_d_dh"]
    drag = s4["manning_gravity_drag_m2_s2"]
    _draw_line_chart(
        parts, 590, 65, 450, 320,
        title="Wet-Dry Manning Sensitivity near h_eps = 1e-3 m",
        xlabel="log10(Water Depth h [m])  [-3 => 1 mm]",
        ylabel="log10 Magnitude",
        series=[
            ("|d(g*h*S_f)/dh| [m/s^2]", depths_log, sens, "#dc2626", ""),
            ("g*h*S_f [m^2/s^2]", depths_log, [max(v, 1e-8) for v in drag], "#2563eb", "5,4"),
        ],
        y_is_log=True,
    )

    parts.append("</svg>\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts), encoding="utf-8")
    return dst


def plot_loss_and_gradient_dominance(loss_grad: Dict[str, Any], dst: Path) -> Path:
    """Plot physics loss component scales and gradient norm dominance before vs after masking."""
    w, h = 1080, 460
    parts = [_svg_open(w, h, "Loss Scale Imbalance & Gradient Dominance (Real UKEA Crop + Building-NaN Patch)")]

    # Left: log10 gradient norm comparison across UKEA 32x32 (0 NaNs) & Urban Building-NaN Patch (+50m wall)
    bx0, by0, bpw, bph = 85, 65, 440, 320
    parts.append(f'<rect x="{bx0}" y="{by0}" width="{bpw}" height="{bph}" fill="#f9fafb" stroke="#d1d5db"/>\n')
    parts.append(f'<text x="{bx0 + bpw / 2:.1f}" y="{by0 - 8}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">Grad Norm ||dL/dh_raw||_2 (log10): Unmasked vs Masked</text>\n')

    gu = loss_grad["grad_norms_unmasked_ukea"]
    gm = loss_grad["grad_norms_masked_ukea"]
    bu = loss_grad["grad_norms_bldg_unmasked"]
    bm = loss_grad["grad_norms_bldg_masked"]
    items = [
        ("UKEA Data", gu["data_loss"], gm["data_loss"]),
        ("UKEA Cont", gu["pde_continuity"], gm["pde_continuity"]),
        ("UKEA MomX", gu["pde_momentum_x"], gm["pde_momentum_x"]),
        ("Bldg MomX", bu["pde_momentum_x"], bm["pde_momentum_x"]),
        ("Bldg 0.1*Phys", bu["weighted_physics_total"], bm["weighted_physics_total"]),
    ]
    all_vals = [v for _, a, b in items for v in (a, b) if v > 0]
    lmin = min(-9.0, math.floor(math.log10(min(all_vals))) - 0.5)
    lmax = math.ceil(math.log10(max(all_vals))) + 0.5
    for g in range(int(lmin), int(lmax) + 1, 2):
        yy = by0 + bph - (g - lmin) / (lmax - lmin) * bph
        if by0 <= yy <= by0 + bph:
            parts.append(f'<line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bpw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-dasharray="3,3"/>\n')
            parts.append(f'<text x="{bx0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">1e{g}</text>\n')

    slot = bpw / len(items)
    for idx, (label, v_un, v_mk) in enumerate(items):
        for offset, val, col in ((0.14, v_un, "#dc2626"), (0.50, v_mk, "#16a34a")):
            lv = max(lmin, math.log10(max(val, 1e-16)))
            top_y = by0 + bph - (lv - lmin) / (lmax - lmin) * bph
            bh = max(2.0, by0 + bph - top_y)
            cx = bx0 + idx * slot + slot * offset
            bw = slot * 0.34
            parts.append(f'<rect x="{cx:.1f}" y="{top_y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" opacity="0.88"/>\n')
        parts.append(f'<text x="{bx0 + idx * slot + slot * 0.5:.1f}" y="{by0 + bph + 15}" text-anchor="middle" font-size="10" fill="#374151">{html.escape(label)}</text>\n')

    # Legend for bar chart
    parts.append(f'<rect x="{bx0 + 12}" y="{by0 + 12}" width="12" height="12" fill="#dc2626"/><text x="{bx0 + 28}" y="{by0 + 22}" font-size="10.5" fill="#111827">Current Unmasked (w_phys=0.1)</text>\n')
    parts.append(f'<rect x="{bx0 + 12}" y="{by0 + 30}" width="12" height="12" fill="#16a34a"/><text x="{bx0 + 28}" y="{by0 + 40}" font-size="10.5" fill="#111827">Building-Masked (w_phys=0.1)</text>\n')
    parts.append(f'<text x="{bx0 - 44}" y="{by0 + bph / 2:.1f}" text-anchor="middle" font-size="11" fill="#374151" transform="rotate(-90 {bx0 - 44} {by0 + bph / 2:.1f})">Gradient Norm (log10)</text>\n')

    # Right: Optimization trajectory comparing weighted physics vs data loss
    tu = loss_grad["optimization_trajectory_unmasked"]
    tc = loss_grad["optimization_trajectory_corrected"]
    _draw_line_chart(
        parts, 595, 65, 445, 320,
        title="Optimization Steps: Weighted Physics vs Data MSE",
        xlabel="Gradient Descent Step",
        ylabel="log10 Loss Component",
        series=[
            ("Unmasked Weighted Physics", tu["step"], tu["physics_loss_weighted"], "#dc2626", ""),
            ("Unmasked Data MSE", tu["step"], tu["data_loss"], "#f59e0b", "5,4"),
            ("Corrected Weighted Physics", tc["step"], tc["physics_loss_weighted"], "#16a34a", ""),
            ("Corrected Data MSE", tc["step"], tc["data_loss"], "#2563eb", "5,4"),
        ],
        y_is_log=True,
    )

    parts.append("</svg>\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts), encoding="utf-8")
    return dst


def plot_softplus_negative_paradox(synth: Dict[str, Any], loss_grad: Dict[str, Any], dst: Path) -> Path:
    """Plot the softplus(beta=5) + h_raw nonnegative-depth penalty paradox."""
    w, h = 1080, 460
    parts = [_svg_open(w, h, "Negative-Depth Penalty Paradox: Penalizing h_raw < 0 Under Softplus(beta=5)")]

    s5 = synth["S5_negative_depth_and_softplus_paradox"]
    hr = s5["h_raw_grid"]
    tot_w1 = [d + n for d, n in zip(s5["dry_data_loss_curve"], s5["nonneg_h_raw_loss_curve"])]
    _draw_line_chart(
        parts, 80, 65, 445, 320,
        title=f"Dry-Cell Loss Landscape (trapped h = {s5['trapped_dry_depth_m_w1']:.3f} m at w=1.0)",
        xlabel="Raw Pre-Activation Head h_raw [m]",
        ylabel="Water Depth [m] / Loss",
        series=[
            ("h = softplus(h_raw, beta=5) [m]", hr, s5["h_softplus_grid"], "#2563eb", ""),
            ("Data MSE (h - 0)^2", hr, s5["dry_data_loss_curve"], "#16a34a", "5,4"),
            ("Penalty ReLU(-h_raw)^2", hr, s5["nonneg_h_raw_loss_curve"], "#dc2626", "5,4"),
            ("Total Loss (Data + ReLU(-h_raw)^2)", hr, tot_w1, "#7c3aed", ""),
        ],
        y_min_override=0.0,
        y_max_override=0.6,
    )

    tu = loss_grad["optimization_trajectory_unmasked"]
    tc = loss_grad["optimization_trajectory_corrected"]
    _draw_line_chart(
        parts, 595, 65, 445, 320,
        title="Dry-Cell Depth Convergence: h_raw Penalty vs Corrected on-h",
        xlabel="Gradient Descent Step (target dry depth = 0.0 m)",
        ylabel="Mean Predicted Dry-Cell Depth [m]",
        series=[
            ("Current (penalizes h_raw < 0 & unmasked walls)", tu["step"], tu["dry_cell_mean_h"], "#dc2626", ""),
            ("Corrected (building-masked & penalty on h only)", tc["step"], tc["dry_cell_mean_h"], "#16a34a", ""),
        ],
        y_min_override=0.0,
    )

    parts.append("</svg>\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts), encoding="utf-8")
    return dst


def plot_validation_mass_balance(val_diag: Dict[str, Any], dst: Path) -> Path:
    """Plot actual open-patch validation volume vs closed-domain rain accumulation."""
    events = val_diag.get("events", {})
    w, h = 1080, 460
    parts = [_svg_open(w, h, "Official UKEA Validation Events: Actual Volume vs Closed-Domain Mass Assumption")]

    ev_items = list(events.items())[:2]
    for idx, (ev_id, rep) in enumerate(ev_items):
        x0 = 85 if idx == 0 else 595
        vs = rep["volume_series_m3"]
        steps_min = [k * 5.0 for k in range(len(vs["actual_valid_volume_m3"]))]
        outflow_pct = vs["outflow_fraction_of_rain_input"] * 100.0
        _draw_line_chart(
            parts, x0, 65, 445, 320,
            title=f"{ev_id} (Net Outflow = {outflow_pct:.1f}% of Rain Input)",
            xlabel="Time (minutes)",
            ylabel="Domain Water Volume (m^3)",
            series=[
                ("LarNO Ground-Truth Volume (valid 32x32)", steps_min, vs["actual_valid_volume_m3"], "#2563eb", ""),
                ("Closed-Domain Assumption (V0 + Rain)", steps_min, vs["closed_domain_expected_m3"], "#dc2626", "5,4"),
            ],
            y_min_override=0.0,
        )

    parts.append("</svg>\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts), encoding="utf-8")
    return dst


def plot_validation_spatial_maps(arrays: Dict[str, Any], dst: Path) -> Path:
    """Render 4-panel spatial SVG heatmaps for real UKEA validation fields and S6 +50m building wall test."""
    w, h = 1120, 360
    parts = [_svg_open(w, h, "Validation Spatial Audit (Real UKEA Crop + S6 Synthetic +50m Building-Wall Residual)")]

    panels = [
        ("Real UKEA DEM [m] (0 NaNs)", arrays["dem_clean"], _viridis_like),
        ("Real UKEA Final Depth [m]", arrays["h_final"], _blues_like),
        ("Real UKEA |R_diffusive| [m/s]", arrays["diffusive_res_ukea_final"], _viridis_like),
        ("S6 Bldg Wall: log1p(|r_mom|)", np.log1p(arrays["s6_bldg_mom_unmasked"]), _viridis_like),
    ]

    cell_px = 7
    grid_px = 32 * cell_px  # 224 px
    for idx, (title, arr2d, cmap_fn) in enumerate(panels):
        px = 35 + idx * 270
        py = 75
        parts.append(f'<text x="{px + grid_px / 2:.1f}" y="{py - 10}" text-anchor="middle" font-size="11.5" font-weight="bold" fill="#1f2937">{html.escape(title)}</text>\n')
        parts.append(f'<rect x="{px - 1}" y="{py - 1}" width="{grid_px + 2}" height="{grid_px + 2}" fill="none" stroke="#111827"/>\n')

        vmin = float(np.nanmin(arr2d))
        vmax = float(np.nanmax(arr2d))
        span = max(vmax - vmin, 1e-12)
        rows, cols = arr2d.shape
        for r in range(rows):
            for c in range(cols):
                col_hex = cmap_fn((float(arr2d[r, c]) - vmin) / span)
                parts.append(
                    f'<rect x="{px + c * cell_px}" y="{py + r * cell_px}" width="{cell_px}" height="{cell_px}" fill="{col_hex}"/>\n'
                )
        parts.append(f'<text x="{px + grid_px / 2:.1f}" y="{py + grid_px + 18}" text-anchor="middle" font-size="10" fill="#4b5563">min={vmin:.3g}, max={vmax:.3g}</text>\n')

    parts.append("</svg>\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(parts), encoding="utf-8")
    return dst
