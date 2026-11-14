"""Unit tests for PINO visualization helpers."""

import numpy as np

from crestix_pino.evaluation.metrics import compute_horizon_metrics
from crestix_pino.evaluation.visualization import (
    plot_horizon_metrics,
    plot_spatial_error_map,
    plot_time_series_comparison,
)


def test_visualizations_create_nonempty_pngs(tmp_path):
    rng = np.random.default_rng(0)
    target = rng.uniform(0.0, 1.0, size=(5, 8, 8))
    pred = target + rng.normal(0.0, 0.05, size=(5, 8, 8))

    p1 = plot_spatial_error_map(pred, target, tmp_path / "spatial.png", step_idx=2)
    p2 = plot_time_series_comparison(
        pred,
        target,
        tmp_path / "timeseries.png",
        initial_water_depth=np.zeros((8, 8)),
        rainfall_mm_per_hr=20.0,
    )
    horizon = compute_horizon_metrics(pred, target)
    p3 = plot_horizon_metrics(horizon, tmp_path / "horizon.png")

    for path in (p1, p2, p3):
        assert path.exists()
        assert path.stat().st_size > 1000
