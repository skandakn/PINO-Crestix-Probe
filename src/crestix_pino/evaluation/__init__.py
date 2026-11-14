"""Evaluation metrics, runner, and visualization for PINO flood forecasting."""

from crestix_pino.evaluation.evaluate import evaluate_dataset, evaluate_event
from crestix_pino.evaluation.metrics import (
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
from crestix_pino.evaluation.visualization import (
    plot_horizon_metrics,
    plot_spatial_error_map,
    plot_time_series_comparison,
)

__all__ = [
    "compute_horizon_metrics",
    "compute_mae",
    "compute_mass_conservation_error",
    "compute_nonnegative_violations",
    "compute_physics_residual",
    "compute_r2",
    "compute_relative_depth_error",
    "compute_rmse",
    "evaluate_dataset",
    "evaluate_event",
    "evaluate_predictions",
    "plot_horizon_metrics",
    "plot_spatial_error_map",
    "plot_time_series_comparison",
]
