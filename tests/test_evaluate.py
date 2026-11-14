"""Unit tests for end-to-end flood event evaluation runner."""

import json
import numpy as np

from crestix_pino.evaluation.evaluate import evaluate_dataset
from crestix_pino.inference.predictor import PINOPredictor


def _dummy_model(dem, rainfall, h0, steps):
    return np.stack([h0 + 0.01 * (t + 1) for t in range(steps)], axis=0)


def test_evaluate_dataset_writes_artifacts(tmp_path):
    data_dir = tmp_path / "events"
    data_dir.mkdir()
    dem = np.zeros((6, 6), dtype=np.float32)
    h0 = np.full((6, 6), 0.1, dtype=np.float32)
    gt = np.stack([h0 + 0.01 * (t + 1) for t in range(4)], axis=0)

    np.savez(
        data_dir / "event_01.npz",
        dem=dem,
        rainfall=np.full((4,), 20.0, dtype=np.float32),
        initial_water_depth=h0,
        larno_water_depth=gt,
        dt_seconds=30.0,
        cell_size_m=10.0,
    )

    out_dir = tmp_path / "eval_out"
    predictor = PINOPredictor(model=_dummy_model)
    summary = evaluate_dataset(predictor, data_dir, output_dir=out_dir)

    assert summary["num_events"] == 1
    assert summary["aggregate_metrics"]["mean_mae_m"] < 1e-6
    assert (out_dir / "evaluation_summary.json").exists()
    assert (out_dir / "event_01" / "event_01_spatial_error.png").exists()
    assert (out_dir / "event_01" / "event_01_time_series.png").exists()
    assert (out_dir / "event_01" / "event_01_horizon_metrics.png").exists()
    assert (out_dir / "event_01" / "event_01_predictions.npz").exists()

    loaded_summary = json.loads((out_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert loaded_summary["num_events"] == 1
