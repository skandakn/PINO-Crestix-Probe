"""Unit tests for PINO model loader and inference predictor."""

import numpy as np
import pytest

from crestix_pino.inference.model_loader import ModelContract, load_model
from crestix_pino.inference.predictor import PINOPredictor, predict


def _dummy_pino_model(dem, rainfall, initial_water_depth, forecast_steps):
    """Deterministic callable model matching the agreed tensor contract."""
    h0 = np.asarray(initial_water_depth, dtype=np.float32)
    rain = np.asarray(rainfall, dtype=np.float32)
    rain_mean = float(np.mean(rain)) * 1e-4
    if h0.ndim == 2:
        return np.stack([h0 + (t + 1) * rain_mean for t in range(forecast_steps)], axis=0)
    return np.stack([h0 + (t + 1) * rain_mean for t in range(forecast_steps)], axis=1)


def test_predict_unbatched_and_batched():
    dem = np.zeros((8, 8), dtype=np.float32)
    h0 = np.full((8, 8), 0.05, dtype=np.float32)
    out = predict(dem, 25.0, h0, 5, model=_dummy_pino_model)
    assert out["water_depth"].shape == (5, 8, 8)
    assert out["metadata"]["units"]["water_depth"] == "m"
    assert out["metadata"]["units"]["rainfall"] == "mm/h"
    assert out["metadata"]["latency_ms"]["model_only"] >= 0.0
    assert out["metadata"]["latency_ms"]["total_pipeline"] >= out["metadata"]["latency_ms"]["model_only"]

    dem_b = np.zeros((2, 8, 8), dtype=np.float32)
    h0_b = np.full((2, 8, 8), 0.05, dtype=np.float32)
    predictor = PINOPredictor(model=_dummy_pino_model)
    out_b = predictor.predict(dem_b, np.full((2, 5), 25.0), h0_b, 5)
    assert out_b["water_depth"].shape == (2, 5, 8, 8)
    assert out_b["metadata"]["batched"] is True


def test_negative_depth_reporting_and_clamping():
    def _neg_model(dem, rainfall, h0, steps):
        out = np.full((steps, *h0.shape), -0.05, dtype=np.float32)
        return out

    dem = np.zeros((4, 4), dtype=np.float32)
    h0 = np.zeros((4, 4), dtype=np.float32)

    raw_res = predict(dem, 0.0, h0, 3, model=_neg_model, clamp_negative=False)
    assert raw_res["metadata"]["raw_negative_cell_count"] == 48
    assert float(np.min(raw_res["water_depth"])) < 0.0

    clamped_res = predict(dem, 0.0, h0, 3, model=_neg_model, clamp_negative=True)
    assert clamped_res["metadata"]["raw_negative_cell_count"] == 48
    assert float(np.min(clamped_res["water_depth"])) == 0.0


def test_strict_contract_validation():
    predictor = PINOPredictor(model=_dummy_pino_model, contract=ModelContract())
    dem = np.zeros((4, 4), dtype=np.float32)
    h0 = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="must match dem shape"):
        predictor.predict(dem, 10.0, np.zeros((5, 5), dtype=np.float32), 3)

    with pytest.raises(ValueError, match="forecast_steps must be >= 1"):
        predictor.predict(dem, 10.0, h0, 0)

    with pytest.raises(ValueError, match="NaN or Inf"):
        bad_dem = dem.copy()
        bad_dem[0, 0] = np.nan
        predictor.predict(bad_dem, 10.0, h0, 3)


def test_load_model_errors():
    with pytest.raises(FileNotFoundError):
        load_model("nonexistent_checkpoint.pt")
    with pytest.raises(ValueError, match="must be provided"):
        predict(np.zeros((4, 4)), 0.0, np.zeros((4, 4)), 2)
