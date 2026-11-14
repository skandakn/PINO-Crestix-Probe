"""Tests for PINO audit enhancements: GridMetadata, masked invalid DEMs, horizons, and inference."""

import numpy as np
import torch

from crestix_pino.inference.predictor import PINOPredictor
from crestix_pino.losses.physics_loss import PINOCompositeLoss
from crestix_pino.models.pino import GridMetadata, PINOFloodModel, PINOModelConfig
from crestix_pino.physics.shallow_water import ShallowWaterConfig


def test_pino_from_config_and_masked_invalid_dem_cells() -> None:
    cfg = PINOModelConfig(
        width=8,
        modes_t=2,
        modes_y=4,
        modes_x=4,
        n_layers=2,
        padding_t=1,
        padding_y=2,
        padding_x=2,
        input_steps=2,
        forecast_steps=4,
        grid_metadata=GridMetadata(dx=8.0, dy=8.0, dt=300.0, grid_height=12, grid_width=12),
    )
    model = PINOFloodModel.from_config(cfg)

    dem = torch.full((2, 12, 12), 10.0)
    dem[:, :3, :3] = float("nan")  # Building / invalid DEM cells
    rainfall = torch.full((2, 4, 12, 12), 1e-5)
    h_context = torch.full((2, 2, 12, 12), 0.05)  # T_in = 2 historical steps

    out = model.forecast(dem_m=dem, rainfall_m_s=rainfall, initial_depth_m=h_context, forecast_steps=4)
    assert out.h.shape == (2, 4, 12, 12)
    assert out.valid_mask is not None
    assert torch.isfinite(out.h).all()
    assert torch.all(out.h[:, :, :3, :3] == 0.0)
    assert torch.all(out.h >= 0.0)


def test_physics_loss_with_valid_mask_backprop() -> None:
    model = PINOFloodModel(width=8, modes_t=2, modes_y=4, modes_x=4, n_layers=2)
    dem = torch.full((1, 10, 10), 5.0)
    valid_mask = torch.ones((1, 10, 10), dtype=torch.bool)
    valid_mask[:, 0:2, 0:2] = False
    rain = torch.full((1, 4, 10, 10), 1e-5)
    h0 = torch.zeros((1, 10, 10))

    x = PINOFloodModel.build_input_tensor(dem, rain, h0)
    out = model(x, valid_mask=valid_mask)
    target = torch.full((1, 4, 10, 10), 0.02) * valid_mask.unsqueeze(1).float()

    loss_fn = PINOCompositeLoss()
    loss_fn.physics_loss.sw_config = ShallowWaterConfig(dx=8.0, dy=8.0, dt=300.0)
    total, breakdown = loss_fn(
        output=out,
        h_true=target,
        dem=dem,
        rainfall_m_s=rain,
        h_init=h0,
        valid_mask=valid_mask,
    )
    assert torch.isfinite(total)
    total.backward()
    assert "total_physics" in breakdown


def test_pino_predictor_integration_with_nan_dem() -> None:
    model = PINOFloodModel(
        width=8,
        modes_t=2,
        modes_y=4,
        modes_x=4,
        n_layers=2,
        grid_metadata=GridMetadata(dx=8.0, dy=8.0, dt=300.0, grid_height=10, grid_width=10),
    )
    predictor = PINOPredictor(model=model, allow_dem_nan=True)

    dem = np.full((10, 10), 12.0, dtype=np.float32)
    dem[:2, :2] = np.nan
    rainfall = np.full((5,), 36.0, dtype=np.float32)  # mm/h
    h0 = np.zeros((10, 10), dtype=np.float32)

    res = predictor.predict(dem=dem, rainfall=rainfall, initial_water_depth=h0, forecast_steps=5)
    assert res["water_depth"].shape == (5, 10, 10)
    assert res["metadata"]["framework"] == "pytorch_pino"
    assert np.all(res["water_depth"][:, :2, :2] == 0.0)
    assert np.all(res["water_depth"] >= 0.0)
