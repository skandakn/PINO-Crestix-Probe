import pytest
import torch

from crestix_pino.losses import PhysicsLoss
from crestix_pino.models import PINOFloodModel
from crestix_pino.physics import mm_per_hour_to_m_per_sec


def test_pino_build_input_tensor_and_forward_shapes() -> None:
    b, t_dim, h_dim, w_dim = 2, 5, 12, 14
    dem = torch.rand(b, h_dim, w_dim) * 10.0
    rain_m_s = mm_per_hour_to_m_per_sec(torch.full((b, t_dim, h_dim, w_dim), 50.0))
    h_init = torch.zeros(b, h_dim, w_dim)

    x = PINOFloodModel.build_input_tensor(
        dem_m=dem,
        rainfall_m_s=rain_m_s,
        initial_depth_m=h_init,
        manning_n=0.035,
    )
    assert x.shape == (b, 4, t_dim, h_dim, w_dim)

    model = PINOFloodModel(
        in_channels=4,
        width=16,
        modes_t=2,
        modes_y=4,
        modes_x=4,
        n_layers=2,
        predict_latent_velocity=True,
        nonnegative_depth_mode="softplus",
    )
    out = model(x)
    assert out.h.shape == (b, t_dim, h_dim, w_dim)
    assert out.u is not None and out.u.shape == (b, t_dim, h_dim, w_dim)
    assert out.v is not None and out.v.shape == (b, t_dim, h_dim, w_dim)
    assert (out.h >= 0.0).all()


def test_pino_without_latent_velocity_raises_in_physics_loss() -> None:
    model = PINOFloodModel(
        in_channels=4,
        width=8,
        modes_t=2,
        modes_y=3,
        modes_x=3,
        n_layers=2,
        predict_latent_velocity=False,
    )
    b, t_dim, h_dim, w_dim = 1, 4, 8, 8
    dem = torch.zeros(b, h_dim, w_dim)
    rain = torch.zeros(b, t_dim, h_dim, w_dim)
    h_init = torch.zeros(b, h_dim, w_dim)
    x = PINOFloodModel.build_input_tensor(dem, rain, h_init)
    out = model(x)
    assert out.u is None and out.v is None

    loss_fn = PhysicsLoss()
    with pytest.raises(ValueError, match="latent velocity"):
        loss_fn(out, dem=dem, rainfall_m_s=rain, h_init=h_init)
