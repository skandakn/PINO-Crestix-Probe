import torch

from crestix_pino.losses import (
    DataLoss,
    PINOCompositeLoss,
    PhysicsLoss,
    PhysicsLossWeights,
    compute_initial_condition_loss,
    compute_nonnegative_depth_loss,
)
from crestix_pino.models import PINOFloodModel
from crestix_pino.physics import BoundaryConfig, ShallowWaterConfig, mm_per_hour_to_m_per_sec


def test_data_loss_modes_and_mask() -> None:
    pred = torch.tensor([[[[1.0, 3.0], [2.0, 99.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [2.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])

    mse_loss = DataLoss(loss_type="mse")
    # Valid cells errors: 0^2, 2^2, 0^2 -> mean = 4 / 3
    assert torch.isclose(mse_loss(pred, target, valid_mask=mask), torch.tensor(4.0 / 3.0))

    l1_loss = DataLoss(loss_type="l1")
    # Valid cells errors: 0, 2, 0 -> mean = 2 / 3
    assert torch.isclose(l1_loss(pred, target, valid_mask=mask), torch.tensor(2.0 / 3.0))

    rel_l2 = DataLoss(loss_type="rel_l2")
    val = rel_l2(pred, target, valid_mask=mask)
    assert val.item() > 0.0


def test_nonnegative_and_initial_condition_losses() -> None:
    h_neg = torch.tensor([[[[-0.5, 0.2], [0.0, -1.0]]]])
    loss_neg = compute_nonnegative_depth_loss(h_neg)
    expected_neg = (0.25 + 0.0 + 0.0 + 1.0) / 4.0
    assert torch.isclose(loss_neg, torch.tensor(expected_neg))

    h_pred = torch.zeros(1, 3, 4, 4)
    h_pred[:, 0] = 0.5
    h_init = torch.full((1, 4, 4), 0.5)
    assert compute_initial_condition_loss(h_pred, h_init).item() == 0.0


def test_end_to_end_pino_composite_loss_backprop() -> None:
    b, t_dim, h_dim, w_dim = 2, 4, 8, 8
    dem = torch.rand(b, h_dim, w_dim) * 5.0
    rain = mm_per_hour_to_m_per_sec(torch.full((b, t_dim, h_dim, w_dim), 40.0))
    h_init = torch.full((b, h_dim, w_dim), 0.05)
    h_true = torch.full((b, t_dim, h_dim, w_dim), 0.08)

    model = PINOFloodModel(
        in_channels=4,
        width=8,
        modes_t=2,
        modes_y=3,
        modes_x=3,
        n_layers=2,
        padding_t=1,
        padding_y=1,
        padding_x=1,
        predict_latent_velocity=True,
    )
    x = PINOFloodModel.build_input_tensor(dem, rain, h_init, manning_n=0.035)
    out = model(x)

    criterion = PINOCompositeLoss(
        data_loss=DataLoss("mse"),
        physics_loss=PhysicsLoss(
            sw_config=ShallowWaterConfig(dx=2.0, dy=2.0, dt=60.0),
            boundary_config=BoundaryConfig(north="closed", south="open", east="closed", west="closed"),
            weights=PhysicsLossWeights(),
        ),
    )

    total_loss, breakdown = criterion(
        output=out,
        h_true=h_true,
        dem=dem,
        rainfall_m_s=rain,
        h_init=h_init,
        manning_n=0.035,
    )

    expected_keys = {
        "data_loss",
        "pde_continuity",
        "pde_momentum_x",
        "pde_momentum_y",
        "mass_conservation",
        "boundary_condition",
        "initial_condition",
        "nonnegative_depth",
        "total_physics",
        "total_loss",
    }
    assert expected_keys.issubset(set(breakdown.keys()))
    assert torch.isfinite(total_loss)

    total_loss.backward()
    # Verify gradients propagate to model parameters
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
