import torch
from crestix_pino.models.fno import FNO3d, SpectralConv3d


def test_spectral_conv3d_shape_and_gradients() -> None:
    layer = SpectralConv3d(in_channels=8, out_channels=8, modes_t=3, modes_y=4, modes_x=4)
    x = torch.randn(2, 8, 6, 12, 14, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fno3d_forward_with_grid_coords() -> None:
    model = FNO3d(
        in_channels=4,
        out_channels=1,
        width=16,
        modes_t=2,
        modes_y=4,
        modes_x=4,
        n_layers=3,
        padding_t=1,
        padding_y=2,
        padding_x=2,
        append_grid_coords=True,
    )
    x = torch.randn(2, 4, 5, 10, 12)
    y = model(x)
    assert y.shape == (2, 1, 5, 10, 12)
    assert torch.isfinite(y).all()


def test_fno3d_handles_small_spatial_dims() -> None:
    # Ensure mode clamping works safely when grid is smaller than nominal modes
    model = FNO3d(
        in_channels=4,
        out_channels=2,
        width=8,
        modes_t=8,
        modes_y=16,
        modes_x=16,
        n_layers=2,
        padding_t=0,
        padding_y=0,
        padding_x=0,
    )
    x = torch.randn(1, 4, 4, 6, 6)
    y = model(x)
    assert y.shape == (1, 2, 4, 6, 6)
