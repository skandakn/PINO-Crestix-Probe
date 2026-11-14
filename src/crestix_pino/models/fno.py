"""3D Spatio-Temporal Fourier Neural Operator (FNO) for 2D+T flood prediction.

Tensor convention:
    Input x:  (B, C_in, T, H, W)
    Output y: (B, C_out, T, H, W)
where:
    B = batch size
    T = number of time steps
    H = spatial grid height (rows, y-axis)
    W = spatial grid width (cols, x-axis)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    """3D Fourier layer: FFT -> linear transform on low-frequency modes -> inverse FFT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_t: int,
        modes_y: int,
        modes_x: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_t, modes_y, modes_x)
        # Four quadrants in (T, Y) for real-to-complex rfftn along last axis (X)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (B, in_c, T, Y, X) x (in_c, out_c, T, Y, X) -> (B, out_c, T, Y, X)
        return torch.einsum("bityx,iotyx->botyx", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, t_dim, y_dim, x_dim = x.shape

        # Compute 3D real FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-3],
            x_ft.shape[-2],
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )

        mt = min(self.modes_t, max(1, t_dim // 2))
        my = min(self.modes_y, max(1, y_dim // 2))
        mx = min(self.modes_x, x_ft.shape[-1])

        out_ft[:, :, :mt, :my, :mx] = self._compl_mul3d(
            x_ft[:, :, :mt, :my, :mx],
            self.weights1[:, :, :mt, :my, :mx],
        )
        out_ft[:, :, -mt:, :my, :mx] = self._compl_mul3d(
            x_ft[:, :, -mt:, :my, :mx],
            self.weights2[:, :, :mt, :my, :mx],
        )
        out_ft[:, :, :mt, -my:, :mx] = self._compl_mul3d(
            x_ft[:, :, :mt, -my:, :mx],
            self.weights3[:, :, :mt, :my, :mx],
        )
        out_ft[:, :, -mt:, -my:, :mx] = self._compl_mul3d(
            x_ft[:, :, -mt:, -my:, :mx],
            self.weights4[:, :, :mt, :my, :mx],
        )

        return torch.fft.irfftn(out_ft, s=(t_dim, y_dim, x_dim), dim=(-3, -2, -1))


class FNOBlock3d(nn.Module):
    """Single FNO block: SpectralConv3d + pointwise Conv3d bypass + GELU."""

    def __init__(
        self,
        width: int,
        modes_t: int,
        modes_y: int,
        modes_x: int,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes_t, modes_y, modes_x)
        self.pointwise_conv = nn.Conv3d(width, width, kernel_size=1)
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral_conv(x) + self.pointwise_conv(x))


class FNO3d(nn.Module):
    """3D Fourier Neural Operator mapping spatio-temporal fields (B, C_in, T, H, W) -> (B, C_out, T, H, W)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        width: int = 32,
        modes_t: int = 4,
        modes_y: int = 12,
        modes_x: int = 12,
        n_layers: int = 4,
        padding_t: int = 2,
        padding_y: int = 4,
        padding_x: int = 4,
        append_grid_coords: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.append_grid_coords = append_grid_coords
        self.padding_t = padding_t
        self.padding_y = padding_y
        self.padding_x = padding_x

        lifted_in = in_channels + (3 if append_grid_coords else 0)
        self.lift = nn.Conv3d(lifted_in, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                FNOBlock3d(
                    width=width,
                    modes_t=modes_t,
                    modes_y=modes_y,
                    modes_x=modes_x,
                    activation=(i < n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

        self.proj = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, kernel_size=1),
        )

    @staticmethod
    def _build_grid(batch_size: int, t_dim: int, y_dim: int, x_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_coords = torch.linspace(0.0, 1.0, t_dim, device=device, dtype=dtype)
        y_coords = torch.linspace(0.0, 1.0, y_dim, device=device, dtype=dtype)
        x_coords = torch.linspace(0.0, 1.0, x_dim, device=device, dtype=dtype)
        tt, yy, xx = torch.meshgrid(t_coords, y_coords, x_coords, indexing="ij")
        grid = torch.stack([tt, yy, xx], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        return grid

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute latent spatio-temporal feature tensor of shape (B, width, T, H, W)."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C_in, T, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        if self.append_grid_coords:
            b, _, t_dim, y_dim, x_dim = x.shape
            grid = self._build_grid(b, t_dim, y_dim, x_dim, x.device, x.dtype)
            x = torch.cat([x, grid], dim=1)

        x = self.lift(x)

        # Zero-pad non-periodic boundaries for FFT stability
        pad = (0, self.padding_x, 0, self.padding_y, 0, self.padding_t)
        if any(p > 0 for p in pad):
            x = F.pad(x, pad)

        for block in self.blocks:
            x = block(x)

        # Remove padding
        if self.padding_t > 0:
            x = x[..., :-self.padding_t, :, :]
        if self.padding_y > 0:
            x = x[..., :, :-self.padding_y, :]
        if self.padding_x > 0:
            x = x[..., :, :, :-self.padding_x]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.proj(features)
