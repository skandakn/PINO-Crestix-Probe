"""Model contract and checkpoint loader for Crestix Probe PINO inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ModelContract:
    """Agreed input/output tensor and SI-unit contract for PINO flood forecasting.

    Units:
        - dem: meters above datum (m)
        - rainfall: millimeters per hour (mm/h)
        - initial_water_depth: meters (m)
        - water_depth (output): meters (m)
    """

    dem_units: str = "m"
    rainfall_units: str = "mm/h"
    depth_units: str = "m"
    spatial_rank: int = 2
    output_layout: str = "(T, H, W) for unbatched or (B, T, H, W) for batched"
    supports_velocity_output: bool = False
    dt_seconds: float = 30.0
    cell_size_m: float = 10.0


@dataclass
class LoadedModel:
    """Container pairing a loaded PINO model callable with its contract and metadata."""

    backend: Callable[..., Any]
    contract: ModelContract = field(default_factory=ModelContract)
    checkpoint_path: str | None = None
    device: str = "cpu"
    framework: str = "custom"

    def __call__(
        self,
        dem: Any,
        rainfall: Any,
        initial_water_depth: Any,
        forecast_steps: int,
    ) -> Any:
        return self.backend(dem, rainfall, initial_water_depth, forecast_steps)


def _wrap_pino_model(
    pino_model: Any,
    device: str,
    checkpoint_path: str | None = None,
    contract: ModelContract | None = None,
    framework: str = "pytorch_pino",
) -> LoadedModel:
    """Wrap a `PINOFloodModel` instance into the unified `LoadedModel` contract."""
    import numpy as np
    import torch

    pino_model.to(device)
    pino_model.eval()
    meta = pino_model.grid_metadata
    pino_contract = contract or ModelContract(
        dt_seconds=meta.dt,
        cell_size_m=meta.dx,
    )

    def _pino_backend(
        dem: Any,
        rainfall: Any,
        initial_water_depth: Any,
        forecast_steps: int,
    ) -> Any:
        with torch.no_grad():
            dem_arr = np.asarray(dem, dtype=np.float32)
            rain_arr = np.asarray(rainfall, dtype=np.float32)
            h0_arr = np.asarray(initial_water_depth, dtype=np.float32)
            unbatched = dem_arr.ndim == 2

            dem_t = torch.as_tensor(dem_arr, dtype=torch.float32, device=device)
            rain_t = torch.as_tensor(rain_arr, dtype=torch.float32, device=device) / 3_600_000.0
            if rain_t.ndim == 0:
                h_dim, w_dim = dem_arr.shape[-2], dem_arr.shape[-1]
                b_dim = 1 if unbatched else dem_arr.shape[0]
                rain_t = rain_t.view(1, 1, 1, 1).expand(b_dim, int(forecast_steps), h_dim, w_dim)
            elif rain_t.ndim == 2 and unbatched:
                rain_t = rain_t.unsqueeze(0).expand(int(forecast_steps), -1, -1)
            h0_t = torch.as_tensor(h0_arr, dtype=torch.float32, device=device)

            valid_mask = torch.isfinite(dem_t) & (dem_t > -9000.0)
            out = pino_model.forecast(
                dem_m=dem_t,
                rainfall_m_s=rain_t,
                initial_depth_m=h0_t,
                forecast_steps=int(forecast_steps),
                valid_mask=None if bool(valid_mask.all()) else valid_mask,
                wall_height=50.0,
                default_manning_n=0.035,
            )
            h_np = out.h.detach().cpu().numpy()
            return h_np[0] if unbatched else h_np

    return LoadedModel(
        backend=_pino_backend,
        contract=pino_contract,
        checkpoint_path=checkpoint_path,
        device=device,
        framework=framework,
    )


def _instantiate_pino_from_checkpoint_payload(payload: dict[str, Any]) -> Any:
    """Reconstruct a `PINOFloodModel` from a `save_checkpoint` dictionary payload."""
    from crestix_pino.models.pino import GridMetadata, PINOFloodModel, PINOModelConfig

    cfg = payload.get("config") or {}
    m_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    p_cfg = cfg.get("physics", {}) if isinstance(cfg, dict) else {}
    d_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    meta = GridMetadata(
        dx=float(p_cfg.get("dx", 8.0)),
        dy=float(p_cfg.get("dy", 8.0)),
        dt=float(p_cfg.get("dt", 300.0)),
        grid_height=int(d_cfg.get("region_height", 32)),
        grid_width=int(d_cfg.get("region_width", 32)),
    )
    pino_cfg = PINOModelConfig(
        in_channels=int(m_cfg.get("in_channels", 4)),
        width=int(m_cfg.get("width", 32)),
        modes_t=int(m_cfg.get("modes_t", 4)),
        modes_y=int(m_cfg.get("modes_y", 12)),
        modes_x=int(m_cfg.get("modes_x", 12)),
        n_layers=int(m_cfg.get("n_layers", 4)),
        padding_t=int(m_cfg.get("padding_t", 2)),
        padding_y=int(m_cfg.get("padding_y", 4)),
        padding_x=int(m_cfg.get("padding_x", 4)),
        append_grid_coords=bool(m_cfg.get("append_grid_coords", True)),
        predict_latent_velocity=bool(m_cfg.get("predict_latent_velocity", True)),
        nonnegative_depth_mode=m_cfg.get("nonnegative_depth_mode", "softplus"),
        forecast_steps=int(d_cfg.get("window_t", 12)),
        wall_height=float(d_cfg.get("wall_height", 50.0)),
        default_manning_n=float(p_cfg.get("default_manning_n", d_cfg.get("manning_n", 0.035))),
        grid_metadata=meta,
    )
    model = PINOFloodModel.from_config(pino_cfg)
    state = payload.get("model_state_dict") or payload.get("state_dict")
    model.load_state_dict(state)
    return model


def load_model(
    checkpoint: str | Path | Callable[..., Any] | LoadedModel,
    device: str = "cpu",
    model_factory: Callable[[], Any] | None = None,
    contract: ModelContract | None = None,
) -> LoadedModel:
    """Load a trained PINO model from a checkpoint file or Python callable.

    Supports:
        - Already-wrapped `LoadedModel` instances.
        - Python callables / objects with signature `(dem, rainfall, initial_water_depth, forecast_steps)`.
        - Training checkpoints saved via `save_checkpoint` (`{"model_state_dict": ..., "config": ...}`).
        - TorchScript archives (`.pt`, `.pts`) via `torch.jit.load`.
        - PyTorch `state_dict` checkpoints (`.pth`, `.pt`) when `model_factory` is provided.
    """
    active_contract = contract or ModelContract()

    if isinstance(checkpoint, LoadedModel):
        return checkpoint

    try:
        import torch
        from crestix_pino.models.pino import PINOFloodModel
    except ImportError:
        PINOFloodModel = None  # type: ignore[assignment,misc]

    if PINOFloodModel is not None and isinstance(checkpoint, PINOFloodModel):
        return _wrap_pino_model(checkpoint, device=device, contract=contract, framework="pytorch_pino")

    if callable(checkpoint) and not isinstance(checkpoint, (str, Path)):
        return LoadedModel(
            backend=checkpoint,
            contract=active_contract,
            checkpoint_path=None,
            device=device,
            framework="callable",
        )

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".pt", ".pts", ".pth"}:
        raise ValueError(
            f"Unsupported checkpoint extension '{suffix}' for {path}. "
            "Expected .pt, .pts, or .pth, or pass a callable model."
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .pt/.pts/.pth checkpoints. "
            "Install it with `pip install torch`."
        ) from exc

    if model_factory is not None:
        model = model_factory()
        state = torch.load(path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        framework = "pytorch_state_dict"
    else:
        try:
            loaded_obj = torch.load(path, map_location=device, weights_only=False)
        except Exception:
            loaded_obj = None

        if isinstance(loaded_obj, dict) and (
            "model_state_dict" in loaded_obj or "state_dict" in loaded_obj
        ):
            pino_model = _instantiate_pino_from_checkpoint_payload(loaded_obj)
            return _wrap_pino_model(
                pino_model,
                device=device,
                checkpoint_path=str(path),
                contract=contract,
                framework="pytorch_pino_checkpoint",
            )

        if isinstance(loaded_obj, torch.jit.ScriptModule):
            model = loaded_obj
        else:
            model = torch.jit.load(str(path), map_location=device)
        model.eval()
        framework = "torchscript"

    def _torch_backend(
        dem: Any,
        rainfall: Any,
        initial_water_depth: Any,
        forecast_steps: int,
    ) -> Any:
        with torch.no_grad():
            dem_t = torch.as_tensor(dem, dtype=torch.float32, device=device)
            rain_t = torch.as_tensor(rainfall, dtype=torch.float32, device=device)
            h0_t = torch.as_tensor(initial_water_depth, dtype=torch.float32, device=device)
            out = model(dem_t, rain_t, h0_t, int(forecast_steps))
            if isinstance(out, dict):
                out = out["water_depth"]
            return out.detach().cpu().numpy()

    return LoadedModel(
        backend=_torch_backend,
        contract=active_contract,
        checkpoint_path=str(path),
        device=device,
        framework=framework,
    )
