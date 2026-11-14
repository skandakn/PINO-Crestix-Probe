"""Clean PINO inference pipeline: preprocessing, model execution, and postprocessing."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from crestix_pino.inference.model_loader import LoadedModel, ModelContract, load_model


class PINOPredictor:
    """End-to-end inference wrapper adhering to the Crestix Probe PINO model contract."""

    def __init__(
        self,
        model: str | Callable[..., Any] | LoadedModel,
        device: str = "cpu",
        model_factory: Callable[[], Any] | None = None,
        contract: ModelContract | None = None,
        clamp_negative: bool = False,
        allow_dem_nan: bool = False,
    ) -> None:
        self.loaded_model = load_model(
            model,
            device=device,
            model_factory=model_factory,
            contract=contract,
        )
        self.contract = self.loaded_model.contract
        self.clamp_negative = clamp_negative
        self.allow_dem_nan = allow_dem_nan

    def preprocess(
        self,
        dem: Any,
        rainfall: Any,
        initial_water_depth: Any,
        forecast_steps: int,
    ) -> dict[str, Any]:
        """Validate dimensions and types without silently altering tensor rank or units."""
        if isinstance(forecast_steps, bool) or not isinstance(forecast_steps, (int, np.integer)):
            raise TypeError(f"forecast_steps must be an integer, got {type(forecast_steps).__name__}.")
        steps = int(forecast_steps)
        if steps < 1:
            raise ValueError(f"forecast_steps must be >= 1, got {steps}.")

        dem_arr = np.asarray(dem, dtype=np.float32)
        h0_arr = np.asarray(initial_water_depth, dtype=np.float32)
        rain_arr = np.asarray(rainfall, dtype=np.float32)

        for name, arr in (
            ("dem", dem_arr),
            ("initial_water_depth", h0_arr),
            ("rainfall", rain_arr),
        ):
            if name == "dem" and self.allow_dem_nan:
                if np.any(np.isinf(arr)):
                    raise ValueError("Input 'dem' contains Inf values.")
            elif not np.all(np.isfinite(arr)):
                raise ValueError(f"Input '{name}' contains NaN or Inf values.")

        if dem_arr.ndim not in (2, 3):
            raise ValueError(
                f"dem must be 2D (H, W) or 3D (B, H, W), got shape {dem_arr.shape}."
            )
        if h0_arr.shape != dem_arr.shape:
            raise ValueError(
                f"initial_water_depth shape {h0_arr.shape} must match dem shape {dem_arr.shape}."
            )
        if np.any(h0_arr < 0):
            raise ValueError("initial_water_depth must be nonnegative (meters).")
        if np.any(rain_arr < 0):
            raise ValueError("rainfall must be nonnegative (mm/h).")

        self._validate_rainfall_shape(rain_arr, dem_arr, steps)

        return {
            "dem": dem_arr,
            "rainfall": rain_arr,
            "initial_water_depth": h0_arr,
            "forecast_steps": steps,
            "batched": dem_arr.ndim == 3,
        }

    @staticmethod
    def _validate_rainfall_shape(
        rain_arr: np.ndarray, dem_arr: np.ndarray, steps: int
    ) -> None:
        if dem_arr.ndim == 2:
            h, w = dem_arr.shape
            valid = (
                rain_arr.ndim == 0
                or rain_arr.shape == (steps,)
                or rain_arr.shape == (h, w)
                or rain_arr.shape == (steps, h, w)
            )
        else:
            b, h, w = dem_arr.shape
            valid = (
                rain_arr.ndim == 0
                or rain_arr.shape == (steps,)
                or rain_arr.shape == (b, steps)
                or rain_arr.shape == (b, h, w)
                or rain_arr.shape == (b, steps, h, w)
            )
        if not valid:
            raise ValueError(
                f"Incompatible rainfall shape {rain_arr.shape} for dem shape {dem_arr.shape} "
                f"and forecast_steps={steps}."
            )

    def infer_model_only(self, prepared: dict[str, Any]) -> np.ndarray:
        """Execute raw model forward pass on already-preprocessed arrays."""
        raw_out = self.loaded_model(
            prepared["dem"],
            prepared["rainfall"],
            prepared["initial_water_depth"],
            prepared["forecast_steps"],
        )
        return np.asarray(raw_out, dtype=np.float32)

    def postprocess(
        self,
        raw_depth: np.ndarray,
        prepared: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate output shape, record negative-depth statistics, and optionally clamp."""
        dem_arr = prepared["dem"]
        steps = prepared["forecast_steps"]
        expected_shape = (
            (dem_arr.shape[0], steps, dem_arr.shape[1], dem_arr.shape[2])
            if prepared["batched"]
            else (steps, dem_arr.shape[0], dem_arr.shape[1])
        )
        if raw_depth.shape != expected_shape:
            raise ValueError(
                f"Model output shape {raw_depth.shape} does not match expected "
                f"contract shape {expected_shape}."
            )
        if not np.all(np.isfinite(raw_depth)):
            raise ValueError("Model produced non-finite (NaN or Inf) water_depth values.")

        neg_mask = raw_depth < 0.0
        neg_count = int(np.sum(neg_mask))
        min_raw = float(np.min(raw_depth))

        water_depth = np.maximum(raw_depth, 0.0) if self.clamp_negative else raw_depth
        stats = {
            "raw_negative_cell_count": neg_count,
            "raw_min_depth_m": min_raw,
            "clamped_negative": self.clamp_negative,
        }
        return water_depth, stats

    def predict(
        self,
        dem: Any,
        rainfall: Any,
        initial_water_depth: Any,
        forecast_steps: int,
    ) -> dict[str, Any]:
        """Run full preprocessing -> model inference -> postprocessing pipeline."""
        t0 = time.perf_counter()
        prepared = self.preprocess(dem, rainfall, initial_water_depth, forecast_steps)
        t1 = time.perf_counter()
        raw_depth = self.infer_model_only(prepared)
        t2 = time.perf_counter()
        water_depth, post_stats = self.postprocess(raw_depth, prepared)
        t3 = time.perf_counter()

        return {
            "water_depth": water_depth,
            "metadata": {
                "units": {
                    "dem": self.contract.dem_units,
                    "rainfall": self.contract.rainfall_units,
                    "water_depth": self.contract.depth_units,
                },
                "shape": list(water_depth.shape),
                "dtype": str(water_depth.dtype),
                "forecast_steps": prepared["forecast_steps"],
                "batched": prepared["batched"],
                "device": self.loaded_model.device,
                "framework": self.loaded_model.framework,
                "checkpoint_path": self.loaded_model.checkpoint_path,
                "latency_ms": {
                    "preprocess": (t1 - t0) * 1000.0,
                    "model_only": (t2 - t1) * 1000.0,
                    "postprocess": (t3 - t2) * 1000.0,
                    "total_pipeline": (t3 - t0) * 1000.0,
                },
                **post_stats,
            },
        }


def predict(
    dem: Any,
    rainfall: Any,
    initial_water_depth: Any,
    forecast_steps: int,
    *,
    model: str | Callable[..., Any] | LoadedModel | None = None,
    device: str = "cpu",
    clamp_negative: bool = False,
) -> dict[str, Any]:
    """Functional inference entrypoint matching the Crestix Probe PINO contract."""
    if model is None:
        raise ValueError(
            "A trained PINO model checkpoint path or callable must be provided via `model=...`."
        )
    predictor = PINOPredictor(model=model, device=device, clamp_negative=clamp_negative)
    return predictor.predict(
        dem=dem,
        rainfall=rainfall,
        initial_water_depth=initial_water_depth,
        forecast_steps=forecast_steps,
    )
