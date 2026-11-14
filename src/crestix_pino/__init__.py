"""Crestix Probe — Physics-Informed Fourier Neural Operator (PINO) for 2D Flood Prediction."""

__version__ = "0.1.0"

from crestix_pino.inference.model_loader import ModelContract, load_model
from crestix_pino.inference.predictor import PINOPredictor, predict

from .losses import (
    DataLoss,
    PINOCompositeLoss,
    PhysicsLoss,
    PhysicsLossWeights,
)
from .models import FNO3d, PINOFloodModel, PINOOutput
from .physics import (
    BoundaryConfig,
    ShallowWaterConfig,
    compute_swe_residuals,
    mm_per_hour_to_m_per_sec,
)

__all__ = [
    "BoundaryConfig",
    "DataLoss",
    "FNO3d",
    "ModelContract",
    "PINOCompositeLoss",
    "PINOFloodModel",
    "PINOOutput",
    "PINOPredictor",
    "PhysicsLoss",
    "PhysicsLossWeights",
    "ShallowWaterConfig",
    "__version__",
    "compute_swe_residuals",
    "load_model",
    "mm_per_hour_to_m_per_sec",
    "predict",
]
