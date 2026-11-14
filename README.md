# PINO-Crestix-Probe

Evaluation, benchmarking, and inference packaging for the Crestix Probe Physics-Informed Neural Operator (PINO) flood forecasting model.

## Folder Structure

```text
PINO-Crestix-Probe/
├── src/crestix_pino/
│   ├── evaluation/
│   │   ├── metrics.py          # MAE, RMSE, R^2, relative depth, mass conservation, physics residual, horizon metrics
│   │   ├── evaluate.py         # Held-out flood event evaluation runner against LarNO ground truth
│   │   └── visualization.py    # Spatial error maps, time-series comparisons, horizon plots
│   └── inference/
│       ├── predictor.py        # Preprocessing, inference, postprocessing, and `predict(...)` contract
│       └── model_loader.py     # ModelContract and TorchScript / PyTorch / callable checkpoint loader
├── scripts/
│   ├── evaluate_pino.py        # CLI evaluation script
│   └── benchmark_inference.py  # CLI latency benchmark (model-only vs. full pipeline)
├── reports/
│   └── README.md               # Model card, limitations, and evaluation report template
├── tests/                      # Unit tests for metrics, inference, visualization, and evaluation
└── README.md
```

## Installation

```bash
pip install -e ".[dev]"
# Optional: install PyTorch for .pt / .pts / .pth checkpoints
pip install -e ".[torch,dev]"
```

## Inference Example

```python
import numpy as np
from crestix_pino import predict

dem = np.zeros((32, 32), dtype=np.float32)                 # meters (m)
rainfall = np.full((12,), 45.0, dtype=np.float32)          # mm/h
initial_water_depth = np.zeros((32, 32), dtype=np.float32) # meters (m)

result = predict(
    dem=dem,
    rainfall=rainfall,
    initial_water_depth=initial_water_depth,
    forecast_steps=12,
    model="path/to/best_pino_model.pt",
    device="cpu",
)

water_depth = result["water_depth"]  # shape: (12, 32, 32), units: m
metadata = result["metadata"]        # units, shape, latency_ms, negative-depth stats
```

## Running Evaluation & Benchmarks

```bash
# Evaluate trained PINO checkpoint against held-out LarNO .npz flood events
python scripts/evaluate_pino.py \
  --checkpoint path/to/best_pino_model.pt \
  --dataset path/to/held_out_larno_events/ \
  --output-dir outputs/evaluation

# Benchmark model-only vs. full-pipeline latency
python scripts/benchmark_inference.py \
  --checkpoint path/to/best_pino_model.pt \
  --batch-size 1 --height 64 --width 64 --steps 24 \
  --output-json outputs/benchmark.json
```

## Running Tests

```bash
pytest -v
```

## Google Drive Deliverables Checklist

Once trained checkpoints and held-out LarNO datasets are available, archive to Google Drive:
- Best model checkpoints (`.pt` / `.pth`)
- Evaluation logs (`outputs/evaluation/evaluation_summary.json` and per-event `*_metrics.json`)
- Predicted water-depth arrays (`*_predictions.npz`)
- Error maps & plots (`*_spatial_error.png`, `*_time_series.png`, `*_horizon_metrics.png`)
- Benchmark outputs (`outputs/benchmark.json`)
- Experiment reports (`reports/README.md`)
