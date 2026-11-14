# Common Evaluation Framework — PINO vs. ConvLSTM U-Net (`shared_manifest.json` v1.1.0)

> **Primary Evaluation & Comparison Repository:** [`skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-)
> **Model Contract & Format Spec:** [`PREDICTION_FORMAT.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/PREDICTION_FORMAT.md) | [`common_evaluator.py`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/common_evaluator.py)

To avoid duplicating shared evaluation code across model repositories, the canonical evaluator, shared LarNO event manifest, comparison tables, comparison plots, and joint comparison report live in **`ConvLSTM-baased-U-Net-for-Crestix-Probe-`**. `PINO-Crestix-Probe` exports standardized held-out predictions and efficiency metadata against the exact same contract.

---

## 1. Non-Negotiable Shared Contract (`shared_manifest.json` v1.1.0)

Both **PINO** and **ConvLSTM-based U-Net** must satisfy every field of [`shared_manifest.json`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/shared_manifest.json):

| Contract Field | Agreed Value | Notes for PINO (`ukea_32x32.yaml` / `PINOPredictor`) |
|---|---|---|
| `manifest_version` | `1.1.0` | Enforced via `manifest_sha256` (`da79e0b11e6d4e85006c9d8a9da8aaf69664a0f251f0a7001c7db62e210ae0dd`) |
| `dataset` / `region_tag` | `holmescao/LarNO-dataset` / `ukea_8m_5min` | Native 8 m DEM, 300 s water-depth frames |
| `spatial_crop` | `crop_y=9, crop_x=44, 32x32, cell_size_m=8.0` | Matches `configs/ukea_32x32.yaml` |
| `train_events` (6) | `r300y_p0.1_d3h_1`, `r300y_p0.4_d3h_1`, `r200y_p0.3_d3h_1`, `r100y_p0.7_d3h_1`, `r500y_p0.2_d3h_1`, `r300y_p0.6_d3h_1` | Do not retrain on any `val_events` or `test_events` |
| `val_events` (2) | `r200y_p0.5_d3h_1`, `r300y_p0.9_d3h_1` | Validation / checkpoint selection only |
| `test_events` (10) | `r100y_p0.6_d3h_1`, `r100y_p0.1_d3h_1`, `r300y_p0.8_d3h_1`, `r200y_p0.7_d3h_1`, `r500y_p0.3_d3h_1`, `r500y_p0.9_d3h_1`, `r500y_p0.1_d3h_1`, `r500y_p0.6_d3h_1`, `r100y_p0.5_d3h_1`, `r500y_p0.4_d3h_1` | Exact 10 held-out test events; differs from the 12-event upstream split in `configs/ukea_32x32.yaml` |
| `target_variable` / `target_units` | `water_depth_m` / `m` | Depth in meters (`(36, 32, 32)` `float32`) |
| `forecast_horizon_steps` / `dt_seconds` | `36` / `300.0` (3 h rollout) | Full 36-step horizon (`window_t=12` alone is not comparable) |
| `valid_cell_rule` | `non_building_and_finite_dem` | Pass `allow_dem_nan=True` in `PINOPredictor`; zero/mask building cells |
| `flood_threshold_m` | `0.03` | Used identically for binary F1, precision, recall, and confusion matrix |

---

## 2. Rainfall Unit Bridge (`mm` -> `mm/h` -> `m/s`)

Three unit conventions exist across raw LarNO files and the PINO codebase:
1. **Raw LarNO `rainfall.npy`:** Accumulated `mm` per simulation frame (often 30 s sub-steps stacked `360 x 50 x 120`, i.e., 10 frames per 300 s water-depth step). Sum every 10 frames to get 5-minute accumulated depth `r_5min_mm` of shape `(36, 32, 32)`.
2. **`PINOFloodModel.forecast(...)` / `configs/ukea_32x32.yaml`:** Expects `rainfall_m_s` in **`m/s`** (`r_5min_mm * (1e-3 / 300.0)` = `r_5min_mm * 3.3333333e-6`).
3. **`PINOPredictor.predict(...)` (`src/crestix_pino/inference/predictor.py`):** Expects `rainfall` in **`mm/h`** (`r_5min_mm * 12.0` or `rainfall_m_s * 3_600_000.0`), and `src/crestix_pino/inference/model_loader.py` (`_pino_backend`, line 99) automatically divides by `3_600_000.0` before calling `PINOFloodModel.forecast`.

**Wait:** Do not pass `m/s` into `PINOPredictor.predict(...)`, and do not pass `mm/h` directly into `PINOFloodModel.forecast(...)`.

---

## 3. Exporting Held-Out PINO Predictions (Python Snippet)

Run after training `PINOFloodModel` strictly on the 6 `train_events` (with early stopping on the 2 `val_events` in `shared_manifest.json`), with the ConvLSTM repository added to `PYTHONPATH` for `common_evaluator.py`:

```python
from pathlib import Path
import time
import numpy as np
import torch

from crestix_pino.inference.predictor import PINOPredictor
from crestix_pino.models.pino import GridMetadata, PINOFloodModel, PINOModelConfig
from common_evaluator import (
    load_manifest,
    save_standard_prediction,
    write_prediction_manifest,
)

# 1. Paths
CONVLSTM_REPO = Path("../ConvLSTM-baased-U-Net-for-Crestix-Probe-")
DATA_ROOT = Path("data/ukea_8m_5min")
CHECKPOINT_PATH = Path("artifacts/ukea_32x32/best_pino_model.pt")
OUT_DIR = Path("local_cache/pino_predictions_v1.1.0")

manifest = load_manifest(CONVLSTM_REPO / "shared_manifest.json")
crop = manifest["spatial_crop"]
y0, x0 = int(crop["crop_y"]), int(crop["crop_x"])
h_dim, w_dim = int(crop["height"]), int(crop["width"])
horizon = int(manifest["forecast_horizon_steps"])
dt_seconds = float(manifest["dt_seconds"])

# 2. Load trained PINOFloodModel for 32x32 (dx=8.0m, dt=300.0s, horizon=36)
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = PINOModelConfig(
    grid_metadata=GridMetadata(
        dx=float(crop["cell_size_m"]),
        dy=float(crop["cell_size_m"]),
        dt=dt_seconds,
        grid_height=h_dim,
        grid_width=w_dim,
    ),
    forecast_steps=horizon,
)
model = PINOFloodModel.from_config(cfg)
state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
predictor = PINOPredictor(model=model, device=device, allow_dem_nan=True)

# 3. Load static DEM crop and export each of the 10 held-out test events
dem_raw = np.load(DATA_ROOT / "geodata" / "dem.npy")
dem = dem_raw[y0 : y0 + h_dim, x0 : x0 + w_dim].astype(np.float32)
valid_mask = np.isfinite(dem) & (dem > -999.0)

latencies_ms = []
if device.startswith("cuda"):
    torch.cuda.reset_peak_memory_stats()

for event_id in manifest["test_events"]:
    ev_dir = DATA_ROOT / "flood" / event_id
    h_raw = np.load(ev_dir / "h.npy")[:, y0 : y0 + h_dim, x0 : x0 + w_dim]
    r_raw = np.load(ev_dir / "rainfall.npy")[:, y0 : y0 + h_dim, x0 : x0 + w_dim]

    factor = max(r_raw.shape[0] // h_raw.shape[0], 1)
    r_step_mm = r_raw[: horizon * factor].reshape(horizon, factor, h_dim, w_dim).sum(axis=1)
    rainfall_mm_per_hr = (r_step_mm * (3600.0 / dt_seconds)).astype(np.float32)
    h0_m = np.where(valid_mask, np.maximum(h_raw[0], 0.0), 0.0).astype(np.float32)

    t0 = time.perf_counter()
    result = predictor.predict(
        dem=dem,
        rainfall=rainfall_mm_per_hr,
        initial_water_depth=h0_m,
        forecast_steps=horizon,
    )
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    pred_m = np.where(valid_mask[None, :, :], result["water_depth"], 0.0).astype(np.float32)
    save_standard_prediction(
        output_dir=OUT_DIR,
        event_id=event_id,
        predicted_water_depth_m=pred_m,
        valid_mask=valid_mask,
    )

mean_event_ms = float(np.mean(latencies_ms))
peak_gpu_mb = (
    float(torch.cuda.max_memory_allocated() / (1024**2))
    if device.startswith("cuda")
    else 0.0
)

TRAIN_TIME_S = None  # Replace with measured training wall-clock time (seconds)

write_prediction_manifest(
    output_dir=OUT_DIR,
    model_name="PINO",
    manifest=manifest,
    efficiency={
        "training_time_s": TRAIN_TIME_S,
        "inference_time_per_event_ms": mean_event_ms,
        "inference_time_per_step_ms": mean_event_ms / horizon,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_size_mb": float(CHECKPOINT_PATH.stat().st_size / 1e6),
        "checkpoint_location": "GoogleDrive://Crestix-Probe/PINO/checkpoints/best_pino_model.pt",
        "predictions_location": "GoogleDrive://Crestix-Probe/PINO/predictions_v1.1.0/",
        "hardware": {"device": device},
    },
)
```

---

## 4. Storage & Running the Unified Evaluator

- **Large Artifacts (Google Drive, excluded from GitHub):**
  - PINO checkpoints: `GoogleDrive://Crestix-Probe/PINO/checkpoints/best_pino_model.pt`
  - PINO held-out predictions (`prediction_manifest.json` + 10 `<event_id>.npz`): `GoogleDrive://Crestix-Probe/PINO/predictions_v1.1.0/`
  - ConvLSTM held-out predictions: `GoogleDrive://Crestix-Probe/ConvLSTM/predictions_v1.1.0/`
- **Run Joint Comparison (from `ConvLSTM-baased-U-Net-for-Crestix-Probe-`):**

```bash
python common_evaluator.py \
  --manifest shared_manifest.json \
  --data-root /path/to/ukea_8m_5min \
  --convlstm-preds /path/to/ConvLSTM_predictions_v1.1.0 \
  --pino-preds /path/to/PINO_predictions_v1.1.0 \
  --output-dir outputs/comparison
```

Outputs written to `outputs/comparison/` in the ConvLSTM repository:
- `model_comparison.csv` & `model_comparison.json`
- `plots/*.svg` & `plots/*.png` (continuous error, flood classification, ROC, Precision-Recall, horizon error, and efficiency)
