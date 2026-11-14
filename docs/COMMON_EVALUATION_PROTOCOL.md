# Common Cross-Model Evaluation Protocol — Crestix Probe

This protocol governs fair, leakage-free evaluation across both deep-learning flood forecasting models in **Crestix Probe**:
- **ConvLSTM-based U-Net:** [`skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-)
- **PINO (Physics-Informed Neural Operator):** [`skandakn/PINO-Crestix-Probe`](https://github.com/skandakn/PINO-Crestix-Probe)

---

## 1. Unified Input / Output Contract (`ModelContract`)

Both repositories expose identical Python contract types (`ModelContract`, `LoadedModel`, `load_model`, `predict`):

| Argument / Field | Shape | SI Units | Constraints |
|---|---|---|---|
| `dem` | `(H, W)` or `(B, H, W)` | `m` | Elevation above datum; `NaN` / `-9999.0` permitted when `allow_dem_nan=True` |
| `rainfall` | scalar, `(T,)`, `(B, T)`, `(H, W)`, `(T, H, W)`, or `(B, T, H, W)` | `mm/h` | Finite, `>= 0.0` |
| `initial_water_depth` | `(H, W)` or `(B, H, W)` | `m` | Finite, `>= 0.0`, matches `dem` spatial shape |
| `forecast_steps` | `int` | steps | `>= 1` (standard 5-min step `dt_seconds = 300.0`) |
| **Output `water_depth`** | `(T, H, W)` or `(B, T, H, W)` | `m` | Non-negative `float32`, masked to `0.0` on invalid/building DEM cells |

---

## 2. Dataset, Manifests & Test Isolation

- **Source Dataset:** [`holmescao/LarNO-dataset`](https://huggingface.co/datasets/holmescao/LarNO-dataset) (`ukea_8m_5min` at `8.0 m` and `region1_20m` at `20.0 m`).
- **Event-Level Stratified Manifest (`data/split_manifest.json`, version `1.0.0`):**
  - **100 Total Events:** 80 Futian (`region1_20m`, `400x560`, 72 steps) + 20 UKEA (`ukea_8m_5min`, `50x120`, 36 depth / 360 rain steps).
  - **Split Counts:** **80 Train** (64 Futian + 16 UKEA) / **10 Val** (8 Futian + 2 UKEA) / **10 Test** (8 Futian + 2 UKEA).
  - **Normalization Moments (`data/dataset_statistics.json`):** Fitted strictly on the **80 training events** only.
- **Cross-Model UKEA `32x32` Benchmark Manifest (`shared_manifest.json`, version `1.1.0`):**
  - **Integrity Digest:** `sha256: da79e0b11e6d4e85006c9d8a9da8aaf69664a0f251f0a7001c7db62e210ae0dd` (synchronized byte-for-byte across both repositories).
  - **Region Tag & Horizon:** `ukea_8m_5min`, `dt_seconds = 300.0`, `cell_size_m = 8.0`, `forecast_horizon_steps = 36` (`3 hours` rollout).
  - **Crop Window:** `crop_y = 9, crop_x = 44, height = 32, width = 32` (`[9:41, 44:76]`). *(Note: `configs/larno_data_pipeline.yaml` originally drafted `[12:44, 32:64]`, while `shared_manifest.json`, `configs/ukea_32x32.yaml`, and both model loaders standardize on `[9:41, 44:76]`.)*
  - **Split Partition:** **6 `train_events`** / **2 `val_events`** / **10 `test_events`** (exact event IDs listed in `shared_manifest.json`).
  - **Leakage Guard (`validate_fairness_contract`):** `8/10` of `shared_manifest.json` `v1.1.0`'s UKEA test events appear in `data/split_manifest.json` `v1.0.0`'s train/val splits. Therefore, cross-model comparisons using `common_evaluator.py` enforce training/evaluation strictly on identical `shared_manifest.json` `v1.1.0` splits and refuse mixed-manifest comparisons (`FairnessViolationError`).

---

## 3. Canonical Evaluator, Prediction Format & Comparison Suite

The cross-repository evaluation framework is hosted in [`ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-) and documented for PINO export in [`PINO-Crestix-Probe/docs/COMMON_EVALUATION_FRAMEWORK.md`](https://github.com/skandakn/PINO-Crestix-Probe/blob/main/docs/COMMON_EVALUATION_FRAMEWORK.md):
- **Evaluator & Schema:** `common_evaluator.py`, `PREDICTION_FORMAT.md`, `tests/test_common_evaluator.py`
- **Comparison Reports & Plots:** `reports/MODEL_COMPARISON_REPORT.md`, `reports/LIMITATIONS_AND_FAIRNESS_CHECKLIST.md`, `outputs/comparison/model_comparison.{csv,json}`, `outputs/comparison/plots/*`
- **Valid Cell Rule (`non_building_and_finite_dem`):** Exclude `NaN`/`Inf` cells and sentinel nodata cells (`np.isfinite(dem) & (dem > -999.0)` in `common_evaluator.py`, plus `-9999.0`/`<= -9000.0` sentinels and catchment wall thresholds `>= 40.0 m` in UKEA / `>= 50.0 m` in Futian where configured).
- **Binary Flood Hazard Threshold:** `flood_threshold_m = 0.03 m` (`3 cm`), corresponding to urban nuisance/street-flooding onset.

---

## 4. Shared Evaluation Metrics & Units

| Metric | Definition | Units |
|---|---|---|
| **MAE** | Mean absolute depth error over valid cells | `m` |
| **RMSE** | Root mean squared depth error over valid cells | `m` |
| **R^2** | Coefficient of determination over valid cells | unitless |
| **AUROC / PR-AUC** | ROC and Precision-Recall area using continuous predicted depth scores (`undefined` if single-class) | unitless (`[0, 1]`) |
| **Flood Precision / Recall / F1 / CSI / CM** | Binary inundation detection (`h >= 0.03 m`) and confusion matrix `[[TN, FP], [FN, TP]]` | unitless / count |
| **Rel. L2 / Wet-Cell MRE** | Normalized L2 depth error and relative error over cells with `target >= 0.03 m` | unitless |
| **Nonnegative Violations** | Count and fraction of predicted cells with `h < 0.0 m` before clamping | count / fraction |
| **Diffusive-Wave Residual** | Interior 2D continuity residual computed at true `(dt=300 s, dx=8 m)` without fabricating `(u, v)` | `m/s` |
| **Inference Latency & Peak Memory** | Wall-clock latency (`ms/event`, `ms/step`) and peak GPU/CPU memory (`MB`) | `ms`, `MB` |

> **Important Cross-Model Comparison Caveat on Mass Balance:**  
> On the open UKEA `32x32` sub-catchment crop (`[9:41, 44:76]`), **75.5%–78.3%** of cumulative rainfall leaves the sub-domain laterally across open boundaries. Do **not** apply closed-basin global volume conservation error (`V(t) - V(0) == cumsum(Rain)`) in isolation without noting boundary outflow; compare models primarily using masked depth error (`MAE`, `RMSE`, `R^2`), flood detection (`AUROC`, `PR-AUC`, `F1`, `CSI` at `0.03 m`), horizon error growth, and inference latency.

---

## 5. Checkpoint Reproduction Verification Rule

A release checkpoint is considered verified only when:
1. Saved checkpoint weights load from disk without relying on hardcoded source-only factories.
2. Two independent `load_model(checkpoint_path)` calls produce bitwise identical (`max_abs_diff == 0.0 m`) predictions on the same deterministic input tensor.
3. Outputs satisfy shape, `float32` dtype, non-negativity (`h >= 0.0 m`), and building-mask zeroing (`h == 0.0 m` on `NaN`/invalid DEM cells).
