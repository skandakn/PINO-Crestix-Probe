# PINO-Crestix-Probe — Physics-Informed Neural Operator for 2D Flood Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

> **Physics-Informed Neural Operator (FNO3d + 2D Shallow-Water Residual Heads)** for **Crestix Probe**  
> Configured for **LarNO Urban Flood Benchmark (`ukea_8m_5min` 8.0 m, 32×32 crop)** and integrated with the shared 1000-event **LarNO** data pipeline and common evaluation framework.  
> Sibling Deep-Learning model & comparison host: [`skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-)

---

## 1. Governing Physics & Latent Velocity Heads

The model (`9,448,260` PyTorch parameter elements / `18,885,444` real FP32 scalars, `~75.54 MB` `state_dict`) predicts space-time water depth $h(t, x, y) \ge 0\ \text{m}$ alongside auxiliary unsupervised velocity fields $(u, v)\ [\text{m/s}]$ over terrain $z_b(x, y)\ [\text{m}]$ to evaluate the 2D Shallow-Water (Saint-Venant) system:

- **Mass Conservation:** $\partial_t h + \partial_x(hu) + \partial_y(hv) = R - I$ ($\text{m/s}$)
- **Momentum Conservation ($x, y$):** Conservative shallow-water flux + topographic bed slope $\nabla z_b$ + Manning friction slope $S_{f,x}, S_{f,y}$ with wet-dry threshold $h_\varepsilon = 10^{-3}\ \text{m}$.
- **Physical Assumptions:** Hydrostatic pressure, incompressible constant-density flow, depth-averaged uniform horizontal velocity $(u, v)$, mild bed slope ($\cos\theta \approx 1$), and $I = 0$ unless supplied.

> **Important Note on Labels (`u, v`):** LarNO UKEA core datasets provide terrain (DEM), rainfall, and water depth (`h`), **not** ground-truth horizontal velocities $(u, v)$. Consequently, $(u, v)$ are **unsupervised latent variables** constrained only by PDE/boundary losses; no supervised velocity accuracy is claimed.

---

## 2. Unified Input / Output Contract (`ModelContract`)

Both `PINO-Crestix-Probe` and `ConvLSTM-baased-U-Net-for-Crestix-Probe-` implement the same contract (`ModelContract`, `LoadedModel`, `load_model`, `predict`):

| Tensor / Argument | Shape | SI Units | Constraints & Conversion |
|---|---|---|---|
| `dem` | `(H, W)` or `(B, H, W)` | `m` | Bare-earth elevation; `NaN` / `-9999.0` masked to `valid_mask` when `allow_dem_nan=True` |
| `rainfall` | scalar, `(T,)`, `(B, T)`, `(H, W)`, `(T, H, W)`, or `(B, T, H, W)` | `mm/h` | Finite, `>= 0.0` (converted internally to SI `m/s` via `/ 3,600,000`) |
| `initial_water_depth` | `(H, W)` or `(B, H, W)` | `m` | Finite, `>= 0.0`, spatial shape matching `dem` |
| `forecast_steps` | `int` | steps | `>= 1` (UKEA native cadence `dt_seconds=300.0`, `cell_size_m=8.0`) |
| **Output `water_depth`** | `(T, H, W)` or `(B, T, H, W)` | `m` | `float32`, masked to `0.0` on invalid/building DEM cells (`clamp_negative` optional) |

### Internal PINO Tensor Shapes & Units
- **Model Input `x`:** `(B, 4, T, H, W)` with channels `[dem_m (m), rainfall_m_s (m/s), initial_depth_m (m), manning_n (s*m^-1/3)]` (+ 3 normalized `(t, y, x)` coordinate channels when `append_grid_coords=True`).
- **Model Output (`PINOOutput`):** `h` `(B, T, H, W)` [$\text{m}$], latent `u, v` `(B, T, H, W)` [$\text{m/s}$].

---

## 3. LarNO Dataset Acquisition & Expected Shapes / Units

- **Upstream Benchmark Source:** [`holmescao/LarNO-dataset`](https://huggingface.co/datasets/holmescao/LarNO-dataset) (`geodata/*` + `flood/*` + `benchmark.zip`, 2.28 GB).
- **Shared Google Drive Mirror:** `GoogleDrive://Crestix-Probe/Datasets/LarNO-UKEA-1000Events/` (~20 GB uncompressed across 1000 events).

| Catchment | Region Tag | Events | Spatial Res. | Native Grid `(H, W)` | Target Crop `(H, W)` | Depth Array `h.npy` | Rainfall Array `rainfall.npy` | Cadence |
|---|---|---|---|---|---|---|---|---|
| **UKEA Native** | `ukea_8m_5min` | 20 | `8.0 m` | `(50, 120)` | — | `(36, 50, 120)` `float64` (`m`) | `(360, 50, 120)` `float64` (`mm/30s`) | 5 min depth / 30 s rain (10:1 sum) |
| **UKEA PINO Crop** | `ukea_8m_5min` | 20 | `8.0 m` | `(50, 120)` | `(32, 32)` `[9:41, 44:76]` | `(36, 32, 32)` (`m`) | `(36, 32, 32)` summed `mm/5min` -> `m/s` (`* 3.333e-6`) | 5 min (`dt=300 s`, `dx=8 m`) |
| **Futian (Shared Manifest)** | `region1_20m` | 80 | `20.0 m` | `(400, 560)` | — | `(72, 400, 560)` `float64` (`m`) | `(72, 400, 560)` `float64` (`mm/5min`) | 5 min (1:1 alignment, 6 h) |

```bash
# 1. Dry-run download check via shared manifest
python scripts/download_larno_dataset.py --manifest data/event_manifest.csv --dry-run

# 2. Download UKEA subset for PINO training/validation
python scripts/download_larno_dataset.py --manifest data/event_manifest.csv --catchment UKEA --split val --output-dir data/raw_larno
# Or download directly via training flag: python scripts/train_pino.py --config configs/ukea_32x32.yaml --download-data

# 3. Inspect manifest integrity & array schemas
python scripts/inspect_larno_dataset.py --manifest data/event_manifest.csv --split-manifest data/split_manifest.json
```

---

## 4. Shared Event Split Manifests & Versions

- **Official 1000-Event Stratified Manifest:** [`data/split_manifest.json`](data/split_manifest.json) — **version `1.0.0`**
  - **Train (`80` events):** `64` Futian + `16` UKEA (`Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`).
  - **Val (`10` events):** `8` Futian + `2` UKEA (`r100y_p0.5_d3h_1`, `r200y_p0.7_d3h_1`).
  - **Test (`10` events):** `8` Futian + `2` UKEA (`r100y_p0.6_d3h_1`, `r500y_p0.9_d3h_1`).
  - Normalization moments in [`data/dataset_statistics.json`](data/dataset_statistics.json) are computed strictly on the 80 training events ([`reports/data_quality_report.md`](reports/data_quality_report.md)).
- **Cross-Model UKEA `32x32` Benchmark Manifest:** [`shared_manifest.json`](shared_manifest.json) — **version `1.1.0`** (`sha256: da79e0b11e6d4e85006c9d8a9da8aaf69664a0f251f0a7001c7db62e210ae0dd`)
  - Synchronized byte-for-byte with [`ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/shared_manifest.json): `ukea_8m_5min`, crop `[9:41, 44:76]` (`crop_y=9, crop_x=44`, `32x32`, `cell_size_m=8.0`), `forecast_horizon_steps=36` (`dt_seconds=300.0`), `flood_threshold_m=0.03`, `valid_cell_rule="non_building_and_finite_dem"`.
  - **Split:** **6 `train_events`** / **2 `val_events`** / **10 `test_events`** (exact IDs in `shared_manifest.json`).
  - **Leakage Safeguard:** `8/10` of `shared_manifest.json` `v1.1.0` UKEA test events sit inside `data/split_manifest.json` `v1.0.0` train/val, and `configs/ukea_32x32.yaml` defaults to `UKEAPinoDataset`'s upstream 8 train/val (`6/2`) + 12 test split. Do **not** evaluate `v1.0.0`-trained weights against `shared_manifest.json` `v1.1.0`; train and evaluate strictly on the exact `shared_manifest.json` `v1.1.0` IDs when exporting for cross-model comparison.

---

## 5. Environment Setup & Dependency Installation

### Option A: `pip` + Virtual Environment
```bash
git clone https://github.com/skandakn/PINO-Crestix-Probe.git
cd PINO-Crestix-Probe

python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

### Option B: `conda` (`environment.yml`)
```bash
conda env create -f environment.yml
conda activate crestix-pino
```

---

## 6. Exact Commands: Preprocessing, Training, Validation, Testing & Inference

### A. Preprocessing & Physics Diagnostics
```bash
# Audit 1000-event manifests and split disjointness
python scripts/inspect_larno_dataset.py --manifest data/event_manifest.csv --split-manifest data/split_manifest.json

# Run physics validation & dimensional diagnostics suite
python scripts/validate_pino_physics.py
```

### B. Training
```bash
# CPU Smoke Test (2 epochs; auto-generates deterministic synthetic UKEA data if raw data is absent)
python scripts/train_pino.py --config configs/ukea_32x32.yaml --smoke-test --output-dir artifacts/smoke_test

# Full UKEA 32x32 Training (auto-downloads raw UKEA data if --download-data is passed)
python scripts/train_pino.py --config configs/ukea_32x32.yaml --download-data --output-dir artifacts/ukea_32x32
```

### C. Validation & Testing
```bash
# Validate saved checkpoint on val or held-out test split (writes optional JSON summary)
python scripts/validate_pino.py --checkpoint artifacts/smoke_test/checkpoints/best.pt --split val --output-json outputs/smoke_validation.json
python scripts/validate_pino.py --checkpoint artifacts/ukea_32x32/checkpoints/best.pt --split test --output-json outputs/test_validation.json

# Evaluate against held-out LarNO .npz events
python scripts/evaluate_pino.py \
  --checkpoint artifacts/ukea_32x32/checkpoints/best.pt \
  --dataset path/to/held_out_larno_events/ \
  --output-dir outputs/evaluation
```

### D. Sample Inference & Checkpoint Reproduction Verification
```bash
# Run standardized CLI inference and verify bitwise identical reproduction across two checkpoint loads
python scripts/predict_pino.py \
  --checkpoint artifacts/smoke_test/checkpoints/best.pt \
  --steps 12 --verify-reproduction --output-json outputs/sample_inference.json
```

### E. Standardized Python Inference API
```python
import numpy as np
from crestix_pino import load_model, predict

dem = np.full((32, 32), 25.0, dtype=np.float32)       # m (NaN building cells masked to 0.0 depth)
dem[:2, :2] = np.nan
rainfall = np.full((12,), 42.0, dtype=np.float32)     # mm/h
h0 = np.zeros((32, 32), dtype=np.float32)             # m

res = predict(
    dem=dem,
    rainfall=rainfall,
    initial_water_depth=h0,
    forecast_steps=12,
    model="artifacts/smoke_test/checkpoints/best.pt",
    allow_dem_nan=True,
    clamp_negative=True,
)
print(res["water_depth"].shape)       # (12, 32, 32)
print(res["metadata"]["units"])       # {'dem': 'm', 'rainfall': 'mm/h', 'water_depth': 'm'}
```

### F. Latency Benchmark & Full Test Suite
```bash
# Measure model-only vs full-pipeline latency
python scripts/benchmark_inference.py \
  --checkpoint artifacts/smoke_test/checkpoints/best.pt \
  --batch-size 1 --height 32 --width 32 --steps 12 --warmup 1 --runs 3 \
  --output-json outputs/benchmark.json

# Run full unit, physics & inference test suite (45 tests)
python -m pytest -v
```

---

## 7. Shared Checkpoints & Common Evaluation Protocol

- **Common Evaluation Protocol:** [`docs/COMMON_EVALUATION_PROTOCOL.md`](docs/COMMON_EVALUATION_PROTOCOL.md) (synchronized identically in [`ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/docs/COMMON_EVALUATION_PROTOCOL.md)).
- **Outputs Policy & Model Card:** [`outputs/README.md`](outputs/README.md) & [`reports/README.md`](reports/README.md).
  - Best UKEA checkpoint (`train_pino.py`): `GoogleDrive://Crestix-Probe/PINO/checkpoints/best.pt` -> local `artifacts/ukea_32x32/checkpoints/best.pt`
  - Common-Eval checkpoint alias (`docs/COMMON_EVALUATION_FRAMEWORK.md`): `GoogleDrive://Crestix-Probe/PINO/checkpoints/best_pino_model.pt` -> local `artifacts/ukea_32x32/best_pino_model.pt`
  - Resume checkpoint: `GoogleDrive://Crestix-Probe/PINO/checkpoints/last.pt` -> local `artifacts/ukea_32x32/checkpoints/last.pt`
  - PINO held-out prediction archive (`v1.1.0`): `GoogleDrive://Crestix-Probe/PINO/predictions_v1.1.0/`
  - Raw dataset: `GoogleDrive://Crestix-Probe/Datasets/LarNO-UKEA-1000Events/`
  - Check local availability: `python scripts/download_checkpoints.py --check`

---

## 8. Verified Reproduction & Smoke Check Results

All verification checks below are runnable on CPU in `.venv` (`torch 2.15.0+cpu`, Python `3.12.10`):
- **Full Pytest Suite (`python -m pytest -v`):** **45/45 tests pass**, covering FNO3d gradients, PINO forward/masking, physics residuals, diagnostics `S1`–`S7`, dataset/manifest disjointness, training/resume, and checkpoint load+reproduction (`test_checkpoint_save_load_and_reproduce`).
- **CPU Smoke Training (`scripts/train_pino.py --smoke-test --output-dir artifacts/smoke_test`):** Saves `artifacts/smoke_test/checkpoints/best.pt` (`best_val_loss=0.002935`, `val_rmse=0.03579 m` across 2 epochs of deterministic synthetic UKEA windows).
- **Checkpoint Validation (`scripts/validate_pino.py --checkpoint artifacts/smoke_test/checkpoints/best.pt --split val`):** Loads `artifacts/smoke_test/checkpoints/best.pt` (`framework=pytorch_pino_checkpoint`, `num_windows=14`), obtaining `mae_m=0.031895 m`, `rmse_m=0.038645 m`, `nonnegative_violations.violation_count=0` across `172,032` cells.
- **Sample Inference & Reproduction (`scripts/predict_pino.py --verify-reproduction`):** Loads `artifacts/smoke_test/checkpoints/best.pt` independently twice on a `32x32` DEM with `NaN` building cells -> `output_water_depth_shape=[12, 32, 32]`, `nan_cell_zeroed=true`, `raw_negative_cell_count=0`, `reproduction_verified=true`, `reproduction_max_abs_diff_m=0.0`.
- **Latency Benchmark (`scripts/benchmark_inference.py`):** Measures CPU/GPU `model_only` and `full_pipeline` latency (`ms/sample`) and throughput (`samples/s`) cleanly on loaded `.pt` checkpoints.

---

## 9. Limitations & Physics Diagnostic Findings

Extensive physics verification documented in [`docs/PINO_PHYSICS_DIAGNOSTICS_REPORT.md`](docs/PINO_PHYSICS_DIAGNOSTICS_REPORT.md), [`docs/PINO_DIMENSIONAL_ANALYSIS.md`](docs/PINO_DIMENSIONAL_ANALYSIS.md), and [`docs/PINO_AUDIT_REPORT.md`](docs/PINO_AUDIT_REPORT.md) establishes the following critical limitations:

1. **Domain & Scale Differences:** The default FNO config operates on a `32x32` crop of `ukea_8m_5min` (`dx=8.0 m`, `dt=300.0 s`). Default fallbacks in `ModelContract`/`metrics.py` (`dt=30 s, dx=10 m`) and `ShallowWaterConfig` (`dx=2 m, dt=60 s`) differ sharply from UKEA and Futian (`20.0 m`); callers must always propagate explicit catchment `(dx, dt)` metadata.
2. **Limited Event Count:** `holmescao/LarNO-dataset` contains only 20 UKEA events (`8/12` in upstream `UKEAPinoDataset`; `16/2/2` in `split_manifest.json` v1.0.0; `6/2/10` in `shared_manifest.json` v1.1.0) and 80 Futian events, limiting statistical power for rare return-period storms (`r300y`, `r500y`).
3. **Simulated & Unsupervised Labels:** Ground-truth depths `h` come from synthetic/simulated MIKE+ 2D solver runs rather than field gauges, while velocity fields $(u, v)$ have **no labels at all**. Unsupervised $(u, v)$ gauge freedom allows `BoundaryConfig.south="open"` to satisfy conservation residuals analytically even with inaccurate depth volumes (`S7`).
4. **Open Sub-Catchment Boundary Outflow vs Closed Metrics:** On the open `32x32` UKEA crop (`[9:41, 44:76]`), **`75.53%`–`78.28%`** of storm rainfall exits laterally across sub-catchment edges. Therefore, closed-basin `compute_mass_conservation_error` flags even ground-truth LarNO data with `87.6%–88.6%` error (`BoundaryConfig` defaults `north/east/west="closed"` also need open/calibrated outflow treatment on sub-domain crops).
5. **Physics Loss Defects & Assumptions (`S4`–`S6`):**
   - `softplus(beta=5)` paired with `MSE(ReLU(-h_raw))` traps dry streets at a positive bias (`~0.041–0.115 m` in `S5`) when negative `h_raw` is penalized directly.
   - `NaN` DEM building fills (`+50 m`) must be paired with `valid_mask` masking (`allow_dem_nan=True`) so central differences do not differentiate across artificial vertical cliffs (`S6`).
   - Adding unscaled residuals across incompatible SI units (`m^2`, `m^2/s^2`, `m^4/s^4`, mixed BCs) suppresses continuity gradients relative to momentum and data losses without nondimensionalization (`docs/PINO_DIMENSIONAL_ANALYSIS.md`).

---

## 10. Pre-Demonstration Verification Checklist

- [x] New contributor can install dependencies via `requirements.txt` or `environment.yml`
- [x] Dataset manifests (`data/split_manifest.json` v1.0.0, `shared_manifest.json` v1.1.0) and downloader dry-run work cleanly
- [x] Synthetic & physics tests pass (`45/45` via `pytest -v`)
- [x] Saved checkpoint (`artifacts/smoke_test/checkpoints/best.pt`) loads independently via `load_model` / `predict`
- [x] Sample inference reproduces bitwise identically across independent checkpoint loads (`max_abs_diff = 0.0 m`)
- [x] All evaluation metrics (`MAE [m]`, `RMSE [m]`, `R^2`, `Rel. L2`, diffusive-wave residual `m/s`) and SI units are documented
- [x] Held-out test events remain isolated from training splits and normalization statistics (`Train ∩ Val ∩ Test = ∅`)

---

## 11. Citation & Dataset Attribution

- **Dataset:** [`holmescao/LarNO-dataset`](https://huggingface.co/datasets/holmescao/LarNO-dataset)
- **Reference Paper:** *Large-scale urban flood modeling and zero-shot high-resolution generalization with LarNO*, *Journal of Hydrology*, 2026.

```bibtex
@misc{crestix_probe_pino_2026,
  title  = {Crestix Probe: Physics-Informed Neural Operator for 2D Urban Flood Forecasting},
  author = {Crestix Probe Modeling Team},
  year   = {2026},
  url    = {https://github.com/skandakn/PINO-Crestix-Probe}
}
```

---

## 12. Common Evaluation Framework (ConvLSTM U-Net vs. PINO)

Cross-model benchmarking against **ConvLSTM-based U-Net** is governed by a single shared protocol hosted in [`skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-):

- **Integration Guide & Export Instructions:** [`docs/COMMON_EVALUATION_FRAMEWORK.md`](docs/COMMON_EVALUATION_FRAMEWORK.md)
- **Synchronized Evaluation Protocol:** [`docs/COMMON_EVALUATION_PROTOCOL.md`](docs/COMMON_EVALUATION_PROTOCOL.md)
- **Shared Event Manifest (`v1.1.0`):** [`shared_manifest.json`](shared_manifest.json) (synchronized with [`ConvLSTM/shared_manifest.json`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/shared_manifest.json) — `holmescao/LarNO-dataset`, `ukea_8m_5min`, crop `(9, 44, 32x32)`, 6 train / 2 val / 10 test events, 36-step horizon, `0.03 m` flood threshold, `non_building_and_finite_dem` masking)
- **Canonical Evaluator & Prediction Format:** [`common_evaluator.py`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/common_evaluator.py) | [`PREDICTION_FORMAT.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/PREDICTION_FORMAT.md)
- **Comparison Report & Fairness Checklist:** [`reports/MODEL_COMPARISON_REPORT.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/reports/MODEL_COMPARISON_REPORT.md) | [`reports/LIMITATIONS_AND_FAIRNESS_CHECKLIST.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/reports/LIMITATIONS_AND_FAIRNESS_CHECKLIST.md)
- **Google Drive Prediction Archive:** `GoogleDrive://Crestix-Probe/PINO/predictions_v1.1.0/`
