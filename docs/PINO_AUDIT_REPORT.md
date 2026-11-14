# PINO Repository Audit Report — Crestix Probe

**Repository:** `https://github.com/skandakn/PINO-Crestix-Probe`  
**Audited Baseline:** `main` (`6f10c54`)  
**Reference Dataset:** `holmescao/LarNO-dataset` (`ukea_8m_5min`)  

---

## 1. Repository Structure Inspection

At baseline (`6f10c54`), the repository contained four main subsystems:
1. **Model & Spectral Operator (`src/crestix_pino/models/`):**
   - `fno.py`: `SpectralConv3d`, `FNOBlock3d`, `FNO3d` (3D Fourier Neural Operator over `(B, C_in, T, H, W)` with optional `(t, y, x)` coordinate concatenation and asymmetric zero-padding).
   - `pino.py`: `PINOFloodModel`, `PINOOutput`, `build_input_tensor(...)`.
2. **Physics & Losses (`src/crestix_pino/physics/`, `src/crestix_pino/losses/`):**
   - `shallow_water.py`, `residuals.py`, `conservation.py`, `boundary_conditions.py`.
   - `data_loss.py` (`DataLoss`), `physics_loss.py` (`PhysicsLoss`, `PINOCompositeLoss`).
3. **Data & Training (`src/crestix_pino/data/`, `src/crestix_pino/training/`, `configs/`, `scripts/train_pino.py`):**
   - `UKEAPinoDataset`, `preprocess_dem`, `align_rainfall_to_depth`, `build_pino_sample`.
   - `configs/ukea_32x32.yaml`, `configs/default_pino.yaml`, `PINOTrainer`.
4. **Evaluation & Inference (`src/crestix_pino/evaluation/`, `src/crestix_pino/inference/`):**
   - `metrics.py`, `evaluate.py`, `visualization.py`, `model_loader.py`, `predictor.py`.

---

## 2. Comparison Against Shared LarNO UKEA Data Contract

| Field / Parameter | LarNO UKEA (`ukea_8m_5min`) Contract | Baseline `PINO-Crestix-Probe` | Status / Action |
|---|---|---|---|
| **DEM (`dem.npy`)** | `(H, W)` in `m`; buildings/nodata encoded as `NaN` | `preprocess_dem` produces `dem_clean`, `building_mask`, `valid_mask` | `valid_mask` was unused in `PINOFloodModel` and `PhysicsLoss` — **fixed** |
| **Rainfall (`rainfall.npy`)** | `(T_r, H, W)` in `mm / 5min` (30s substeps, 360 steps for 3h) | `align_rainfall_to_depth` sums substeps and scales to `m/s` | Consistent |
| **Water Depth (`h.npy`)** | `(T, H, W)` in `m` (`T=36`, 5-min cadence) | `h` target `(B, T, H, W)` in `m` | Consistent |
| **Grid Resolution** | `dx = 8.0 m`, `dy = 8.0 m`, `dt = 300.0 s` | `ukea_32x32.yaml` used `8.0 / 300.0`; `default_pino.yaml` used `2.0 / 60.0` | `default_pino.yaml` **updated** to `8.0 / 300.0` + `GridMetadata` |
| **Velocity `(u, v)`** | **Not provided** in LarNO UKEA | `PINOFloodModel` predicts unsupervised latent `(u, v)` | Explicitly documented as unsupervised latent variables |

---

## 3. What Worked vs. What Was Incomplete

### Working Components Preserved
- `FNO3d` 3D spectral convolution backbone with `(t, y, x)` coordinate grid injection and zero-padding.
- `UKEAPinoDataset` memory-mapped event-disjoint `train / val / test` loader (`8 / 12` official UKEA split).
- `DataLoss` with `valid_mask` support (`mse`, `l1`, `rel_l2`).
- `evaluation/` and `inference/` packages (`PINOPredictor`, `evaluate_dataset`, `plot_spatial_error_map`, etc.).

### Gaps & Issues Identified
1. **Invalid/Building DEM Masking Gap:** `PINOFloodModel.forward` did not accept `valid_mask`, so `softplus(h_raw)` produced strictly positive water depths (`softplus(0) ≈ 0.1386 m`) on building/NaN cells. `PhysicsLoss` also evaluated SWE spatial derivatives across artificial `wall_height` steps at building boundaries.
2. **Missing Native `GridMetadata` & Configurable Horizon Interface:** `PINOFloodModel` lacked a structured `GridMetadata` / `PINOModelConfig` and a `forecast(dem_m, rainfall_m_s, initial_depth_m, forecast_steps)` entrypoint supporting `T_in` context frames and `T_out` forecast steps.
3. **Inference Adapter Gap:** `load_model` and `PINOPredictor` could not directly wrap a PyTorch `PINOFloodModel` instance or handle `NaN` DEM building cells when explicitly enabled.
4. **Config Mismatch:** `configs/default_pino.yaml` specified `dx=2.0, dy=2.0, dt=60.0` instead of native UKEA `dx=8.0, dy=8.0, dt=300.0`.
5. **Windows MKL FFT Test Failure:** In `.venv` (PyTorch CPU on Windows with oneMKL), `tests/test_fno.py::test_spectral_conv3d_shape_and_gradients` raises `RuntimeError: MKL FFT error: Intel oneMKL DFTI ERROR: Inconsistent configuration parameters` during `irfftn` backward on `(2, 8, 6, 12, 14)`. `FNO3d` with padding and `32×32` UKEA grids passes without error.
