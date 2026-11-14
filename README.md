# PINO-Crestix-Probe — Physics-Informed Fourier Neural Operator for 2D Flood Prediction

Core Physics-Informed Fourier Neural Operator (PINO) architecture, 2D Shallow-Water Equations (SWE) physics engine, LarNO UKEA 32×32 dataset/training pipeline, and evaluation/benchmarking/inference package for predicting spatial-temporal water-depth distributions.

---

## 1. Governing Equations (2D Shallow Water Equations)

The model enforces the depth-averaged 2D Saint-Venant (Shallow Water) system in conservative form over bed topography $z_b(x, y)$:

### Mass Conservation (Continuity)
$$\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = R(x, y, t) - I(x, y, t)$$

### $x$-Momentum Conservation
$$\frac{\partial (hu)}{\partial t} + \frac{\partial}{\partial x}\left(hu^2 + \frac{1}{2}gh^2\right) + \frac{\partial (huv)}{\partial y} = -g h \left(\frac{\partial z_b}{\partial x} + S_{f,x}\right)$$

### $y$-Momentum Conservation
$$\frac{\partial (hv)}{\partial t} + \frac{\partial (huv)}{\partial x} + \frac{\partial}{\partial y}\left(hv^2 + \frac{1}{2}gh^2\right) = -g h \left(\frac{\partial z_b}{\partial y} + S_{f,y}\right)$$

where Manning's bed friction slopes are:
$$S_{f,x} = \frac{n^2 u \sqrt{u^2 + v^2}}{\max(h, h_{\varepsilon})^{4/3}}, \qquad S_{f,y} = \frac{n^2 v \sqrt{u^2 + v^2}}{\max(h, h_{\varepsilon})^{4/3}}$$
with wetting-drying threshold $h_{\varepsilon} = 10^{-3}\ \text{m}$ and wet-cell masking ($h > h_{\varepsilon}$) to prevent division-by-zero singularities on dry ground.

### Physical Assumptions
1. **Hydrostatic pressure distribution**: Vertical fluid acceleration is negligible compared to gravity ($L_{\text{horizontal}} \gg h$).
2. **Incompressible fluid**: Constant water density $\rho$.
3. **Depth-averaged horizontal velocity**: Uniform vertical profile $(u, v)$ at each $(x, y, t)$.
4. **Mild bed slope**: $\cos\theta \approx 1$.
5. **Infiltration**: $I(x, y, t) = 0$ unless an explicit infiltration field is supplied.

---

## 2. Physical Units & Conversions

All model inputs, outputs, and PDE finite differences operate strictly in **SI units**:

| Quantity | Symbol | Unit | Conversion Rule |
|---|---|---|---|
| Water depth | $h, h_0$ | $\text{m}$ | `mm_to_meters`: $h_{\text{m}} = h_{\text{mm}} \times 10^{-3}$ |
| Bed elevation (DEM) | $z_b$ | $\text{m}$ | Native meters |
| Grid spacing | $\Delta x, \Delta y$ | $\text{m}$ | Configured from LarNO UKEA resolution |
| Dataset time interval | $\Delta t$ | $\text{s}$ | Configured from dataset frame step |
| Rainfall intensity | $R$ | $\text{m/s}$ | `mm_per_hour_to_m_per_sec`: $R_{\text{m/s}} = R_{\text{mm/h}} / 3{,}600{,}000$ |
| Horizontal velocities | $u, v$ | $\text{m/s}$ | Latent output head |
| Manning's roughness | $n$ | $\text{s}\cdot\text{m}^{-1/3}$ | Default $0.035$ |
| Gravity | $g$ | $\text{m/s}^2$ | $9.81\ \text{m/s}^2$ |

All spatial derivatives ($\partial/\partial x, \partial/\partial y$) and temporal derivatives ($\partial/\partial t$) are scaled by actual physical $\Delta x, \Delta y$ ($\text{m}$) and $\Delta t$ ($\text{s}$), rather than unitless grid indices.

---

## 3. Input & Output Tensor Shapes

### Input Tensor `x`: `(B, C_in, T, H, W)` (`C_in = 4`)
- Channel `0`: **DEM** $z_b(x, y)$ [$\text{m}$]
- Channel `1`: **Rainfall rate** $R(x, y, t)$ [$\text{m/s}$]
- Channel `2`: **Initial water depth** $h_0(x, y)$ [$\text{m}$]
- Channel `3`: **Manning's roughness** $n(x, y)$ [$\text{s}\cdot\text{m}^{-1/3}$]
- *(Internal)*: When `append_grid_coords=True`, 3 normalized spatio-temporal coordinate channels $(t, y, x) \in [0, 1]^3$ are concatenated before the lifting layer.

### Output (`PINOOutput`)
- **Primary supervised output**: `h` of shape `(B, T, H, W)` [$\text{m}$], constrained to $h \ge 0$ (`softplus` / `relu` / `none`).
- **Latent physics outputs**: `u, v` of shape `(B, T, H, W)` [$\text{m/s}$] predicted by an auxiliary `velocity_head` solely for evaluating the 2D SWE mass and momentum PDE residuals.

### Limitations of Available Labels
The selected **LarNO UKEA** core dataset provides terrain elevation (DEM), rainfall forcing, and water-depth ($h$) fields, **not** direct ground-truth horizontal velocity $(u, v)$ labels. Accordingly:
- `h` is trained with both supervised data loss and physics-informed losses.
- `u` and `v` are **unsupervised latent state variables** constrained strictly by the continuity equation, momentum equations, bed friction, and boundary conditions.
- **No supervised velocity accuracy is claimed.**

---

## 4. Boundary Conditions

Supported per-edge boundary conditions (`BoundaryConfig(north, south, east, west)`):
- **`"closed"`** (impermeable wall): Zero normal velocity ($u_n = 0$) and zero normal depth gradient ($\partial h / \partial n = 0$); zero boundary outflow flux in global mass balance.
- **`"open"`** (free outflow): Zero-gradient Neumann outflow ($\partial h / \partial n = 0$, $\partial u_n / \partial n = 0$); boundary outflow flux $\int_{\partial \Omega} h (\mathbf{u} \cdot \mathbf{n})\, ds$ is accounted for in global mass balance.
- **`"periodic"`**: Periodic matching across opposite edges.

---

## 5. Loss Formulation

The composite PINO training objective is:
$$\mathcal{L}_{\text{total}} = w_{\text{data}}\mathcal{L}_{\text{data}} + w_{\text{phys}}\mathcal{L}_{\text{physics}}$$

where $\mathcal{L}_{\text{physics}}$ explicitly combines:
1. **Shallow-Water PDE Residuals** ($\mathcal{L}_{\text{cont}}, \mathcal{L}_{\text{mom}, x}, \mathcal{L}_{\text{mom}, y}$): Mean-squared PDE residuals over `(B, T, H, W)` with dry-cell momentum masking.
2. **Global / Step-wise Mass Conservation** ($\mathcal{L}_{\text{mass}}$):
   $$\frac{V(t_{k+1}) - V(t_k)}{\Delta t} = \bar{Q}_{\text{rain}, k} - \bar{Q}_{\text{outflow}, k}, \qquad V(t_k) = \sum_{i,j} h_{i,j}(t_k)\Delta x \Delta y$$
   normalized by domain area $A = H W \Delta x \Delta y$ [$\text{m/s}$].
3. **Boundary-Condition Consistency** ($\mathcal{L}_{\text{BC}}$): Penalizes violations of wall / open / periodic constraints on all 4 edges.
4. **Initial-Condition Constraint** ($\mathcal{L}_{\text{IC}}$): $\text{MSE}\big(h(\cdot, t=0), h_0\big)$.
5. **Nonnegative Water Depth** ($\mathcal{L}_{\ge 0}$): $\text{mean}\big(\text{ReLU}(-h)^2\big)$.

---

## 6. Verified UKEA Dataset Dimensions & Splits

Source: `holmescao/LarNO-dataset` (`ukea_8m_5min` subset on Hugging Face).

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `dem.npy` | `(50, 120)` | `float32` | Static terrain elevation (m); building/border cells handled via `preprocess_dem` |
| `h.npy` | `(36, 50, 120)` | `float64` | Ground-truth water depth (m) over `T = 36` steps (3 h at 5-min = 300 s intervals) |
| `rainfall.npy` | `(360, 50, 120)` | `float64` | Rainfall forcing over `T_r = 360` sub-steps (10 sub-steps per 5-min water-depth frame) |

- **Official Event-Level Split**: Train (6 events) + Validation (2 events) from `ukea_train.txt`; Test (12 events) from `ukea_test.txt`.
- **Selected 32×32 Region**: Deterministic `(32, 32)` crop at `crop_y=9, crop_x=44` preserving `dx = dy = 8.0 m`.

---

## 7. Folder Structure

```text
PINO-Crestix-Probe/
├── src/crestix_pino/
│   ├── models/
│   │   ├── fno.py
│   │   └── pino.py
│   ├── physics/
│   │   ├── shallow_water.py
│   │   ├── residuals.py
│   │   ├── conservation.py
│   │   └── boundary_conditions.py
│   ├── losses/
│   │   ├── data_loss.py
│   │   └── physics_loss.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── train.py
│   │   ├── trainer.py
│   │   ├── checkpoint.py
│   │   └── config.py
│   ├── evaluation/
│   │   ├── metrics.py          # MAE, RMSE, R^2, relative depth, mass conservation, physics residual, horizon metrics
│   │   ├── evaluate.py         # Held-out flood event evaluation runner against LarNO ground truth
│   │   └── visualization.py    # Spatial error maps, time-series comparisons, horizon plots
│   └── inference/
│       ├── predictor.py        # Preprocessing, inference, postprocessing, and `predict(...)` contract
│       └── model_loader.py     # ModelContract and TorchScript / PyTorch / callable checkpoint loader
├── configs/
│   ├── default_pino.yaml
│   └── ukea_32x32.yaml
├── scripts/
│   ├── train_pino.py
│   ├── evaluate_pino.py        # CLI evaluation script
│   └── benchmark_inference.py  # CLI latency benchmark (model-only vs. full pipeline)
├── reports/
│   └── README.md               # Model card, limitations, and evaluation report template
├── tests/                      # Unit tests for models, physics, losses, dataset, training, metrics, inference, visualization, evaluation
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 8. Quick Start: Training, Evaluation & Inference

```bash
pip install -r requirements.txt
pip install -e ".[dev]"

# Reproducible Training Command (GPU/CPU)
python scripts/train_pino.py --config configs/ukea_32x32.yaml --download-data

# Small CPU Smoke Test
python scripts/train_pino.py --config configs/ukea_32x32.yaml --smoke-test --output-dir artifacts/smoke_test

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

```python
import numpy as np
from crestix_pino import predict

dem = np.zeros((32, 32), dtype=np.float32)                  # meters (m)
rainfall = np.full((12,), 45.0, dtype=np.float32)           # mm/h
initial_water_depth = np.zeros((32, 32), dtype=np.float32)  # meters (m)

result = predict(
    dem=dem,
    rainfall=rainfall,
    initial_water_depth=initial_water_depth,
    forecast_steps=12,
    model="path/to/best_pino_model.pt",
    device="cpu",
)
```

---

## 9. Running Unit Tests

```bash
pytest -v
```

---

## 10. Audit, Physics Design & Masked DEM Inference

- **Repository Audit Report:** [docs/PINO_AUDIT_REPORT.md](docs/PINO_AUDIT_REPORT.md)
- **Physics-Loss Design Document:** [docs/PINO_PHYSICS_DESIGN.md](docs/PINO_PHYSICS_DESIGN.md)

```python
import numpy as np
from crestix_pino.inference.predictor import PINOPredictor
from crestix_pino.models.pino import GridMetadata, PINOFloodModel, PINOModelConfig

cfg = PINOModelConfig(
    grid_metadata=GridMetadata(dx=8.0, dy=8.0, dt=300.0, grid_height=32, grid_width=32)
)
model = PINOFloodModel.from_config(cfg)
predictor = PINOPredictor(model=model, allow_dem_nan=True)

dem = np.full((32, 32), 15.0, dtype=np.float32)
dem[:4, :4] = np.nan  # building / invalid DEM cells -> masked to 0.0 m depth
res = predictor.predict(
    dem=dem,
    rainfall=np.full((12,), 36.0, dtype=np.float32),
    initial_water_depth=np.zeros((32, 32), dtype=np.float32),
    forecast_steps=12,
)
```


---

## 11. Official LarNO Shared Data Pipeline (100 Events)

This repository integrates the verified, shared LarNO dataset pipeline and contracts coordinated with the [ConvLSTM U-Net](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-) team:

### Catchment Architecture & Resolution Preservation
- **Futian Catchment (`region1_20m`):** 80 events at **20.0 m** resolution, $400 \times 560$ grid, 72 timesteps (5 min), 1:1 temporal alignment.
- **UKEA Catchment (`ukea_8m_5min`):** 20 events at **8.0 m** resolution, $50 \times 120$ grid, 360 rainfall sub-steps aligned to 36 depth frames (10:1 accumulation).
- Native spatial resolutions are strictly preserved across both catchments.

### Manifests & Training Statistics
- `data/event_manifest.csv`: Exhaustive metadata table for all 100 events (dimensions, timesteps, dtypes, units, alignment ratios).
- `data/split_manifest.json`: Invariant event-level stratified split (80 Train / 10 Val / 10 Test). Zero leakage between splits.
- `data/dataset_statistics.json`: Normalization parameters fitted **strictly on the 80 training events** (test events excluded).
- `configs/larno_data_pipeline.yaml`: Declarative shared pipeline configuration.
- `reports/data_quality_report.md`: Complete QA audit and verification report.
- `src/crestix_pino/data/larno_dataset.py`: Multi-catchment memory-mapped dataset loader supporting both PINO and ConvLSTM formats.

### Pipeline Tools
```bash
# Audit manifest integrity & event counts
python scripts/inspect_larno_dataset.py --manifest data/event_manifest.csv --split-manifest data/split_manifest.json

# Reproducible download utility (dry-run check)
python scripts/download_larno_dataset.py --manifest data/event_manifest.csv --catchment UKEA --split val --dry-run
```
