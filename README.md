# Crestix Probe — PINO Dataset Adapter & Training Pipeline

Reproducible PyTorch dataset adapter and training pipeline for the Physics-Informed Neural Operator (PINO) on the **LarNO UKEA 32×32** urban flood benchmark.

## Verified UKEA Dataset Dimensions

Source: `holmescao/LarNO-dataset` (`ukea_8m_5min` subset on Hugging Face).

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `dem.npy` | `(50, 120)` | `float32` | Static terrain elevation (m); building/border cells handled via `preprocess_dem` |
| `h.npy` | `(36, 50, 120)` | `float64` | Ground-truth water depth (m) over `T = 36` steps (3 h at 5-min = 300 s intervals) |
| `rainfall.npy` | `(360, 50, 120)` | `float64` | Rainfall forcing over `T_r = 360` sub-steps (10 sub-steps per 5-min water-depth frame) |

### Official Event-Level Split

All splits are strictly **event-level** (never splitting frames or pixels across splits):
- **Train (6 events)** + **Validation (2 events)**: carved deterministically from the 8 official `ukea_train.txt` events.
- **Test (12 events)**: held-out official `ukea_test.txt` events.

### Selected 32×32 Region & Input/Target Tensors

- **Spatial region**: Deterministic `(32, 32)` crop at `crop_y=9, crop_x=44` preserving the native `dx = dy = 8.0 m` resolution.
- **Input tensor** `(B, 4, T, 32, 32)`:
  1. `dem_m`: terrain elevation (m)
  2. `rainfall_m_s`: aligned rainfall forcing (m/s)
  3. `initial_water_depth`: initial water depth `h(t=t0)` (m)
  4. `manning_n`: Manning roughness coefficient (`0.035 s·m^(-1/3)`)
- **Target tensor** `(B, T, 32, 32)`: future water-depth maps (m).
- **Memory-efficient loading**: `UKEAPinoDataset` opens `.npy` files with `np.load(..., mmap_mode="r")` and slices only the requested temporal window and `32 × 32` patch per `__getitem__`.

## Folder Structure

```text
PINO-Crestix-Probe/
├── src/crestix_pino/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── preprocessing.py
│   └── training/
│       ├── train.py
│       ├── trainer.py
│       ├── checkpoint.py
│       └── config.py
├── configs/
│   └── ukea_32x32.yaml
├── scripts/
│   └── train_pino.py
├── tests/
│   ├── test_dataset.py
│   └── test_training_pipeline.py
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install matplotlib tensorboard huggingface_hub
```

### 2. Reproducible Training Command (GPU/CPU)

```bash
python scripts/train_pino.py --config configs/ukea_32x32.yaml --download-data
```

### 3. Small CPU Smoke Test

```bash
python scripts/train_pino.py --config configs/ukea_32x32.yaml --smoke-test --output-dir artifacts/smoke_test
```

### 4. Resume Training from Checkpoint

```bash
python scripts/train_pino.py --config configs/ukea_32x32.yaml --resume artifacts/ukea_32x32/checkpoints/last.pt
```

### 5. Run Unit & Smoke Tests

```bash
pytest -v
```

## Google Drive Artifacts

Large binary artifacts (model checkpoints `*.pt`, TensorBoard event logs `tensorboard/`, training histories `history.json`, loss plots `loss_curves.png`, and prediction arrays) are stored in the shared Google Drive folder rather than GitHub:

- **Shared Drive Path:** `Crestix-Probe/PINO/ukea_32x32/`
  - `checkpoints/best.pt` & `checkpoints/best.metadata.json`
  - `checkpoints/last.pt` & `checkpoints/last.metadata.json`
  - `tensorboard/` (view with `tensorboard --logdir artifacts/ukea_32x32/tensorboard`)
  - `history.json`
  - `loss_curves.png`

To point training outputs directly to a mounted Google Drive folder (e.g., in Colab or local Drive sync):

```bash
python scripts/train_pino.py --config configs/ukea_32x32.yaml --output-dir "/content/drive/MyDrive/Crestix-Probe/PINO/ukea_32x32"
```

