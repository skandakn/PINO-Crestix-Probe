# LarNO Shared Dataset & Manifests — PINO Crestix Probe

This directory hosts the official dataset manifests, split configurations, and train-only statistical moments for the **100-event LarNO Urban Flood Benchmark** shared with the [ConvLSTM U-Net](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-) team.

---

## 1. Deliverables Catalog

| File | Type | Description |
| :--- | :---: | :--- |
| `event_manifest.csv` | CSV | Complete catalog of all 100 events (80 Futian + 20 UKEA), detailing spatial resolutions, grid dimensions, timestep counts, array dtypes, alignment ratios, and units |
| `split_manifest.json` | JSON | Formal event-level stratified split: **80 Train / 10 Val / 10 Test** (0 leakage) |
| `dataset_statistics.json` | JSON | Normalization parameters fitted strictly across the 80 training events |

---

## 2. Catchment Geometry & Invariant Resolutions

| Catchment | Events | Native Resolution | Spatial Shape | Temporal Steps | Temporal Alignment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Futian, Shenzhen** | 80 | **20.0 m** | $(400, 560)$ | 72 frames (5-min) | 1:1 direct match |
| **UKEA Catchment** | 20 | **8.0 m** | $(50, 120)$ | 36 frames (5-min) | 10:1 decimation ($360 \to 36$) |

Native spatial resolutions and coordinate grids are strictly preserved; no artificial spatial resampling is performed between catchments.

---

## 3. Storage & Download Policy

Raw arrays (~20 GB) are hosted on Google Drive (`Crestix Probe Shared / ml_data / larno_raw_100_events`) and can be reproducibly downloaded via:

```bash
# Dry-run download verification
python scripts/download_larno_dataset.py --manifest data/event_manifest.csv --dry-run

# Download specific catchment & split
python scripts/download_larno_dataset.py --manifest data/event_manifest.csv --catchment UKEA --split val --output-dir data/raw_larno
```

Memory mapping (`np.load(..., mmap_mode="r")`) in `src/crestix_pino/data/larno_dataset.py` ensures low RAM utilization during training and inference.
