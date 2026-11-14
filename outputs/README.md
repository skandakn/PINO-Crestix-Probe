# Outputs & External Artifacts — PINO-Crestix-Probe

Per project repository policy, **no large datasets (`.npy`, `.npz`) or binary PyTorch checkpoints (`.pt`, `.pth`, `.ckpt`) are committed to GitHub**.

## 1. Local Ignored Output Directories
When running training, validation, evaluation, inference, prediction export, or benchmarks locally, outputs are written to git-ignored directories:
- `artifacts/ukea_32x32/checkpoints/best.pt` & `last.pt` (and `artifacts/ukea_32x32/best_pino_model.pt` per common-evaluation export guide) + `.metadata.json`, `history.json`, `loss_curves.png`
- `artifacts/smoke_test/checkpoints/best.pt` (CPU `--smoke-test` run with synthetic UKEA fallback)
- `local_cache/pino_predictions_v1.1.0/` (`prediction_manifest.json` + 10 `<event_id>.npz` files exported for `common_evaluator.py`)
- `outputs/evaluation/` (aggregate/per-event metrics JSON, `.npz` predictions, spatial error PNGs from `scripts/evaluate_pino.py`)
- `outputs/smoke_validation.json`, `outputs/sample_inference.json`, `outputs/benchmark.json`

## 2. External Google Drive Locations (`Crestix-Probe`)
Large trained model weights, `.npz` prediction archives, and raw 1000-event LarNO arrays are hosted externally on the team's shared Google Drive:

| Artifact | Shared Google Drive Path | Local Ignored Destination |
|---|---|---|
| Raw 1000-Event LarNO Dataset (~20 GB) | `GoogleDrive://Crestix-Probe/Datasets/LarNO-UKEA-1000Events/` | `data/raw_larno/` |
| UKEA `ukea_8m_5min` Subset | `GoogleDrive://Crestix-Probe/Datasets/LarNO-UKEA-1000Events/ukea_8m_5min/` | `data/ukea_8m_5min/` |
| Trained PINO UKEA `32x32` Checkpoint | `GoogleDrive://Crestix-Probe/PINO/checkpoints/best.pt` | `artifacts/ukea_32x32/checkpoints/best.pt` |
| Trained PINO Common-Eval Checkpoint Alias | `GoogleDrive://Crestix-Probe/PINO/checkpoints/best_pino_model.pt` | `artifacts/ukea_32x32/best_pino_model.pt` |
| Trained PINO Latest Epoch State | `GoogleDrive://Crestix-Probe/PINO/checkpoints/last.pt` | `artifacts/ukea_32x32/checkpoints/last.pt` |
| PINO Held-Out Predictions (`v1.1.0`) | `GoogleDrive://Crestix-Probe/PINO/predictions_v1.1.0/` | `local_cache/pino_predictions_v1.1.0/` |

Helper command:
```bash
python scripts/download_checkpoints.py --check
```
