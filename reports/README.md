# PINO Model Card & Evaluation Report — Crestix Probe

> **Status:** Evaluation and inference harness implemented. Held-out test metrics and benchmark numbers are **TBD** pending upstream trained PINO checkpoint and held-out LarNO flood event datasets.

## 1. Model Overview

- **Model Family:** Physics-Informed Neural Operator (PINO) for 2D pluvial/urban flood depth forecasting
- **Target Integration:** Crestix Probe Backend (`skandakn/PINO-Crestix-Probe`)
- **Ground-Truth Reference:** LarNO shallow-water solver outputs
- **Predicted Output:** Water depth $h(t, y, x)$ in meters (`m`) — **velocity fields $(u, v)$ are not predicted or fabricated**

## 2. Agreed Input/Output Contract (`ModelContract`)

| Tensor / Argument | Shape | Units | Constraints |
|---|---|---|---|
| `dem` | `(H, W)` or `(B, H, W)` | `m` | Finite float32 |
| `rainfall` | scalar, `(T,)`, `(H, W)`, `(T, H, W)`, or batched equivalents | `mm/h` | Finite, `>= 0` |
| `initial_water_depth` | `(H, W)` or `(B, H, W)` | `m` | Matches `dem` shape, finite, `>= 0` |
| `forecast_steps` | `int` | steps | `>= 1` |
| **Output** `water_depth` | `(T, H, W)` or `(B, T, H, W)` | `m` | Finite float32 |

## 3. Held-Out Test Metrics (vs. LarNO Ground Truth)

*Populate via `python scripts/evaluate_pino.py --checkpoint <path> --dataset <held_out_dir> --output-dir outputs/evaluation`.*

| Metric | Value | Units |
|---|---|---|
| MAE | TBD | `m` |
| RMSE | TBD | `m` |
| $R^2$ | TBD | unitless |
| Relative L2 Depth Error | TBD | unitless |
| Wet-Cell Mean Relative Error | TBD | unitless |
| Mean Relative Mass-Conservation Error | TBD | unitless |
| Mean Abs Physics Residual (Diffusive-Wave Continuity) | TBD | `m/s` |
| Nonnegative-Depth Violations | TBD | count / fraction |

## 4. Inference Latency Benchmark

*Populate via `python scripts/benchmark_inference.py --checkpoint <path> --output-json outputs/benchmark.json`.*

| Setting | Model-Only Mean (ms) | Full Pipeline Mean (ms) | P95 Pipeline (ms) | Throughput (samples/s) |
|---|---|---|---|---|
| `batch=1, 64x64, T=24` | TBD | TBD | TBD | TBD |

## 5. Known Limitations & Numerical Stability Notes

1. **No Velocity Outputs:** The inference contract outputs water depth `h` only; physics residuals are evaluated using depth-based diffusive-wave continuity without fabricating velocity fields.
2. **Nonnegative Depth Violations:** Unconstrained neural operator outputs can produce small negative depths (`h < 0`) in dry cells. `PINOPredictor` always records `raw_negative_cell_count` and `raw_min_depth_m` in `metadata`; optional clamping (`clamp_negative=True`) can be enabled for downstream backend safety, though clamping alters raw mass conservation.
3. **Horizon Drift:** Autoregressive or multi-step rollouts may accumulate mass-conservation and phase errors at longer horizons; always inspect `horizon_metrics` per step.
4. **Domain & Unit Sensitivity:** Inputs must strictly adhere to SI units (`m` for DEM/depth, `mm/h` for rainfall). Silent unit or dimension conversions are rejected by `PINOPredictor.preprocess`.
5. **Diffusive-Wave Residual Approximation:** `compute_physics_residual` uses interior finite-difference gradients of $\eta = z_{\text{DEM}} + h$ with a small slope regularizer $\epsilon$, so boundary cells are excluded and steep shocks/discontinuities may exhibit elevated finite-difference residuals.

---

## 6. Cross-Model Comparison (ConvLSTM U-Net vs. PINO)

To compare PINO fairly against ConvLSTM-based U-Net on identical held-out LarNO test events (`shared_manifest.json` `v1.1.0`, 36-step 3 h rollout, `0.03 m` flood threshold):

- **PINO Export Guide:** [`../docs/COMMON_EVALUATION_FRAMEWORK.md`](../docs/COMMON_EVALUATION_FRAMEWORK.md)
- **Unified Evaluator & Comparison Outputs:** [`skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-) (`common_evaluator.py`, `outputs/comparison/model_comparison.csv`)
- **Joint Comparison Report & Checklist:** [`MODEL_COMPARISON_REPORT.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/reports/MODEL_COMPARISON_REPORT.md) | [`LIMITATIONS_AND_FAIRNESS_CHECKLIST.md`](https://github.com/skandakn/ConvLSTM-baased-U-Net-for-Crestix-Probe-/blob/main/reports/LIMITATIONS_AND_FAIRNESS_CHECKLIST.md)

