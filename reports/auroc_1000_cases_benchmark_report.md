# Model Benchmark Report: ConvLSTM-based U-Net vs. PINO
**Evaluation Metric:** Area Under the Receiver Operating Characteristic (AUROC) & Inundation Discrimination  
**Evaluation Scope:** 1,000 Spatiotemporal Flood Cases (1,005,283 Valid Overland Grid Cell Evaluations)  
**Benchmark Dataset:** LarNO Urban Flood Benchmark (`holmescao/LarNO-dataset`)  
**Operational Inundation Threshold:** $\tau = 0.03\text{ m}$ ($3\text{ cm}$)

---

## 1. Executive Summary & Verdict: Which Model is Better?

### **Primary Finding:**
- **For Flood Extent Classification and Water Boundary Discrimination: ConvLSTM-based U-Net is Superior.**
  - **ConvLSTM-based U-Net Global AUROC:** **0.9965** (Case Mean: $0.9944 \pm 0.0067$)
  - **PINO Global AUROC:** **0.9907** (Case Mean: $0.9863 \pm 0.0149$)
  - **Net AUROC Advantage:** $+0.0057$ in global pooled AUROC and $+0.0081$ in case-by-case mean AUROC favoring **ConvLSTM-based U-Net**.
  - **Operational F1-Score (@ 3 cm):** ConvLSTM-based U-Net achieves **0.9788** vs. PINO's **0.9598** ($\Delta = +0.0190$).
  - **Critical Success Index (CSI / Threat Score @ 3 cm):** ConvLSTM-based U-Net achieves **0.9585** vs. PINO's **0.9228** ($\Delta = +0.0358$).
  - **False Positive Rate at Operational Operating Point:** ConvLSTM-based U-Net incurs only **4.68%** false positive rate on dry cells vs. PINO's **12.64%** false positive rate.

### **Secondary Finding (Physical Consistency Trade-off):**
- While ConvLSTM-based U-Net dominates spatial classification, boundary sharpness, and cell-level ranking, **PINO remains superior in physical mass conservation** (mean relative volume error $< 1.2\%$ vs. $\approx 4.8\%$ for ConvLSTM) and zero-shot mesh resolution invariance.

---

## 2. Quantitative Metric Comparison (1,000 Cases)

| Metric | ConvLSTM-based U-Net | PINO (Physics-Informed Operator) | $\Delta$ (ConvLSTM - PINO) | Winning Architecture |
| :--- | :---: | :---: | :---: | :---: |
| **Global Pooled AUROC** | **0.9965** | 0.9907 | **+0.0057** | **ConvLSTM-based U-Net** |
| **1,000-Case Mean AUROC** | **0.9944** | 0.9863 | **+0.0081** | **ConvLSTM-based U-Net** |
| **Case AUROC Std. Dev.** | **$\pm$ 0.0067** | $\pm$ 0.0149 | **-0.0082 (Lower Variance)** | **ConvLSTM-based U-Net** |
| **Case AUROC Median** | **0.9969** | 0.9926 | **+0.0043** | **ConvLSTM-based U-Net** |
| **Case AUROC IQR [25%, 75%]** | **[0.9933, 0.9984]** | [0.9829, 0.9959] | **Tighter Spread** | **ConvLSTM-based U-Net** |
| **Precision-Recall AUC (PR-AUC)** | **0.9984** | 0.9960 | **+0.0024** | **ConvLSTM-based U-Net** |
| **Precision ($\tau = 0.03\text{ m}$)** | **0.9776** | 0.9417 | **+0.0359** | **ConvLSTM-based U-Net** |
| **Recall ($\tau = 0.03\text{ m}$)** | **0.9800** | 0.9786 | **+0.0014** | **ConvLSTM-based U-Net** |
| **F1-Score ($\tau = 0.03\text{ m}$)** | **0.9788** | 0.9598 | **+0.0190** | **ConvLSTM-based U-Net** |
| **Critical Success Index (CSI)** | **0.9585** | 0.9228 | **+0.0358** | **ConvLSTM-based U-Net** |
| **False Positive Rate (FPR @ 3cm)** | **0.0468 (4.68%)** | 0.1264 (12.64%) | **-7.96% (Fewer False Alarms)** | **ConvLSTM-based U-Net** |
| **MAE (Water Depth, m)** | **0.0136 m** | 0.0177 m | **-0.0041 m** | **ConvLSTM-based U-Net** |
| **RMSE (Water Depth, m)** | **0.0191 m** | 0.0244 m | **-0.0053 m** | **ConvLSTM-based U-Net** |
| **Global Mass Conservation Error** | $\sim 4.8\%$ | **$< 1.2\%$** | $-3.6\%$ (Physics Enforced) | **PINO** |

---

## 3. Hydrological & Architectural Analysis: Why ConvLSTM-based U-Net Wins on AUROC

1. **U-Net Skip Connections & High-Frequency Spatial Fidelity:**
   - In urban hydrology, floodwater propagation is bounded by sharp micro-topographic barriers (curbs, road crowns, building walls, storm grates).
   - The U-Net encoder-decoder preserves high-resolution spatial feature maps via skip connections, allowing the network to delineate exact sub-grid wet/dry interfaces without spatial blur.

2. **Spectral Truncation (Gibbs Phenomenon) in PINO:**
   - PINO operates via Fourier Neural Operators (FNO) with spectral convolution in frequency space.
   - Truncating higher Fourier modes ($k_{\text{max}} = 12$) naturally acts as a low-pass filter. While this ensures smooth physical velocity and depth fields, it creates mild spatial diffusion and Gibbs ringing along discontinuous, steep wetting fronts.
   - This diffusion causes near-boundary dry cells to be predicted with shallow depths ($0.01 - 0.04\text{ m}$), elevating PINO's False Positive Rate (12.64% vs 4.68%).

3. **Loss Function Inductive Biases:**
   - ConvLSTM was optimized with `MaskedFloodDepthLoss` with an explicit wet-cell loss weight ($w_{\text{wet}} = 2.0$), heavily prioritizing the accuracy of cells near and above $\tau = 0.03\text{ m}$.
   - PINO balances data loss against Shallow Water Equation continuity/momentum residuals and boundary condition penalties, distributing gradient capacity to enforce physical equilibrium rather than purely maximizing boundary classification sharpness.

---

## 4. Benchmark Artifacts
- **High-Resolution Figure:** `reports/auroc_comparison_1000_cases.png`
- **Machine-Readable Metrics:** `reports/auroc_benchmark_1000_cases.json`
- **Evaluation Runner Script:** `benchmark_auroc_1000_cases.py`
