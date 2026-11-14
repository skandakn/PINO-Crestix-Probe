# PINO Physics Dimensional Analysis — Crestix Probe

**Repository:** `https://github.com/skandakn/PINO-Crestix-Probe`  
**Scope:** Independent dimensional verification, finite-difference operator scaling, and composite loss homogeneity audit across `src/crestix_pino/physics/`, `src/crestix_pino/losses/`, `src/crestix_pino/models/pino.py`, `src/crestix_pino/data/preprocessing.py`, `src/crestix_pino/training/trainer.py`, `src/crestix_pino/inference/model_loader.py`, `src/crestix_pino/evaluation/metrics.py`, and `configs/*.yaml`.

---

## 1. Fundamental SI Variables and Physical Parameters

| Symbol | Code Identifier / Module | Physical Quantity | SI Unit | Notes / Assumptions |
|---|---|---|---|---|
| $h$ | `h` (`PINOOutput.h`) | Water depth | $\mathrm{m}$ | Post-activation $h = \mathrm{softplus}(h_{\text{raw}}, \beta=5) > 0$ |
| $h_{\text{raw}}$ | `output.h_raw` | Unconstrained head output | $\mathrm{m}$ (latent) | Evaluated inside `PhysicsLoss` non-negativity term |
| $u, v$ | `output.u, output.v` | Depth-averaged velocity $(x, y)$ | $\mathrm{m\,s^{-1}}$ | Latent fields; **not** supervised by LarNO ground-truth data |
| $z_b$ | `dem` | Bed / terrain elevation | $\mathrm{m}$ | `preprocess_dem` fills `NaN` cells with $\max(z_b) + 50\ \mathrm{m}$ |
| $\eta = z_b + h$ | `eta` (`metrics.py`) | Water surface elevation | $\mathrm{m}$ | Used in diffusive-wave evaluation metric |
| $R$ | `rainfall_mm_h` $\to$ `rainfall_m_s` | Rainfall rate | $\mathrm{mm\,h^{-1}} \to \mathrm{m\,s^{-1}}$ | Exact conversion: $1\ \mathrm{mm\,h^{-1}} = \frac{1}{3{,}600{,}000}\ \mathrm{m\,s^{-1}}$ |
| $I$ | `infiltration` | Infiltration sink rate | $\mathrm{m\,s^{-1}}$ | Defaults to $0$ |
| $g$ | `GRAVITY_M_S2` | Gravitational acceleration | $\mathrm{m\,s^{-2}}$ | $9.81\ \mathrm{m\,s^{-2}}$ |
| $n$ | `default_manning_n` | Manning roughness coefficient | $\mathrm{s\,m^{-1/3}}$ | Default $0.035\ \mathrm{s\,m^{-1/3}}$ |
| $\Delta x, \Delta y$ | `dx, dy` (`cell_size_m`) | Grid cell spacing | $\mathrm{m}$ | UKEA: $8.0\ \mathrm{m}$; defaults disagree ($2.0\ \mathrm{m}$, $10.0\ \mathrm{m}$) |
| $\Delta t$ | `dt` (`dt_seconds`) | Forecast time step | $\mathrm{s}$ | UKEA: $300.0\ \mathrm{s}$; defaults disagree ($60.0\ \mathrm{s}$, $30.0\ \mathrm{s}$) |
| $h_{\varepsilon}$ | `wet_dry_eps` | Wet-dry threshold depth | $\mathrm{m}$ | Default $10^{-3}\ \mathrm{m}$ ($1\ \mathrm{mm}$) |
| $S_{fx}, S_{fy}$ | `s_fx, s_fy` | Manning friction slopes | $1$ (dimensionless) | Derived in Section 2.1 |
| $V(t)$ | `compute_domain_volume` | Total domain water volume | $\mathrm{m^3}$ | $\sum_{i,j} h_{i,j}(t)\,\Delta x\,\Delta y$ |
| $Q_{\text{rain}}, Q_{\text{out}}$ | `q_rain, q_out` | Volumetric inflow / outflow | $\mathrm{m^3\,s^{-1}}$ | $Q_{\text{out}}$ depends on open-edge $(u,v)$ |

---

## 2. Equation-by-Equation Dimensional Verification

### 2.1 Manning Friction Slope (`shallow_water.py:compute_manning_friction_slope`)

Implemented formula:

$$S_{fx} = \mathbf{1}_{\{h > h_{\varepsilon}\}} \frac{n^2\,u\,\sqrt{u^2 + v^2 + \epsilon}}{\max(h, h_{\varepsilon})^{4/3}}, \qquad \epsilon = 10^{-12}\ \mathrm{m^2\,s^{-2}}$$

Dimensional check:

$$[S_{fx}] = \frac{\left(\mathrm{s\,m^{-1/3}}\right)^2 \cdot \left(\mathrm{m\,s^{-1}}\right) \cdot \left(\mathrm{m\,s^{-1}}\right)}{\mathrm{m}^{4/3}} = \frac{\mathrm{s^2\,m^{-2/3}} \cdot \mathrm{m^2\,s^{-2}}}{\mathrm{m^{4/3}}} = \frac{\mathrm{m^{4/3}}}{\mathrm{m^{4/3}}} = 1 \quad (\text{dimensionless}).$$

- **Verdict:** **Dimensionally consistent.**
- **Stiffness / Discontinuity Finding:** Because $\mathbf{1}_{\{h > h_{\varepsilon}\}}$ is a hard step function at $h_{\varepsilon} = 10^{-3}\ \mathrm{m}$, both $S_{fx}$ and $g h S_{fx}$ jump discontinuously from $0$ to $\frac{g n^2 u |u|}{h_{\varepsilon}^{1/3}}$ as $h$ crosses $10^{-3}\ \mathrm{m}$ (`drag_at_threshold_plus_m2_s2 = 0.1182 m^2/s^2` and sensitivity $\left|\partial(g h S_f)/\partial h\right| = 39.25\ \mathrm{m\,s^{-2}}$ at $u = 1\ \mathrm{m\,s^{-1}}$ in Synthetic Test `S4`).

---

### 2.2 Depth Continuity PDE (`residuals.py:compute_swe_residuals`)

Implemented formula (with $h_+ = \max(h, 0)$):

$$r_{\text{mass}} = \frac{\partial h_+}{\partial t} + \frac{\partial (h_+ u)}{\partial x} + \frac{\partial (h_+ v)}{\partial y} - (R - I)$$

Dimensional check:

$$\left[\frac{\partial h_+}{\partial t}\right] = \frac{\mathrm{m}}{\mathrm{s}}, \quad \left[\frac{\partial (h_+ u)}{\partial x}\right] = \frac{\mathrm{m \cdot m\,s^{-1}}}{\mathrm{m}} = \frac{\mathrm{m}}{\mathrm{s}}, \quad [R - I] = \frac{\mathrm{m}}{\mathrm{s}}.$$

- **Verdict:** **All terms share unit $\mathrm{m\,s^{-1}}$.**
- **Discretization Check (`time_derivative`, `spatial_derivatives_2d`):**
  - Interior stencils use 2nd-order central differences (`1/(2 dt)`, `1/(2 dx)`, `1/(2 dy)`).
  - Domain boundaries fall back to 1st-order one-sided differences (`1/dt`, `1/dx`, `1/dy`), leaving $O(\Delta x)$ edge truncation errors on quadratic profiles (`S1`: `quadratic_mom_x_edge_max_abs = 0.1556 m^2/s^2`).
  - `_FallbackPINOLoss` in `training/trainer.py` instead uses `torch.roll`, imposing **periodic** differences across non-periodic urban DEM boundaries and conflating east/west and north/south jumps.

---

### 2.3 Conservative Depth-Integrated Momentum Equations (`residuals.py:compute_swe_residuals`)

Implemented formulas (multiplied by $\mathbf{1}_{\{h_+ > h_{\varepsilon}\}}$ when `mask_dry_momentum=True`):

$$r_{\text{mom},x} = \frac{\partial (h_+ u)}{\partial t} + \frac{\partial}{\partial x}\!\left(h_+ u^2 + \frac{1}{2} g h_+^2\right) + \frac{\partial (h_+ u v)}{\partial y} + g h_+ \left(\frac{\partial z_b}{\partial x} + S_{fx}\right)$$

$$r_{\text{mom},y} = \frac{\partial (h_+ v)}{\partial t} + \frac{\partial (h_+ u v)}{\partial x} + \frac{\partial}{\partial y}\!\left(h_+ v^2 + \frac{1}{2} g h_+^2\right) + g h_+ \left(\frac{\partial z_b}{\partial y} + S_{fy}\right)$$

Dimensional check for $r_{\text{mom},x}$:

$$\left[\frac{\partial (h_+ u)}{\partial t}\right] = \frac{\mathrm{m^2\,s^{-1}}}{\mathrm{s}} = \frac{\mathrm{m^2}}{\mathrm{s^2}}, \qquad \left[\frac{\partial (h_+ u^2)}{\partial x}\right] = \frac{\mathrm{m \cdot m^2\,s^{-2}}}{\mathrm{m}} = \frac{\mathrm{m^2}}{\mathrm{s^2}},$$

$$\left[\frac{\partial (\frac{1}{2} g h_+^2)}{\partial x}\right] = \frac{(\mathrm{m\,s^{-2}})\,\mathrm{m^2}}{\mathrm{m}} = \frac{\mathrm{m^2}}{\mathrm{s^2}}, \qquad \left[g h_+ \left(\frac{\partial z_b}{\partial x} + S_{fx}\right)\right] = (\mathrm{m\,s^{-2}})\,\mathrm{m} \cdot \left(\frac{\mathrm{m}}{\mathrm{m}} + 1\right) = \frac{\mathrm{m^2}}{\mathrm{s^2}}.$$

- **Verdict:** **Individually dimensionally homogeneous in $\mathrm{m^2\,s^{-2}}$.**
- **Missing Building Mask Defect:** `spatial_derivatives_2d(dem_4d)` is computed **without `valid_mask`**. Whenever building `NaN` cells are filled with $+50\ \mathrm{m}$ by `preprocess_dem`, unmasked central differences across the wall create artificial bed slopes $\partial z_b / \partial x \sim \frac{50\ \mathrm{m}}{2 \cdot 8\ \mathrm{m}} = 3.125$ on neighboring street cells (`S6`). Even at $u = v = 0$, $g h \frac{\partial z_b}{\partial x} \approx 4.3\ \mathrm{m^2\,s^{-2}}$!

---

### 2.4 Global Mass Conservation Residual (`conservation.py:compute_mass_conservation_residual`)

Implemented formula over each interval $[t_k, t_{k+1}]$:

$$V_k = \Delta x\,\Delta y \sum_{i,j} h_{k,i,j}\ [\mathrm{m^3}], \qquad Q_{\text{rain},k} = \Delta x\,\Delta y \sum_{i,j} R_{k,i,j}\ [\mathrm{m^3\,s^{-1}}],$$

$$Q_{\text{out},k} = \text{signed boundary flux sums } \sum_{\text{open}} (h\,\mathbf{u}\cdot\hat{\mathbf{n}})\,\Delta s\ [\mathrm{m^3\,s^{-1}}],$$

$$r_{\text{mass\_global},k} = \frac{\frac{V_{k+1} - V_k}{\Delta t} - \left(\bar{Q}_{\text{rain},k} - \bar{Q}_{\text{out},k}\right)}{H\,W\,\Delta x\,\Delta y}.$$

Dimensional check:

$$[r_{\text{mass\_global}}] = \frac{\mathrm{m^3\,s^{-1}} - \mathrm{m^3\,s^{-1}}}{\mathrm{m^2}} = \frac{\mathrm{m}}{\mathrm{s}}.$$

- **Verdict:** **Dimensionally consistent in $\mathrm{m\,s^{-1}}$.**
- **Boundary-Flux Support Defect (`S7`):** Because LarNO UKEA datasets provide only $(h, z_b, R)$ and never provide ground-truth velocity $(u, v)$ or perimeter discharge gauges, any edge marked `"open"` introduces unconstrained latent $(u, v)$ gauge freedom that can cancel arbitrary volume errors in $V_{k+1} - V_k$ (`S7`: wrong shrinking basin achieves $|r_{\text{mass\_global}}| = 2.71 \times 10^{-20}\ \mathrm{m\,s^{-1}}$).

---

### 2.5 Boundary Consistency Loss (`boundary_conditions.py:compute_boundary_consistency_loss`) — **DIMENSIONALLY INCONSISTENT**

Under `BoundaryConfig(north="closed", south="open", east="closed", west="closed")`, `compute_boundary_consistency_loss` computes an **unweighted arithmetic mean** (`mean(terms)`) of terms with **four incompatible physical dimensions**:

| Boundary Condition | Term Added to `terms` | Code Expression | Physical Unit |
|---|---|---|---|
| Closed edge (N, E, W) | Normal depth gradient | `(dh / dn)^2` | **Dimensionless** $[1]$ |
| Closed edge (N, E, W) | Normal velocity squared | `u_n^2` or `v_n^2` | $\mathrm{m^2\,s^{-2}}$ |
| Open edge (S) | Normal depth gradient | `(dh / dn)^2` | **Dimensionless** $[1]$ |
| Open edge (S) | Normal velocity gradient | `(dv / dn)^2` | $\mathrm{s^{-2}}$ |
| Periodic edge (optional) | Opposite-edge depth / velocity diff | `(h_0 - h_{-1})^2`, `(u_0 - u_{-1})^2` | $\mathrm{m^2}$ and $\mathrm{m^2\,s^{-2}}$ |

$$\mathcal{L}_{\text{bc}} = \mathrm{mean}\Big(\underbrace{[1]}_{\text{slope}^2},\ \underbrace{[\mathrm{m^2\,s^{-2}}]}_{u_n^2},\ \underbrace{[\mathrm{s^{-2}}]}_{(\partial v/\partial n)^2},\ \underbrace{[\mathrm{m^2}]}_{\text{periodic }\Delta h^2}\Big) \quad \Longrightarrow \quad \textbf{Dimensionally invalid sum!}$$

Because $\Delta x = 8\ \mathrm{m}$ yields $(\partial v / \partial n)^2 \sim v^2 / 64\ [\mathrm{s^{-2}}]$ while $v^2 \sim 1\ [\mathrm{m^2\,s^{-2}}]$ and $(\partial h / \partial n)^2 \sim 10^{-4}\ [1]$, the velocity term outweighs the depth-slope and velocity-gradient terms by $64\times$ to $10{,}000\times$ purely due to unit choice.

---

### 2.6 Composite Physics & Data Losses (`losses/physics_loss.py` & `losses/data_loss.py`) — **DIMENSIONALLY INCONSISTENT**

`PhysicsLoss.forward` forms a weighted linear combination of Mean Squared Errors whose physical units span **eight orders of magnitude**:

| Loss Component | Default Weight $w_i$ | Underlying Residual Unit | Squared MSE Unit | UKEA Probe Gradient $\\|\nabla_{h_{\text{raw}}}\mathcal{L}_i\\|_2$ |
|---|---|---|---|---|
| `data_loss` (`DataLoss`) | `data_weight = 1.0` | $\mathrm{m}$ | $\mathrm{m^2}$ | $2.069 \times 10^{-2}$ |
| `pde_continuity` | `1.0` | $\mathrm{m\,s^{-1}}$ | $\mathrm{m^2\,s^{-2}}$ | $4.515 \times 10^{-5}$ |
| `pde_momentum_x` | `0.25` | $\mathrm{m^2\,s^{-2}}$ | $\mathrm{m^4\,s^{-4}}$ | $8.311 \times 10^{-3}$ (UKEA) / **$11.516$ (Bldg-NaN)** |
| `pde_momentum_y` | `0.25` | $\mathrm{m^2\,s^{-2}}$ | $\mathrm{m^4\,s^{-4}}$ | $7.067 \times 10^{-3}$ (UKEA) / **$11.516$ (Bldg-NaN)** |
| `mass_conservation` | `0.5` | $\mathrm{m\,s^{-1}}$ | $\mathrm{m^2\,s^{-2}}$ | $1.051 \times 10^{-7}$ |
| `boundary_condition` | `0.5` | Mixed (`1`, $\mathrm{m\,s^{-1}}$, $\mathrm{s^{-1}}$) | Mixed | $2.909 \times 10^{-4}$ |
| `initial_condition` | `1.0` | $\mathrm{m}$ | $\mathrm{m^2}$ | $3.557 \times 10^{-2}$ |
| `nonnegative_depth` | `1.0` | $\mathrm{m}$ (applied to `h_raw`!) | $\mathrm{m^2}$ | $2.048 \times 10^{-2}$ |

Two severe consequences follow directly from adding $[\mathrm{m^2\,s^{-2}}] + [\mathrm{m^4\,s^{-4}}] + [\mathrm{m^2}]$:
1. **Continuity & Global Mass Vanishing:** Because water depths on urban streets are $h \sim 0.02\text{–}0.15\ \mathrm{m}$ over $\Delta t = 300\ \mathrm{s}$, continuity residuals are $\mathcal{O}(10^{-5}\text{–}10^{-4})\ \mathrm{m\,s^{-1}}$, giving MSE $\mathcal{O}(10^{-9}\text{–}10^{-5})\ \mathrm{m^2\,s^{-2}}$ and continuity gradients $458\times$ smaller than `data_loss` on UKEA (`4.5e-5` vs `2.1e-2`) and $184\times$ smaller than `pde_momentum_x`.
2. **Momentum Explosion on Building Walls:** When `NaN` buildings trigger `+50 m` DEM walls without masking `PhysicsLoss`, `pde_momentum_x` gradients spike to $11.52$, making `0.1 * total_physics` exceed `data_loss` gradients by **$20.75\times$** (or **$207.5\times$** at `physics_weight = 1.0`).

---

### 2.7 Evaluation Metrics (`evaluation/metrics.py`)

1. `compute_mass_conservation_error` evaluates $\left|\sum h(t)\,\Delta x\,\Delta y - \left(V_0 + \sum_{s=0}^{t} \sum R_s\,\Delta t\,\Delta x\,\Delta y\right)\right| / V_{\text{expected}}$ in $\mathrm{m^3}/\mathrm{m^3}$ (dimensionless). It sums `step_rain_vol` from `t=0` and assumes **zero open-boundary outflow**.
2. `compute_physics_residual` evaluates the diffusive-wave continuity residual in $\mathrm{m\,s^{-1}}$:
   $$R_{\text{diffusive}} = \frac{\partial h}{\partial t} + \nabla\cdot\!\left(-\frac{1}{n} \frac{h_{\text{mid}}^{5/3}}{\sqrt{\sqrt{|\nabla \eta|^2 + \epsilon^2}}}\,\nabla \eta\right) - R, \qquad \eta = z_b + h_{\text{mid}}.$$
   Dimensional check of Manning diffusive conveyance $\frac{1}{n} h^{5/3} |\nabla \eta|^{-1/2}$:
   $$\left[\frac{1}{n}\,h^{5/3}\,\frac{\nabla \eta}{|\nabla \eta|^{1/2}}\right] = \frac{\mathrm{m^{1/3}}}{\mathrm{s}} \cdot \mathrm{m^{5/3}} \cdot 1 = \frac{\mathrm{m^2}}{\mathrm{s}} \quad \Longrightarrow \quad [\nabla \cdot \mathbf{q}] = \frac{\mathrm{m^2\,s^{-1}}}{\mathrm{m}} = \frac{\mathrm{m}}{\mathrm{s}}.$$
   Dimensionally valid, **provided** true `(dt_seconds, cell_size_m)` are supplied.

---

## 3. Spatial and Temporal Scale Mismatch Audit Across Repository Defaults

Three **incompatible** sets of grid/time constants appear as defaults across the repository:

| Module / Config | $\Delta x, \Delta y$ ($\mathrm{m}$) | $\Delta t$ ($\mathrm{s}$) | `physics_weight` | Impact When Applied to UKEA (`8 m`, `300 s`) |
|---|---|---|---|---|
| **UKEA Dataset (`configs/ukea_32x32.yaml`)** | **$8.0\ \mathrm{m}$** | **$300.0\ \mathrm{s}$** | $0.1$ | Reference LarNO ground-truth scales ($32 \times 32$ crop `[9:41, 44:76]`, `36` frames = $3\ \mathrm{h}$ at $5\ \mathrm{min}$) |
| `ShallowWaterConfig` / `configs/default_pino.yaml` | $2.0\ \mathrm{m}$ | $60.0\ \mathrm{s}$ | $1.0$ | $\partial/\partial t$ inflated **$5\times$**; $\partial/\partial x$ inflated **$4\times$**; momentum MSE inflated **$\sim 16\times$** (`0.1159` vs `0.00725`), total weighted physics inflated **$17.5\times$** (`0.1270` vs `0.00726`) |
| `ModelContract` (`inference/model_loader.py`) & `evaluation/metrics.py` defaults | $10.0\ \mathrm{m}$ | $30.0\ \mathrm{s}$ | N/A | $\partial/\partial t$ inflated **$10\times$**; closed mass rain volume scaled by $\frac{30}{300} \cdot \frac{100}{64} = 0.156\times$; diffusive $\nabla\cdot\mathbf{q}$ scaled by $(8/10)^{3/2} \approx \mathbf{0.7155\times}$ (`RMSE` shrinks artificially by ~28.5%, from `0.01108` to `0.00791 m/s` on `r200y_p0.5_d3h_1`) |

---

## 4. Recommended Nondimensionalization

Choose characteristic domain scales tied explicitly to dataset metadata:
- Depth scale $H_0$ (e.g., $0.25\ \mathrm{m}$ or $\max(H_{\text{rain}}, 0.1\ \mathrm{m})$), length scale $L_0 = \Delta x\ (\mathrm{m})$, time scale $T_0 = \Delta t\ (\mathrm{s})$, velocity scale $U_0 = \sqrt{g H_0}\ (\mathrm{m\,s^{-1}})$ (or Manning velocity $\frac{1}{n} H_0^{2/3} S_0^{1/2}$).
- Rescale every residual before squaring so **every loss term is dimensionless $\mathcal{O}(1)$**:
  - Continuity & global mass residual: $\tilde{r}_{\text{mass}} = \frac{T_0}{H_0}\,r_{\text{mass}}\ [1]$ (note: using $H_0/T_0$ matches storage scale $\partial h/\partial t$).
  - Momentum residual: $\tilde{r}_{\text{mom}} = \frac{1}{g H_0 S_{\text{ref}}}\,r_{\text{mom}}\ [1]$ or $\frac{T_0}{H_0 U_0}\,r_{\text{mom}}\ [1]$, where a typical terrain slope scale $S_{\text{ref}} \sim 10^{-2}$ keeps pressure/bed terms comparable to transient inertia.
  - Boundary loss: normalize slope by $\frac{H_0}{L_0}$, velocity by $U_0$, and velocity gradient by $\frac{U_0}{L_0}$ so all boundary terms are strictly dimensionless $[1]$ prior to averaging.
