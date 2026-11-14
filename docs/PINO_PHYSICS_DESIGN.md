# PINO Physics-Loss Design Document — Crestix Probe

## 1. Design Principle

Simply using a Fourier Neural Operator (`FNO3d`) does **not** make a model physics-informed. Physics-informed behavior in `PINOFloodModel` comes from the combination of:
1. **Hard architectural constraints** (non-negativity and invalid/building DEM cell masking), and
2. **Soft PDE / conservation / boundary / initial-condition residuals** in `PhysicsLoss` (`src/crestix_pino/losses/physics_loss.py`), evaluated strictly in SI units.

---

## 2. Coordinate System, Units, and Grid Metadata

All physics terms operate in SI units defined by `GridMetadata` and `ShallowWaterConfig`:
- Water depth $h(t, y, x)$: $\text{m}$
- Bed elevation $z_b(y, x)$ (`dem`): $\text{m}$
- Rainfall rate $R(t, y, x)$: $\text{m/s}$ (converted from `mm/h` via $\div 3.6 \times 10^6$, or from `mm/5min` via $\div 300{,}000$)
- Grid spacing $\Delta x, \Delta y$: $\text{m}$ (native LarNO UKEA: $8.0\text{ m}$)
- Time step $\Delta t$: $\text{s}$ (native LarNO UKEA: $300.0\text{ s}$)
- Manning roughness $n$: $\text{s}\cdot\text{m}^{-1/3}$ (default $0.035$)
- Gravitational acceleration $g = 9.81\text{ m/s}^2$

---

## 3. Implemented Physical Constraints & Equations

### 3.1 Invalid / Building DEM Cell Masking
In LarNO UKEA, building footprints and nodata cells are encoded as `NaN` in `dem.npy`.
- `valid_mask = torch.isfinite(dem)` ($\Omega_{\text{valid}}$).
- Output depths and latent velocities are masked:
  $$h = \mathbb{1}_{\Omega_{\text{valid}}} \odot \phi(h_{\text{raw}})$$
  where $\phi$ is `softplus(beta=5.0)` or `relu`.
- Spatial PDE residuals in `PhysicsLoss` use an eroded interior mask $\Omega_{\text{PDE}}$ (cells whose $3\times 3$ neighborhood is entirely valid) so artificial `wall_height` imputation steps at building edges do not corrupt finite-difference bed slopes $\nabla z_b$.

### 3.2 Non-Negative Water Depth
$$\mathcal{L}_{\text{nonneg}} = \frac{1}{|\Omega_{\text{valid}}|} \sum_{(t,y,x) \in \Omega_{\text{valid}}} \max(0, -h_{\text{raw}}(t,y,x))^2$$

### 3.3 Initial-Condition Consistency
$$\mathcal{L}_{\text{IC}} = \frac{1}{|\Omega_{\text{valid}}|} \sum_{(y,x) \in \Omega_{\text{valid}}} \left(h(0, y, x) - h_0(y, x)\right)^2$$

### 3.4 Global Mass-Conservation / Water-Balance Residual
In `src/crestix_pino/physics/conservation.py`, total domain water volume $V(t) = \sum_{y,x} h(t, y, x)\,\Delta x\,\Delta y$ satisfies:
$$\frac{V(t+1) - V(t)}{\Delta t} - \sum_{y,x} \left(R(t, y, x) - I(t, y, x)\right)\Delta x\,\Delta y + \Phi_{\text{boundary}}(t) = 0$$
where $\Phi_{\text{boundary}}$ accounts for open-boundary outflow flux when `boundary_conditions` are marked `"open"`.

### 3.5 2D Shallow-Water PDE Residuals (Latent Velocity Formulation)
When `predict_latent_velocity=True`, `PINOFloodModel` predicts depth $h$ and auxiliary latent velocities $(u, v)$:
- **Continuity ($\text{m/s}$):**
  $$r_{\text{mass}} = \frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} - (R - I)$$
- **Momentum ($\text{m}^2/\text{s}^2$):**
  $$r_{\text{mom}, x} = \frac{\partial (hu)}{\partial t} + \frac{\partial (hu^2 + \frac{1}{2}gh^2)}{\partial x} + \frac{\partial (huv)}{\partial y} + gh\left(\frac{\partial z_b}{\partial x} + S_{fx}\right)$$
  $$r_{\text{mom}, y} = \frac{\partial (hv)}{\partial t} + \frac{\partial (huv)}{\partial x} + \frac{\partial (hv^2 + \frac{1}{2}gh^2)}{\partial y} + gh\left(\frac{\partial z_b}{\partial y} + S_{fy}\right)$$
  with Manning friction slope $S_{fx} = \mathbb{1}_{h > \epsilon}\frac{n^2 u \sqrt{u^2 + v^2}}{\max(h, \epsilon)^{4/3}}$ (and analogously for $S_{fy}$), and momentum residuals masked on dry cells ($h \le \epsilon$).

---

## 4. Explicit Assumptions & Limitations

1. **No Ground-Truth Velocity Supervision:** LarNO UKEA provides `dem`, `rainfall`, and `h` only — **not** $(u, v)$ velocity fields. Therefore, $(u, v)$ are **unsupervised latent variables** regularized solely by SWE residuals and boundary conditions. **We do not claim validated physical velocity accuracy.**
2. **Finite-Difference Discretization Error:** Temporal and spatial derivatives use second-order central differences in the interior and one-sided differences at edges; sharp wet-dry fronts or shocks may exhibit discretization error at $\Delta x = 8\text{ m}, \Delta t = 300\text{ s}$.
3. **Closed vs. Open Boundary Simplification:** Default boundary configuration (`north/east/west=closed`, `south=open`) is an approximation of the cropped $32\times 32$ UKEA sub-region; lateral inflows/outflows across cropped interior boundaries are not explicitly prescribed from the parent $50\times 120$ domain.
4. **Softplus Floor:** When `nonnegative_depth_mode="softplus"`, `softplus(0, beta=5) ≈ 0.1386 m` unless `h_raw` is driven negative on dry cells; `valid_mask` zeroes building cells, while dry terrain cells rely on data + IC loss to push `h_raw` negative.
