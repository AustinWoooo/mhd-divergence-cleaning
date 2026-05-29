# MHD Divergence Cleaning — 2D Test Suite

This repository contains a 2D finite-volume ideal-MHD solver using an HLLD
Riemann solver, together with several modular / operator-split divergence-control
methods for studying numerical violations of

\[
\nabla \cdot \mathbf{B} = 0.
\]

The current integrated runner applies divergence-control methods outside the core
HLLD solver. The HLLD solver is kept as a pure ideal-MHD flux kernel. Some newer
plugin components are currently tested separately and are not yet fully wired into
the production `test_mhd_runner`.

## Numerical scheme

The integrated runner uses a second-order **high-resolution shock-capturing
(HRSC)** scheme built around the HLLD flux:

- **Spatial reconstruction** — piecewise-linear MUSCL reconstruction of the
  primitive variables to the cell faces, with a TVD slope limiter
  (`minmod`, `vanleer`, or `mc`; default `mc`). Because every limiter is TVD,
  reconstructed face density and pressure stay positive whenever the cell
  averages are positive. Falling back to `pcm` recovers the first-order Godunov
  scheme for comparison. See [`include/mhd_reconstruction.hpp`](include/mhd_reconstruction.hpp).
- **Riemann solver** — HLLD (Miyoshi & Kusano 2005), with a Local
  Lax-Friedrichs (LLF) positivity fallback. On faces flagged by the positivity
  limiter the scheme drops to first-order states with the diffusive LLF flux.
- **Time integration** — SSP-RK2 (Heun), with a dual-energy formalism for the
  low-beta cold cores.

---

## Build

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j
```

Run the registered tests with:

```bash
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```

---

## Executables

| Executable | Description |
|---|---|
| `test_hlld_primitive_recovery` | Tests raw vs checked primitive recovery and verifies that raw pressure is not silently floored |
| `test_mhd_reconstruction` | Unit tests for the MUSCL slope limiters: linear exactness, extremum clipping, and bounded (positivity-preserving) face values |
| `test_cleaning_plugins` | Tests the first-stage cleaning plugin interface and GLM flux wrapper |
| `test_mhd_runner` | Integrated 2D HLLD + divergence-control runner |
| `test_glm_2d` | Standalone 2D GLM cleaning test on a divergence pulse |
| `test_glm_1d` | Standalone 1D GLM cleaning test |

---

## `test_mhd_runner` — Usage

```bash
./build/test_mhd_runner [options] [problem] [cleaning...]
```

### Arguments

| Argument | Values | Default |
|---|---|---|
| `problem` | `orszag_tang` \| `brio_wu` \| `field_loop` \| `divergence_advection` | `orszag_tang` and `brio_wu` |
| `cleaning` | see table below | all available methods |
| `--reconstruction` | `pcm` (first-order Godunov) \| `plm` (second-order MUSCL HRSC) | `plm` |
| `--first-order` | alias for `--reconstruction pcm` | — |
| `--limiter` | `minmod` \| `vanleer` \| `mc` (only used for `plm`) | `mc` |

### Cleaning / divergence-control method names

| Name | Method |
|---|---|
| `none` | No cleaning / baseline |
| `parabolic` | Parabolic divergence diffusion |
| `hyperbolic_glm` | Hyperbolic GLM divergence propagation |
| `mixed_glm` | Mixed GLM: hyperbolic propagation plus ψ damping |
| `elliptic_projection` | Elliptic projection using a Poisson solve |
| `powell_source` | Powell / 8-wave non-conservative source control |
| `mixed_eglm` | Mixed GLM with EGLM momentum/energy source terms |
| `gi_mixed_eglm` | Galilean-invariant mixed EGLM source-term variant |

### Notes

- `powell_source` is not a true cleaning method in the same sense as GLM or projection. It advects / controls divergence errors through non-conservative source terms.
- `mixed_eglm` and `gi_mixed_eglm` should be treated as source-term variants and require diagnostics of conservation drift.
- The first-stage plugin interface currently has dedicated tests for the GLM flux wrapper, but the production runner is not yet fully plugin-driven.

---

## Examples

```bash
# Default problem set and all cleaning methods
./build/test_mhd_runner

# One problem, all cleaning methods
./build/test_mhd_runner orszag_tang
./build/test_mhd_runner brio_wu
./build/test_mhd_runner field_loop
./build/test_mhd_runner divergence_advection

# One problem, one cleaning method
./build/test_mhd_runner brio_wu mixed_glm
./build/test_mhd_runner orszag_tang hyperbolic_glm
./build/test_mhd_runner field_loop mixed_glm
./build/test_mhd_runner divergence_advection mixed_glm

# One problem, multiple selected cleaning methods
./build/test_mhd_runner orszag_tang none hyperbolic_glm mixed_glm
./build/test_mhd_runner divergence_advection none hyperbolic_glm mixed_glm

# Show help
./build/test_mhd_runner --help
```

---

## Output

Results are written to `results/mhd_runner/`:

```text
results/mhd_runner/
├── divergence/          # divB norm history, one CSV per run
│   ├── mhd_ot_none.csv
│   ├── mhd_ot_mixed_glm.csv
│   └── ...
├── snapshots/           # final field snapshots, one CSV per run
│   ├── mhd_ot_none_final.csv
│   ├── mhd_ot_mixed_glm_final.csv
│   └── ...
└── summaries/           # run summaries and failure diagnostics, when available
    ├── mhd_ot_parabolic_summary.csv
    └── ...
```

Filename prefixes:

| Problem | Prefix |
|---|---|
| Orszag-Tang vortex | `mhd_ot` |
| Brio-Wu shock tube | `mhd_bw` |
| Field-loop advection | `mhd_fl` |
| Divergence advection | `mhd_da` |

Generated `results/` files are reproducibility artifacts. Do not commit regenerated CSVs unless they are intentionally curated for the final report.

---

## Supported Problems

### Orszag-Tang Vortex

- Smooth, periodic, fully 2D MHD vortex problem
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Useful for comparing robustness and divergence growth in a nonlinear MHD flow

### Brio-Wu Shock Tube 2D Strip

- 1D MHD shock tube replicated as a 2D strip
- Parameters: γ = 2, t_end = 0.2, N = 128
- Useful as a shock-capturing stress test
- Current boundary and strip setup should be interpreted carefully when discussing divergence diagnostics

### Field-Loop Advection

- Weak localized magnetic loop advected across a periodic domain
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Background state: ρ = 1, p = 1, u = 1, v = 1
- Normalized divergence diagnostics should be interpreted carefully where \(|B|\) is close to zero

### Divergence Advection

- Controlled non-solenoidal magnetic perturbation advected by a background flow
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Background state: ρ = 1, p = 1, u = 1, v = 0.5
- Useful for checking whether divergence errors are transported, damped, projected, or diffused