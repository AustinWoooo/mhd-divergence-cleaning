# MHD Divergence Cleaning — 2D Test Suite

This repository contains a 2D finite-volume ideal-MHD solver using an HLLD
Riemann solver, together with several divergence-control methods for studying
numerical violations of

\[
\nabla \cdot \mathbf{B} = 0.
\]

The current integrated runner applies the divergence-control update outside the
core HLLD flux kernel. The HLLD solver is kept as a pure ideal-MHD flux kernel.
The GLM-family updates are therefore operator-split from the ideal-MHD flux
update in the integrated runner.

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
  The HLLD kernel also guards near-degenerate star-state denominators with
  relative-tolerance checks; there is not currently a separate nonfinite
  star-state retry inside `MHD::compute_flux`.
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
| `problem` | `orszag_tang` \| `field_loop` \| `divergence_advection` | `orszag_tang`|
| `cleaning` | see table below | all available methods |
| `--reconstruction` | `pcm` (first-order Godunov) \| `plm` (second-order MUSCL HRSC) | `plm` |
| `--first-order` | alias for `--reconstruction pcm` | — |
| `--limiter` | `minmod` \| `vanleer` \| `mc` (only used for `plm`) | `mc` |

### Cleaning / divergence-control method names

Main divergence-control methods:

| Name | Equation / correction | Conservation / splitting | Divergence-error behavior | Expected advantage | Report limitation |
|---|---|---|---|---|---|
| `none` | No cleaning; ideal-MHD HLLD update only | Conservative finite-volume update, aside from documented positivity safeguards | Lets numerical divB evolve unchecked | Baseline for every comparison | Not a cleaning method |
| `hyperbolic_glm` | Dedner B-psi subsystem: `dB/dt + grad(psi)=0`, `dpsi/dt + ch^2 divB=0` | Operator-split; psi energy is not included in conserved energy | Transports divB as waves at cleaning speed `ch` | Local finite-speed correction | Split from HLLD flux in the runner |
| `mixed_glm` | Hyperbolic GLM plus `psi <- psi exp(-dt ch^2/cp^2)` | Operator-split; psi energy is not included in conserved energy | Transports and damps divB | Usually more robust than pure hyperbolic GLM | Damping is applied as a split scalar decay |
| `parabolic` | `dB/dt = cp^2 grad(divB)` using the FV divB/gradient pair | Operator-split diffusion update; energy policy is reported in summaries | Diffuses divB | Simple local smoothing of grid-scale divergence | Explicit diffusion is time-step limited |
| `elliptic_projection` | Periodic Poisson solve `lap(phi)=divB`, then `B <- B - grad(phi)` | Operator-split projection; energy policy and projection relaxation are reported | Projects B toward a discrete divergence-free field | Strongest direct divB reduction | Global solve; relaxed projection is not exact when `theta < 1` |
| `powell_source` | `dU/dt = -(divB)[0,Bx,By,Bz,ux,uy,uz,u.B,0]^T` | Nonconservative source update | Transports divB errors with the flow rather than directly eliminating them | Can advect divergence out of local structures | Conservation drift and pressure robustness must be reported |
| `eglm` / `mixed_eglm` | Extended GLM: mixed GLM plus `d(rho u)/dt=-(divB)B`, `dE/dt=-B.grad(psi)` | Operator-split, nonconservative extended-GLM source update | Transports/damps divB and applies EGLM source corrections | Formal GLM-family extension coupling divB to momentum/energy | Intended continuous EGLM convention needs author confirmation |
| `gi_mixed_eglm` | GI-EGLM: mixed GLM plus `d(rho u)/dt=-(divB)B`, `dB/dt=-(divB)u`, `dE/dt=-(divB)(u.B)-B.grad(psi)`, `dpsi/dt=-u.grad(psi)` | Operator-split, nonconservative extended-GLM source update | Transports/damps divB and applies Galilean-invariant source corrections | Targets frame-invariant extended-GLM behavior | Frame invariance is not proven for the current split update |

Cautionary / robustness-control variants:

| Name | Method | Intended use |
|---|---|---|
| `powell_source_subcycled` | Powell source update with source-CFL subcycling | Robustness diagnostic for the explicit Powell source split |
| `powell_source_limited` | Pressure-limited Powell source update | Positivity-limited nonconservative robustness policy; not exact Powell |

### Notes

- `eglm` is accepted as a CLI alias for the existing public method name `mixed_eglm`; generated output files still use `mixed_eglm` to avoid renaming existing artifacts.
- `hyperbolic_glm`, `mixed_glm`, `mixed_eglm`, and `gi_mixed_eglm` are implemented as operator-split cleaning/source updates in the integrated runner. The scalar psi field is evolved for cleaning, but psi energy is not included in the conserved total energy.
- `powell_source` uses nonconservative source terms. It transports divergence errors with the flow rather than eliminating them, so conservation drift and pressure robustness must be diagnosed separately from divB reduction.
- `mixed_eglm` is treated here as a formal extended-GLM method. The implemented source terms are
  `d(rho u)/dt = -(div B) B` and `dE/dt = -B . grad(psi)`, applied after the mixed-GLM B-psi update.
- `gi_mixed_eglm` is treated here as a formal Galilean-invariant extended-GLM method. The implemented source terms are
  `d(rho u)/dt = -(div B) B`, `dB/dt = -(div B) u`, `dE/dt = -(div B)(u.B) - B.grad(psi)`, and `dpsi/dt = -u.grad(psi)`, applied after the mixed-GLM update.
- The EGLM/GI-EGLM source terms use the conserved-variable ordering `[rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, psi]`; the code updates the matching momentum, magnetic-field, energy, and psi slots consistently with that ordering.
- Because GI-EGLM is applied as a split source using the pre-cleaning reference state, the desired Galilean-invariant continuous formulation needs author confirmation at the report level. The current implementation documents the intended source terms but does not prove exact frame invariance of the full operator-split hydro plus cleaning step.
- The first-stage plugin interface currently has dedicated tests for the GLM flux wrapper, but the production runner is not yet fully plugin-driven.

---

## Examples

```bash
# Default problem set and all cleaning methods
./build/test_mhd_runner

# One problem, all cleaning methods
./build/test_mhd_runner orszag_tang
./build/test_mhd_runner field_loop
./build/test_mhd_runner divergence_advection

# One problem, one cleaning method
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

## Optional MPI Sweep Runner

MPI support is optional and disabled by default. Enable it only when you want to
run coarse-grained parameter sweeps over independent numerical experiments:

```bash
cmake -S . -B build-mpi -DENABLE_MPI=ON
cmake --build build-mpi -j
mpirun -np 2 ./build-mpi/mhd_sweep_mpi --smoke
```

`mhd_sweep_mpi` does **not** domain-decompose the MHD grid. Each MPI rank runs
complete `mhd_runner` cases assigned by job index, so there are no MPI ghost
cells and no MPI calls inside the hydro update, HLLD flux, or cleaning equations.
The default job list is intentionally small and includes `none`,
`hyperbolic_glm`, `mixed_glm`, `parabolic`, `elliptic_projection`,
`powell_source`, `eglm`, and `gi_mixed_eglm`.

Useful options:

```bash
mpirun -np 4 ./build-mpi/mhd_sweep_mpi \
  --problem divergence_advection \
  --nx 32 --ny 32 --t-end 0.02 --cfl 0.4 \
  --methods none,mixed_glm,eglm,gi_mixed_eglm \
  --prefix mpi_da
```

Rank-local sweep summaries are written to
`results/mhd_sweep_mpi/summaries/summary_rank_*.csv`. Individual solver outputs
still go through `results/mhd_runner/` with unique prefixes, avoiding concurrent
writes to the same result file. Merge rank summaries with:

```bash
python scripts/merge_mpi_sweep_summaries.py
```

Full domain-decomposed MPI is future work. In particular, local finite-volume
updates would need ghost-cell exchange, while `elliptic_projection` would need a
parallel/global Poisson solve or a different projection strategy.

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
| Field-loop advection | `mhd_fl` |
| Divergence advection | `mhd_da` |

Generated `results/` files are reproducibility artifacts. Do not commit regenerated
CSVs unless they are intentionally curated for the final report. Use finite-volume
normalized divergence diagnostics (`L1_norm_fv`, `L2_norm_fv`, `Linf_norm_fv`) as
the main comparison metrics. Centered-difference divB diagnostics are retained as
secondary/internal checks for the standalone cleaning demos.

---

## Supported Problems

### Orszag-Tang Vortex

- Smooth, periodic, fully 2D MHD vortex problem
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Useful for comparing robustness and divergence growth in a nonlinear MHD flow

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
