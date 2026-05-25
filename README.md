# MHD Divergence Cleaning — 2D Test Suite

Integrated 2D ideal-MHD solver (HLLD Riemann solver) coupled with various
**∇·B = 0** divergence-cleaning methods.

---

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

---

## Executables

| Executable | Description |
|---|---|
| `test_mhd_runner` | HLLD + GLM runner: Orszag-Tang, Brio-Wu, field-loop, and divergence-advection problems |
| `test_glm_2d` | Standalone 2D GLM cleaning on a divergence pulse |
| `test_glm_1d` | Standalone 1D GLM cleaning test |

---

## `test_mhd_runner` — Usage

```
./build/test_mhd_runner [problem] [cleaning...]
```

### Arguments

| Argument | Values | Default |
|---|---|---|
| `problem` | `orszag_tang` \| `brio_wu` \| `field_loop` \| `divergence_advection` | `orszag_tang` and `brio_wu` |
| `cleaning` | see table below | all |

### Cleaning method names

| Name | Method |
|---|---|
| `none` | No cleaning |
| `parabolic` | Parabolic cleaning |
| `hyperbolic_glm` | Hyperbolic GLM |
| `mixed_glm` | Mixed GLM (hyperbolic + parabolic damping) |
| `elliptic_projection` | Elliptic projection (Poisson solve) |
| `powell_source` | Powell 8-wave source terms |
| `mixed_eglm` | Mixed GLM + EGLM momentum/energy sources |
| `gi_mixed_eglm` | Galilean-invariant mixed EGLM (Dedner et al.) |

### Examples

```bash
# All problems × all cleaning methods (original behaviour)
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
./build/test_mhd_runner orszag_tang none mixed_glm hyperbolic_glm
./build/test_mhd_runner divergence_advection none hyperbolic_glm mixed_glm

# Show help
./build/test_mhd_runner --help
```

---

## Output

Results are written to `results/mhd_runner/`:

```
results/mhd_runner/
├── divergence/          # divB norm history (one .csv per run)
│   ├── mhd_ot_none.csv
│   ├── mhd_ot_mixed_glm.csv
│   └── ...
└── snapshots/           # Final field snapshot (one .csv per run)
    ├── mhd_ot_none_final.csv
    ├── mhd_ot_mixed_glm_final.csv
    └── ...
```

Filename prefixes:

| Problem | Prefix |
|---|---|
| Orszag-Tang vortex | `mhd_ot` |
| Brio-Wu shock tube | `mhd_bw` |
| Field-loop advection | `mhd_fl` |
| Divergence advection | `mhd_da` |

---

## Supported Problems

### Orszag-Tang Vortex
- Smooth, periodic, fully 2D
- Parameters: γ = 5/3, t_end = 0.5, N = 128

### Brio-Wu Shock Tube (2D strip)
- 1D shock replicated as a 2D strip in the y-direction
- Parameters: γ = 2, t_end = 0.2, N = 128

### Field-Loop Advection
- Weak localized magnetic loop advected across a periodic domain
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Background state: ρ = 1, p = 1, u = 1, v = 1

### Divergence Advection
- Controlled non-solenoidal magnetic perturbation advected by a background flow
- Parameters: γ = 5/3, t_end = 0.5, N = 128
- Background state: ρ = 1, p = 1, u = 1, v = 0.5
