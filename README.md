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
| `test_mhd_runner` | HLLD + GLM runner：Orszag-Tang & Brio-Wu problems |
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
| `problem` | `orszag_tang` \| `brio_wu` | both |
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

# One problem, one cleaning method
./build/test_mhd_runner brio_wu mixed_glm
./build/test_mhd_runner orszag_tang hyperbolic_glm

# One problem, multiple selected cleaning methods
./build/test_mhd_runner orszag_tang none mixed_glm hyperbolic_glm

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

Filename prefix: `mhd_ot` for Orszag-Tang, `mhd_bw` for Brio-Wu.

---

## Supported Problems

### Orszag-Tang Vortex
- Smooth, periodic, fully 2D
- Parameters: γ = 5/3, t_end = 0.5, N = 128

### Brio-Wu Shock Tube (2D strip)
- 1D shock replicated as a 2D strip in the y-direction
- Parameters: γ = 2, t_end = 0.2, N = 128
