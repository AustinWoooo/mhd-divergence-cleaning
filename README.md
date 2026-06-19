# mhd-divergence-cleaning

Final archived code and results for a Computational Astrophysics project on
divergence control in two-dimensional ideal MHD.

This repository studies how a finite-volume ideal-MHD solver behaves when it
does not use constrained transport and instead relies on divergence-cleaning or
divergence-control operators to manage numerical `∇·B` error. The project
compares robustness, divergence suppression, energy behavior, and runtime cost
across multiple cleaning families, and includes both OpenMP and MPI execution
paths.

## Project status

This is the final project state on `main`, not an in-progress development
branch. The repository already contains:

- the final solver implementation,
- serial/OpenMP and MPI runners,
- curated result CSVs and figures,
- committed OpenMP, MPI, and hybrid scaling workflows.

## Scientific scope

The code evolves the 2D ideal-MHD equations in conservative form and tracks
finite-volume magnetic-divergence diagnostics during time integration.

Supported cleaning / divergence-control methods exposed by the CLI are:

- `none`: baseline MHD update with no additional cleaning
- `powell_source`: Powell eight-wave source terms
- `hyperbolic_glm`: Dedner-style hyperbolic GLM cleaning
- `mixed_glm`: hyperbolic GLM plus damping
- `parabolic`: parabolic / diffusion-style cleaning
- `elliptic_projection`: Poisson projection of the magnetic field
- `eglm`: energy-coupled GLM variant
- `mixed_eglm`: damped EGLM variant
- `gi_mixed_eglm`: Galilean-invariant mixed EGLM variant

The solver does not claim that every method is equally robust on every problem.
Tracked failure outputs under `results/mhd_runner/failures/` are kept in the
repository on purpose.

## Supported test problems

The current runner advertises these problems:

- `divergence_advection`
- `field_loop`
- `orszag_tang`
- `blast_wave`

These are the main problems used throughout the committed results and figures.

## Repository layout

- `src/`, `include/`: solver, cleaning operators, diagnostics, and MPI domain
  decomposition
- `scripts/`: benchmark, plotting, validation, and result-processing helpers
- `results/`: curated CSV outputs, performance summaries, and runner summaries
- `figures/`: curated report figures, performance plots, and visual inventories
- `CMakeLists.txt`: build targets and tests

Important build targets:

- `mhd_runner_cli`: serial/OpenMP runner
- `mhd_runner_mpi`: MPI runner
- `mhd_sweep_mpi`: MPI sweep helper
- `test_hlld_primitive_recovery`
- `test_mhd_runner`

## Dependencies

Requirements visible from the committed build system and scripts:

- CMake 3.16 or newer
- a C++17 compiler
- OpenMP
- Python 3
- Python packages used by the scripts:
  - `matplotlib`
  - `numpy`
  - `pandas`
- for MPI builds: an MPI implementation with C++ support, such as OpenMPI or
  MPICH

## Build

### Serial / OpenMP build

```bash
cmake -S . -B build
cmake --build build --parallel
```

### MPI build

```bash
cmake -S . -B build-mpi -DENABLE_MPI=ON
cmake --build build-mpi --parallel
```

## Discover supported problems and methods

```bash
./build/mhd_runner_cli --help
./build/mhd_runner_cli --list-problems
./build/mhd_runner_cli --list-methods
```

At the time of this final README, the CLI reports:

- problems: `orszag_tang`, `field_loop`, `divergence_advection`, `blast_wave`
- methods: `none`, `hyperbolic_glm`, `mixed_glm`, `parabolic`,
  `elliptic_projection`, `powell_source`, `eglm`, `mixed_eglm`,
  `gi_mixed_eglm`

## Example runs

### Serial / OpenMP

Orszag-Tang with mixed GLM:

```bash
./build/mhd_runner_cli orszag_tang mixed_glm
```

Blast-wave smoke run without snapshots:

```bash
./build/mhd_runner_cli --smoke --no-snapshots blast_wave none
```

Elliptic projection with thermal-pressure preservation:

```bash
./build/mhd_runner_cli --preserve-thermal-pressure \
  orszag_tang elliptic_projection
```

Performance-mode comparison on a fixed grid:

```bash
./build/mhd_runner_cli --performance-mode --nx 256 --ny 256 \
  orszag_tang none mixed_glm elliptic_projection
```

### MPI

Two-rank smoke run:

```bash
mpirun -np 2 ./build-mpi/mhd_runner_mpi --smoke \
  orszag_tang none mixed_glm elliptic_projection
```

Projection-focused MPI run:

```bash
mpirun -np 4 ./build-mpi/mhd_runner_mpi --performance-mode \
  orszag_tang elliptic_projection
```

The MPI runner includes domain decomposition and an MPI-compatible
`elliptic_projection` path using a distributed matrix-free conjugate-gradient
Poisson solve.

## Smoke tests and light validation

Fast serial checks:

```bash
ctest --test-dir build -R 'hlld|glm_1d' --output-on-failure
```

MPI smoke example:

```bash
mpirun -np 2 ./build-mpi/mhd_runner_mpi --smoke \
  orszag_tang none mixed_glm elliptic_projection
```

Runner output sanity check:

```bash
python3 scripts/check/check_mhd_runner_sanity.py \
  --results-dir results/mhd_runner \
  --prefix mhd_ot_smoke \
  --methods none hyperbolic_glm mixed_glm parabolic elliptic_projection \
            powell_source mixed_eglm gi_mixed_eglm \
  --expected-nx 32 \
  --expected-ny 32
```

## Output locations

### Solver outputs

Default runner outputs are written under `results/mhd_runner/`, including:

- `divergence/`: per-step time-history diagnostics
- `summaries/`: one-line summary CSVs
- `snapshots/`: final-state snapshots
- `failures/`: captured failure diagnostics for unsuccessful runs
- `convergence/`: convergence-study CSVs
- `performance/`: single-node performance summaries
- `glm_sweep/`: GLM parameter-sweep outputs

### Curated figures

- `figures/report_clean/`: final report figure set and figure index
- `figures/mhd_runner/`: scientific diagnostics and comparison plots
- `figures/performance/`: unified OpenMP/MPI/hybrid scaling figures
- `figures/openmp_scaling/`, `figures/mpi_scaling/`: legacy scaling outputs

### Committed performance summaries

- `results/performance/openmp_mpi_scaling.csv`
- `results/performance/openmp_mpi_scaling_report.md`
- `results/performance/openmp_mpi_scaling_256.csv`
- `results/performance/openmp_mpi_scaling_256_report.md`
- `results/performance/openmp_scaling_256_32.csv`
- `results/performance/openmp_scaling_256_32_report.md`

Raw benchmark inventories are intentionally not version-controlled. `.gitignore`
excludes transient benchmark run directories, raw CSVs, and logs while keeping
curated aggregate outputs and figures trackable.

## Performance benchmark workflow

The repository contains both legacy scripts and a unified scaling workflow.

### Unified OpenMP / MPI / hybrid workflow

Primary script:

```bash
python3 scripts/benchmark_parallel_scaling.py --help
```

Tiny smoke benchmark:

```bash
python3 scripts/benchmark_parallel_scaling.py --smoke
```

Example method-separated study:

```bash
python3 scripts/benchmark_parallel_scaling.py \
  --openmp --mpi --hybrid \
  --methods none mixed_glm elliptic_projection \
  --grids 128 256 \
  --threads 1 2 4 8 \
  --ranks 1 2 4 \
  --hybrid-configs 1x8 2x4 4x2 8x1 \
  --repeats 3
```

The unified workflow records:

- method
- grid size
- MPI ranks
- OpenMP threads per rank
- total cores
- wall time and solver time
- speedup and parallel efficiency
- cell updates per second
- final divergence norms
- energy drift
- projection iteration and convergence metadata when applicable

Output files are written to `results/performance/` and
`figures/performance/`.

### Other benchmark and plotting scripts

- `scripts/benchmark_openmp_scaling.py`
- `scripts/benchmark_mpi_scaling.py`
- `scripts/run/run_performance_scaling.py`
- `scripts/plot/plot_performance_scaling_diagnostics.py`

## Final results / summary

This repository contains the final curated outputs for both the scientific and
performance parts of the project.

At a high level, the archived project includes:

- a common MHD runner with multiple divergence-control methods,
- MPI domain decomposition,
- MPI support for `elliptic_projection`,
- committed OpenMP-only, MPI-only, and hybrid scaling workflows,
- curated scientific and performance figures already checked into the repo.

The results are mixed rather than uniformly favorable to every method. The
tracked failure summaries and failure CSVs show that some cleaning families are
less robust than others on demanding low-pressure or blast-wave problems, and
the repository keeps those negative results visible instead of hiding them.

Useful entry points for the archived report material:

- `figures/report_clean/REPORT_CLEAN_INDEX.md`
- `results/performance/openmp_mpi_scaling_report.md`
- `results/performance/openmp_mpi_scaling_256_report.md`

## Reproducing curated figures

Useful plotting entry points:

```bash
python3 scripts/plot/plot_mhd_runner.py
python3 scripts/plot/plot_report_clean_figures.py
python3 scripts/plot/plot_performance_scaling_diagnostics.py
```

These scripts regenerate curated figure products from committed CSV inputs and
do not require rerunning the largest benchmark campaigns.

## Final note

This is the final archived version of the Computational Astrophysics project.
It is intended to be readable, buildable, and reproducible from a fresh clone,
but it is no longer organized as an active development branch.
