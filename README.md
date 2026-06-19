# MHD Divergence Cleaning - 2D Final Project

This repository is a 2D finite-volume ideal-MHD project for studying numerical
violations of

```text
div B = 0
```

It implements and compares divergence-control methods on full 2D MHD test
problems.  The solver does **not** use constrained transport.  The production
path is an operator-split MHD runner:

```text
HLLD/LLF hydro update -> divergence cleaning -> diagnostics -> CSV -> figures
```

The main scientific comparisons are no cleaning, hyperbolic/mixed GLM, mixed
EGLM, GI mixed EGLM, parabolic cleaning, elliptic projection, and Powell-type
source cleaning.  The base MHD update uses HLLD fluxes, a positivity fallback to
LLF, MUSCL/PLM reconstruction, TVD limiters, and RK2 time integration.

## Architecture

Core state and primitive variables:

- `include/state.hpp` defines the conserved state layout:
  `rho, mx, my, mz, Bx, By, Bz, E, psi`.
- `include/HLLD_mhd_solver.hpp` declares primitive recovery and the HLLD/LLF
  flux interface.
- `src/hlld_solver.cpp` implements primitive/conserved conversion, HLLD, and
  LLF fluxes.

Hydro update and HRSC reconstruction:

- `include/mhd_reconstruction.hpp` implements PCM/PLM reconstruction and the
  `minmod`, `vanleer`, and `mc` TVD slope limiters.
- `src/mhd_runner.cpp` contains the full 2D MHD runner: CFL calculation,
  RK2 HLLD update, LLF positivity fallback, dual-energy recovery, cleaning
  dispatch, diagnostics, snapshots, summaries, and performance timing.
- `include/mhd_runner.hpp` exposes the runner parameters and problem
  initialization functions.

Divergence cleaning:

- `src/glm2d.cpp` dispatches `CleaningType` values.
- `src/hyperbolic_glm2d.cpp` implements hyperbolic GLM.
- `src/mixed_glm2d.cpp` implements mixed GLM damping.
- `src/eglm2d.cpp` and `src/galilean_glm2d.cpp` implement EGLM variants.
- `src/parabolic2d.cpp` implements explicit parabolic cleaning.
- `src/projection2d.cpp` implements periodic elliptic projection.
- `src/powell2d.cpp` implements the basic Powell source method.

Diagnostics and scripts:

- `include/glm2d_common.hpp` and `src/glm2d_common.cpp` provide the finite-volume
  `divB` diagnostic used by the production runner.
- `scripts/plot/plot_mhd_runner.py` plots standard MHD comparisons.
- `scripts/run/run_field_loop_convergence.py` and
  `scripts/run/run_divergence_advection_convergence.py` run accuracy evidence
  cases.
- `scripts/run/run_performance_scaling.py` runs the serial wall-clock scaling
  study.
- `scripts/check/check_performance_scaling.py` validates benchmark CSVs and
  figures.

## Build And Test

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Optional MPI support is disabled by default.  Enabling it builds two
independent things: the coarse-grained `mhd_sweep_mpi` (one whole-grid case per
rank) and `mhd_runner_mpi` (a single simulation split across ranks by 2D domain
decomposition).  Without `ENABLE_MPI` the runner is bit-for-bit the serial /
OpenMP build:

```bash
cmake -S . -B build-mpi -DENABLE_MPI=ON
cmake --build build-mpi -j
```

## Executables

`mhd_runner_cli` is the user-facing full MHD runner:

```bash
./build/mhd_runner_cli [options] [problem] [cleaning...]
```

`test_mhd_runner` remains the registered regression/smoke test driver and
accepts the same benchmark-oriented options.

Useful discovery commands:

```bash
./build/mhd_runner_cli --list-problems
./build/mhd_runner_cli --list-methods
./build/mhd_runner_cli --help
```

Supported problems:

- `orszag_tang`
- `field_loop`
- `divergence_advection`
- `blast_wave`

Supported cleaning names:

- `none`
- `hyperbolic_glm`
- `mixed_glm`
- `parabolic`
- `elliptic_projection`
- `powell_source`
- `eglm` / `mixed_eglm`
- `gi_mixed_eglm`

## Running Science Cases

Default Orszag-Tang comparison with all methods:

```bash
./build/mhd_runner_cli orszag_tang
```

Field loop with no cleaning and mixed GLM:

```bash
./build/mhd_runner_cli field_loop none mixed_glm
```

Divergence-advection comparison:

```bash
./build/mhd_runner_cli divergence_advection none mixed_glm mixed_eglm
```

MHD blast wave (strong shock in a magnetized, low-beta background):

```bash
./build/mhd_runner_cli blast_wave none parabolic elliptic_projection
```

The blast sets a high-pressure circular core (`P=10`, `R=0.1`) in a uniform,
strongly magnetized ambient medium (`rho=1`, `P=0.1`, `B=(1,1,0)/sqrt(2)`, plasma
`beta=0.2`).  The fast shock expands anisotropically under magnetic tension.
Periodic boundaries are used and the default `t_end=0.2` keeps the shock interior.
On this low-beta state the GLM-family methods do not complete with the default
conserve-total-energy policy, so `none`, `parabolic`, and `elliptic_projection`
are the methods that run to completion.

Run a smaller explicit case:

```bash
./build/mhd_runner_cli --nx 64 --ny 64 --tfinal 0.1 \
  --output-prefix mhd_da_n64 divergence_advection none mixed_glm
```

Select reconstruction:

```bash
./build/mhd_runner_cli --reconstruction pcm field_loop mixed_glm
./build/mhd_runner_cli --reconstruction plm --limiter mc field_loop mixed_glm
```

## Output Structure

Default C++ outputs are rooted at `results/mhd_runner/`:

- `results/mhd_runner/divergence/`
  Per-step time histories.  Columns include step, time, dt, FV divergence norms,
  normalized divergence norms, mass, momentum, total energy, minimum pressure,
  failure flags, cleaning subcycles, projection iterations, and projection
  residuals.
- `results/mhd_runner/snapshots/`
  Final field snapshots with `i,j,x,y,rho,u,v,w,p,Bx,By,Bz,psi,divB_fv,Bmag`.
- `results/mhd_runner/summaries/`
  One compact per-run summary CSV.  These now include final physics diagnostics
  and performance timing columns.
- `results/mhd_runner/failures/`
  Bad-state and first-bad-cell diagnostic CSVs, when a run triggers them.
- `results/mhd_runner/performance/`
  Merged benchmark tables from `scripts/run/run_performance_scaling.py`.

Figures are written under `figures/mhd_runner/`.

Use `--output-root PATH` to redirect runner CSV output:

```bash
./build/mhd_runner_cli --output-root /tmp/mhd_outputs divergence_advection mixed_glm
```

## Benchmark Controls

The runner has benchmark-friendly options:

- `--no-snapshots`
  disables final snapshot writing.
- `--diagnostic-stride N`
  writes diagnostic rows every `N` accepted steps while always preserving step 0,
  final step, and failure rows.
- `--performance-mode`
  disables snapshots and uses sparse diagnostics by default.
- `--output-root PATH`
  redirects CSV outputs away from the default result tree.

These controls do not change the numerical update.  They only reduce benchmark
I/O and organize outputs.

## Performance Scaling

The assignment requires wall-clock scaling versus number of cells.  Use a short
smoke benchmark only to check the pipeline quickly:

```bash
python3 scripts/run/run_performance_scaling.py \
  --resolutions 32 64 \
  --methods none mixed_glm \
  --output-csv results/mhd_runner/performance/performance_scaling_smoke.csv \
  --skip-build
```

The core report benchmark keeps the main scaling table and figures focused on
the baseline and the primary GLM/projection methods:

```bash
python3 scripts/run/run_performance_scaling.py \
  --resolutions 32 64 128 256 \
  --methods none mixed_glm mixed_eglm elliptic_projection \
  --reconstruction plm \
  --limiter vanleer \
  --skip-build
```

Core benchmark:

- methods: `none mixed_glm mixed_eglm elliptic_projection`
- CSV: `results/mhd_runner/performance/performance_scaling.csv`
- figures: `figures/mhd_runner/performance/performance_*.png` without a
  method-set suffix

The all-problem, all-method benchmark uses the wrapper script to run every
supported cleaning method except the duplicate `eglm` alias on
`divergence_advection`, `field_loop`, and `orszag_tang`.  It writes separate
per-problem CSVs, merges them into one report CSV, and uses `_all_methods`
figure suffixes so it does not overwrite the core benchmark:

```bash
python3 scripts/run/run_all_problem_performance_scaling.py \
  --resolutions 32 64 128 256 \
  --methods none parabolic hyperbolic_glm mixed_glm mixed_eglm gi_mixed_eglm \
    elliptic_projection powell_source \
  --reconstruction plm \
  --limiter vanleer \
  --skip-build
```

All-method benchmark:

- methods: all supported cleaning methods except the duplicate `eglm` alias for
  `mixed_eglm`
- CSV: `results/mhd_runner/performance/performance_scaling_all_methods.csv`
- per-problem CSVs:
  `results/mhd_runner/performance/performance_scaling_all_methods_<problem>.csv`
- figures: `figures/mhd_runner/performance/performance_*_all_methods.png`
- per-resolution breakdown figures:
  `figures/mhd_runner/performance/performance_method_breakdown_all_methods_*.png`
- normalized per-resolution breakdown figures:
  `figures/mhd_runner/performance/performance_method_breakdown_normalized_all_methods_*.png`
- use: comprehensive method comparison; the combined plots can be crowded, so
  the fixed-resolution breakdown plots are usually easier to read

Defaults:

- problem: `divergence_advection`
- resolutions: `32 64 128 256`
- methods: `none mixed_glm mixed_eglm elliptic_projection`
- reconstruction: `plm`
- limiter: `mc` unless overridden; the report command above uses `vanleer`
- final time: `0.05`
- diagnostic stride: `100`

The script builds `mhd_runner_cli` unless `--skip-build` is passed, runs each
case in performance mode, reads the generated per-case summary CSVs, and writes
the full report benchmark to:

```text
results/mhd_runner/performance/performance_scaling.csv
```

Smoke runs should use a separate output path such as:

```text
results/mhd_runner/performance/performance_scaling_smoke.csv
```

Benchmark figures:

```text
figures/mhd_runner/performance/performance_walltime_vs_cells.png
figures/mhd_runner/performance/performance_cell_updates_per_second.png
figures/mhd_runner/performance/performance_seconds_per_step.png
figures/mhd_runner/performance/performance_method_breakdown.png
figures/mhd_runner/performance/performance_method_breakdown_32.png
figures/mhd_runner/performance/performance_method_breakdown_64.png
figures/mhd_runner/performance/performance_method_breakdown_128.png
figures/mhd_runner/performance/performance_method_breakdown_256.png
figures/mhd_runner/performance/performance_method_breakdown_normalized.png
figures/mhd_runner/performance/performance_method_breakdown_normalized_32.png
figures/mhd_runner/performance/performance_method_breakdown_normalized_64.png
figures/mhd_runner/performance/performance_method_breakdown_normalized_128.png
figures/mhd_runner/performance/performance_method_breakdown_normalized_256.png
figures/mhd_runner/performance/performance_cleaning_overhead_fraction.png
```

The overall method-breakdown plot shows every benchmark run together.  The
per-resolution breakdown plots show the same hydro, cleaning, and
diagnostics/output timing stack at a fixed grid size, which is cleaner for
method-to-method comparisons.  Absolute breakdown plots show real wall-clock
cost in seconds.  Normalized breakdown plots show the fraction of measured time
spent in hydro, cleaning, and diagnostics/output, which is useful when one
method, such as elliptic projection, dominates the absolute-time scale.

Validate the benchmark output with:

```bash
python3 scripts/check/check_performance_scaling.py \
  results/mhd_runner/performance/performance_scaling.csv
```

Performance CSV schema:

```text
problem,method,reconstruction,limiter,glm_ch_factor,glm_cd,glm_cr,
glm_subcycles,glm_ch,glm_cp,glm_effective_cd,glm_effective_cr,
nx,ny,ncell,tfinal,steps,status,
total_wall_time_sec,initialization_time_sec,hydro_time_sec,cleaning_time_sec,
diagnostics_compute_time_sec,diagnostics_write_time_sec,snapshot_write_time_sec,
summary_write_time_sec,output_time_sec,total_cell_updates,seconds_per_step,
cell_updates_per_second,cleaning_subcycles_total,projection_iterations_total,
final_L2_fv,final_Linf_fv,final_L2_norm_fv,final_Linf_norm_fv,
peak_L2_norm_fv,peak_Linf_norm_fv,time_integrated_L2_norm_fv,
min_pressure,min_density,has_nonfinite,has_negative_density,
has_negative_pressure,energy_drift,failure_reason,summary_file,diagnostic_file
```

`cell_updates_per_second` is computed as:

```text
nx * ny * steps / total_wall_time_sec
```

Timing categories are measured with `std::chrono::steady_clock`.  The hydro
category includes the RK2 HLLD/LLF finite-volume update.  The cleaning category
measures the end-of-step cleaning update.  If `--project-each-stage` is used,
the optional predictor-stage projection occurs inside the RK2 function and is
therefore counted with hydro time; the default benchmark workflow does not use
stage projection.

Snapshots are disabled during performance runs because full-field snapshot I/O
can dominate wall time and obscure solver scaling.  Sparse diagnostics preserve
the initial/final correctness checks while avoiding dense CSV output.

The basic `powell_source` method is useful as a non-conservative source-term
comparison, but it is not forced into the main report scaling plot unless it
remains stable for the selected problem and resolution range.

## Parallel Strong Scaling (OpenMP and MPI)

The section above measures *serial* wall-clock versus number of cells.  The two
benchmarks here instead measure **strong scaling**: a fixed-size problem is run
on an increasing amount of parallelism, and the speedup over the single-worker
baseline is reported.  Both scripts repeat each run (`--repeats`, median by
default), take `solver_time = hydro_time_sec + cleaning_time_sec` from the runner
summary, and check correctness by comparing each run's final divergence residual
(`final_L2_fv`) against the single-worker baseline at the same grid size.  Each
emits an aggregated CSV, a raw per-repeat CSV, four figures (runtime, speedup
with an ideal-linear reference, parallel efficiency, time-per-iteration), and a
short Markdown report.

Both benchmarks use `divergence_advection` with `--method none` as the default
reference workload, so the reported speedup is for the pure ideal-MHD hydro
update (RK2 HLLD/LLF) with **no divergence cleaning**.  The speedup baseline is
the single-worker run of that same problem and method at each grid size.  Change
the workload with `--problem` and `--method`; picking a cleaning method also
exercises the cleaning step, which (under MPI) adds communication — see the MPI
note below.

### OpenMP (shared memory)

Varies `OMP_NUM_THREADS` on a single `mhd_runner_cli` process (the default
`build/` build already links OpenMP):

```bash
python3 scripts/benchmark_openmp_scaling.py \
  --sizes 128 256 512 --threads 1 2 4 8 --repeats 3 \
  --problem divergence_advection --method none
```

Outputs:

```text
results/openmp_scaling.csv          # aggregated (per size x threads)
results/openmp_scaling_raw.csv      # per-repeat
results/openmp_scaling_report.md
figures/openmp_scaling/openmp_{runtime,speedup,efficiency,time_per_iteration}_vs_threads.{png,pdf}
```

Thread counts above the available CPUs are skipped automatically.

### MPI (distributed memory, domain decomposition)

Varies the number of MPI ranks: the same simulation is split across a 2D
Cartesian domain decomposition via `mhd_runner_mpi`.  This requires the MPI
build, which the script configures on first use into a separate `build-mpi/`
directory (`-DENABLE_MPI=ON`).  Each rank runs with `OMP_NUM_THREADS=1` by
default so the measurement is pure MPI:

```bash
python3 scripts/benchmark_mpi_scaling.py \
  --sizes 256 512 1024 --ranks 1 2 4 8 --repeats 3 \
  --problem divergence_advection --method none
```

Outputs:

```text
results/mpi_scaling.csv             # aggregated (per size x ranks)
results/mpi_scaling_raw.csv         # per-repeat
results/mpi_scaling_report.md
figures/mpi_scaling/mpi_{runtime,speedup,efficiency,time_per_iteration}_vs_ranks.{png,pdf}
```

Because `solver_time` includes the per-substep halo exchanges, the MPI numbers
account for communication as well as compute.  The global `nx`/`ny` must be
divisible by the process-grid dimensions `MPI_Dims_create` chooses for a given
rank count (powers of two are safe); incompatible size/rank combinations are
skipped with a warning.  Use `--oversubscribe` to run more ranks than physical
cores (for functional testing; it distorts timing), `--threads-per-rank N` for a
hybrid MPI+OpenMP measurement, and `--mpirun` / `--mpirun-extra` to choose the
launcher or pass binding flags.  Strong-scaling benefit is clearest on large
grids (512² and up); small grids are communication-bound and scale sub-linearly.

Because the default `--method none` measures only the hydro update, its scaling
does **not** include any divergence-cleaning communication.  Selecting a cleaning
method adds that cost: the GLM / EGLM / parabolic / Powell methods do one extra
halo exchange per cleaning substep, while `elliptic_projection` solves a global
Poisson equation every step with a matrix-free distributed conjugate gradient
(scalar halo exchanges plus a collective `MPI_Allreduce` for the dot products and
convergence norm on every CG iteration).  The projection's communication pattern
is therefore heavier and typically scales worse than the hydro-only or
hyperbolic-cleaning methods.  To benchmark cleaning-inclusive scaling, pass e.g.
`--problem orszag_tang --method mixed_glm` or `--method elliptic_projection`.

The validated reference run on this machine (`divergence_advection`, `none`,
medians of 3 repeats, ranks 1/2/4/8) gave speedups of 1.00 / 1.86 / 3.61 / 4.04
at 256² (parallel efficiency 1.00 / 0.93 / 0.90 / 0.51), with larger grids
scaling better, and the final divergence residual bit-identical to the
single-rank baseline at every rank count.

## Python Figure Workflow

Standard science figures:

```bash
./build/mhd_runner_cli
./build/mhd_runner_cli field_loop none hyperbolic_glm mixed_glm mixed_eglm gi_mixed_eglm
./build/mhd_runner_cli divergence_advection none hyperbolic_glm mixed_glm mixed_eglm gi_mixed_eglm
./build/mhd_runner_cli blast_wave none parabolic elliptic_projection
python3 scripts/plot/plot_mhd_runner.py
```

Accuracy and HRSC evidence:

```bash
python3 scripts/run/run_field_loop_convergence.py
python3 scripts/run/run_divergence_advection_convergence.py
```

The convergence scripts are present, and existing generated outputs live under
`results/mhd_runner/convergence/` and `figures/mhd_runner/convergence/`.  In the
current tree these scripts set their repository root relative to `scripts/`,
so check that path handling before rerunning them.

Additional cleaning/pressure diagnostics:

```bash
python3 scripts/plot/plot_cleaning_diagnostics.py
python3 scripts/plot/plot_pressure_diagnostics.py
```

## Optional MPI Sweep

`mhd_sweep_mpi` is optional coarse-grained task parallelism over independent
runner cases.  It is not domain decomposition and is not the primary serial
wall-clock scaling measurement.

```bash
mpirun -np 4 ./build-mpi/mhd_sweep_mpi \
  --problem divergence_advection \
  --nx 32 --ny 32 --t-end 0.02 \
  --methods none,mixed_glm,eglm,gi_mixed_eglm \
  --prefix mpi_da
```

Rank-local summaries are written to:

```text
results/mhd_sweep_mpi/summaries/summary_rank_*.csv
```

Merge them with:

```bash
python3 scripts/run/merge_mpi_sweep_summaries.py
```

## Optional MPI Domain Decomposition

`mhd_runner_mpi` splits a **single** simulation across ranks with a 2D Cartesian
domain decomposition: each rank evolves its own ghost-padded sub-block and
exchanges halos every substep, CFL/positivity reductions go through
`MPI_Allreduce`, and rank 0 gathers the sub-blocks for all CSV/snapshot output
(formats unchanged, so the plotting/check scripts are untouched).  This is the
layer that makes a single large run faster; it is distinct from the
coarse-grained sweep above.

```bash
mpirun -np 4 ./build-mpi/mhd_runner_mpi \
  --nx 256 --ny 256 --tfinal 0.5 \
  orszag_tang mixed_glm
```

The global `nx`/`ny` must be divisible by the process-grid dimensions (chosen by
`MPI_Dims_create`).  Running with a single rank (`-np 1`) reproduces the serial
result.  Multi-rank `elliptic_projection` uses a matrix-free conjugate-gradient
solve of the periodic Poisson equation, with scalar halo exchanges and global
reductions for dot products and convergence norms.  See
[`docs/parallelism.md`](docs/parallelism.md) for the full design.

## Assignment Checklist

- 2D MHD divergence cleaning implemented: yes, through the integrated
  `CleaningType` and `glm2d` runner path.
- Divergence errors compared with and without correction: yes, through
  `none` versus GLM/EGLM/projection/Powell-type methods.
- No constrained transport: yes.  The code uses divergence cleaning/projection
  methods, not CT.
- Applied to MHD test problems: yes, Orszag-Tang, field loop, divergence
  advection, and the MHD blast wave.
- Performance scaling measured: yes, via
  `scripts/run/run_performance_scaling.py`.
- HRSC bonus: yes, HLLD fluxes with PLM/MUSCL reconstruction, TVD limiters,
  RK2, and LLF positivity fallback.
