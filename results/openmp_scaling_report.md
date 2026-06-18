# OpenMP Scaling Result

OpenMP tests intra-node shared-memory parallelism by running the same divergence_advection / none solver workload on thread counts [1, 2, 4, 8, 16, 32] for grid sizes [128, 256, 512]. The ideal strong-scaling speedup is linear in the number of threads, but real scaling is limited by memory bandwidth, synchronization and barrier overhead, reduction overhead in diagnostics, cache effects, and serial portions of the code. In this run, the best measured speedups were N=128: 3.83x at 8 threads; N=256: 6.34x at 16 threads; N=512: 4.59x at 16 threads. Larger grids usually scale better because each thread receives more stencil and Riemann-solver work relative to fixed OpenMP overhead, while parallel efficiency usually decreases at high thread counts as those overheads and bandwidth limits become more important. The final divergence residuals matched the one-thread baseline within the configured tolerance.

Figures:
- `figures/openmp_scaling/openmp_runtime_vs_threads.png`
- `figures/openmp_scaling/openmp_speedup_vs_threads.png`
- `figures/openmp_scaling/openmp_efficiency_vs_threads.png`
- `figures/openmp_scaling/openmp_time_per_iteration_vs_threads.png`
