# MPI Scaling Result

MPI tests distributed-memory parallelism by splitting the same divergence_advection / none simulation across a 2D Cartesian domain decomposition for rank counts [1, 2, 4] and grid sizes [64]. The ideal strong-scaling speedup is linear in the number of ranks, but real scaling is limited by halo-exchange communication (which grows with the per-rank boundary while compute grows with the per-rank area), the latency of the collective MPI_Allreduce used for the CFL and positivity checks every step, load imbalance, and the serial gather-to-root I/O. In this run, the best measured speedups were N=64: 2.72x at 4 ranks. Larger grids usually scale better because each rank owns more interior cells relative to its fixed-width halo, raising the computation-to-communication ratio, while parallel efficiency usually drops at high rank counts as surface-to-volume and collective-latency costs grow. The final divergence residuals matched the one-rank baseline within the configured tolerance, confirming the decomposition reproduces the serial solution.

Figures:
- `figures/mpi_scaling/mpi_runtime_vs_ranks.png`
- `figures/mpi_scaling/mpi_speedup_vs_ranks.png`
- `figures/mpi_scaling/mpi_efficiency_vs_ranks.png`
- `figures/mpi_scaling/mpi_time_per_iteration_vs_ranks.png`
