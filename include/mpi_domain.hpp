#pragma once

// =============================================================================
//  include/mpi_domain.hpp
//
//  Optional 2D Cartesian domain-decomposition layer for the MHD runner.
//
//  Design goals (see docs/parallelism.md, "domain decomposition" section):
//
//    * The serial / OpenMP code path is completely unaffected.  When the build
//      is configured without -DENABLE_MPI, this header provides inline no-op
//      stubs and the runner behaves bit-for-bit as before.
//
//    * Every core kernel keeps its existing per-cell arithmetic.  Domain
//      decomposition is added by (a) running the unchanged kernels over a local
//      grid that is padded with `ng` ghost layers, (b) filling those ghost
//      layers from neighbouring ranks with exchange_halos(), and (c) replacing
//      the handful of global reductions (CFL max, min pressure, divB norms)
//      with global_*() helpers.  Interior cells then receive exactly the same
//      stencil inputs they would in a single global periodic grid.
//
//    * All functions take a `const MPIDomain*`.  Passing nullptr (or a domain
//      with active == false) selects the serial behaviour, so the same source
//      compiles and runs whether or not MPI is enabled.
//
//  Index convention matches glm2d_common.hpp: idx2d(i, j, nx) = j * nx + i,
//  with i the fast (contiguous) index.  A decomposed rank stores a padded array
//  of size nx_pad() * ny_pad(); interior cell (i_loc, j_loc) lives at padded
//  index (i_loc + ng, j_loc + ng).
// =============================================================================

#include <vector>

#include "state.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

struct MPIDomain {
#ifdef ENABLE_MPI
    MPI_Comm cart = MPI_COMM_NULL;  // 2D Cartesian communicator, periods = {1,1}
#endif
    int rank = 0;
    int size = 1;

    int dims[2]   = {1, 1};   // process grid {px, py}
    int coords[2] = {0, 0};   // this rank's coordinates in the process grid

    // Neighbour ranks (periodic via the Cartesian topology).  -1 if absent.
    int nbr_left  = -1;
    int nbr_right = -1;
    int nbr_down  = -1;
    int nbr_up    = -1;

    int nx_g = 0;   // global interior cell counts
    int ny_g = 0;

    int nx_loc = 0; // this rank's interior cell counts
    int ny_loc = 0;

    int i0 = 0;     // global index of this rank's first interior cell
    int j0 = 0;

    int ng = 0;     // ghost-layer width (0 in serial)

    // active == true only for a genuinely decomposed run (ENABLE_MPI, size set
    // up through make_domain).  When false every helper below is an identity.
    bool active = false;

    int nx_pad() const { return nx_loc + 2 * ng; }
    int ny_pad() const { return ny_loc + 2 * ng; }
    int npad_cells() const { return nx_pad() * ny_pad(); }
};

#ifdef ENABLE_MPI

// ---- Real MPI implementations (src/mpi_domain.cpp) --------------------------

// Build a 2D Cartesian decomposition of an nx_g x ny_g periodic grid over
// MPI_COMM_WORLD, padding each rank's local block with `ng` ghost layers.
// Must be called after MPI_Init.  nx_g/ny_g must be divisible by the chosen
// process-grid dimensions; throws std::invalid_argument otherwise.
MPIDomain make_domain(int nx_g, int ny_g, int ng);

// Fill the `ng` ghost layers of a padded local State field (size
// d->npad_cells()) from neighbouring ranks.  Two-phase exchange (vertical then
// horizontal over full padded height) so diagonal/corner ghosts are populated.
// No-op when d is null or inactive.
void exchange_halos(std::vector<State>& U_padded, const MPIDomain* d);

// Fill ghost layers of a padded scalar field.  This is the scalar counterpart
// of exchange_halos() and is used by matrix-free distributed stencil operators.
void exchange_scalar_halos(
    std::vector<double>& field_padded,
    const MPIDomain* d
);

// Halo exchange of a padded char mask combined with logical OR on overlap.
// Used by the positivity-limiter LLF re-sweep so shared faces agree across
// ranks.  No-op when d is null or inactive.
void exchange_halo_mask_or(std::vector<char>& mask_padded, const MPIDomain* d);

// Collective reductions over the Cartesian communicator.  Identity when d is
// null or inactive (so serial-style callers pass nullptr and pay nothing).
double    global_max(double local,    const MPIDomain* d);
double    global_min(double local,    const MPIDomain* d);
double    global_sum(double local,    const MPIDomain* d);
long long global_sum(long long local, const MPIDomain* d);
int       global_lor(int local_flag,  const MPIDomain* d);  // logical OR
bool      global_all(bool local_true, const MPIDomain* d);  // logical AND

// Gather every rank's interior block into a single global nx_g x ny_g field on
// the root rank (rank 0).  The returned vector is empty on non-root ranks.
// When d is null or inactive, returns the interior of U_padded unchanged.
std::vector<State> gather_to_root(
    const std::vector<State>& U_padded,
    const MPIDomain* d
);

#else  // ----------------- serial build: inline identity stubs -----------------

inline void exchange_halos(std::vector<State>&, const MPIDomain*) {}
inline void exchange_scalar_halos(std::vector<double>&, const MPIDomain*) {}
inline void exchange_halo_mask_or(std::vector<char>&, const MPIDomain*) {}

inline double    global_max(double x,    const MPIDomain*) { return x; }
inline double    global_min(double x,    const MPIDomain*) { return x; }
inline double    global_sum(double x,    const MPIDomain*) { return x; }
inline long long global_sum(long long x, const MPIDomain*) { return x; }
inline int       global_lor(int x,       const MPIDomain*) { return x; }
inline bool      global_all(bool x,      const MPIDomain*) { return x; }

inline std::vector<State> gather_to_root(
    const std::vector<State>& U_padded,
    const MPIDomain*
) {
    return U_padded;
}

#endif  // ENABLE_MPI
