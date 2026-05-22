#include "powell2d.hpp"

#include <vector>

#include "glm2d_common.hpp"

void apply_powell_source_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;

    // Standalone induction-only Powell toy model:
    //
    //   dB/dt = -u divB.
    //
    // This only demonstrates advection of divergence errors. It is NOT the full
    // Powell 8-wave MHD system. The full system also modifies momentum and energy:
    //
    //   S_Powell = -(divB) [0, Bx, By, Bz, u.B, ux, uy, uz]^T
    //
    // for the full conservative MHD state.
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);

            State cell = Uold[id];

            cell[BX] -= dt * params.powell_vx * divB[id];
            cell[BY] -= dt * params.powell_vy * divB[id];

            U[id] = cell;
        }
    }
}
