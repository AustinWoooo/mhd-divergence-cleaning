#include "parabolic2d.hpp"

#include <vector>

#include "glm2d_common.hpp"

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;
    const double cp2 = params.cp * params.cp;

    // Use finite-volume-compatible divB and forward gradient so that
    // divB evolves approximately by:
    //
    //   d(divB)/dt = cp^2 Laplacian(divB).
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);

            const int id = idx2d(i, j, nx);

            const double dDiv_dx =
                (divB[idx2d(ip, j,  nx)] - divB[id]) / dx;

            const double dDiv_dy =
                (divB[idx2d(i,  jp, nx)] - divB[id]) / dy;

            State cell = Uold[id];

            // Parabolic cleaning:
            //
            //   dB/dt = cp^2 grad(divB).
            //
            // Then divB satisfies a diffusion equation.
            cell[BX] += dt * cp2 * dDiv_dx;
            cell[BY] += dt * cp2 * dDiv_dy;

            U[id] = cell;
        }
    }
}
