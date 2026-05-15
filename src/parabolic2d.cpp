#include "parabolic2d.hpp"
#include "glm2d_common.hpp"

#include <vector>

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

    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    std::vector<State> U_new = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);

            const int id = idx2d(i, j, nx);

            const double grad_divB_x =
                (divB[idx2d(ip, j, nx)] - divB[id]) / dx;

            const double grad_divB_y =
                (divB[idx2d(i, jp, nx)] - divB[id]) / dy;

            U_new[id][BX] += dt * cp2 * grad_divB_x;
            U_new[id][BY] += dt * cp2 * grad_divB_y;
        }
    }

    U.swap(U_new);
}