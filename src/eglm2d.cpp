#include "eglm2d.hpp"

#include <stdexcept>
#include <vector>

#include "glm2d_common.hpp"
#include "mixed_glm2d.hpp"

void apply_eglm_source_2d(
    std::vector<State>& U,
    const std::vector<State>& Uref,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;

    const int ncell = nx * ny;

    if (static_cast<int>(U.size()) != ncell ||
        static_cast<int>(Uref.size()) != ncell) {
        throw std::runtime_error(
            "apply_eglm_source_2d: inconsistent grid size"
        );
    }

    const std::vector<double> divB =
        compute_fv_divB_field_2d(Uref, nx, ny, dx, dy);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);

            const int ip = periodic_index(i + 1, nx);
            const int im = periodic_index(i - 1, nx);
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            const int id_ip = idx2d(ip, j, nx);
            const int id_im = idx2d(im, j, nx);
            const int id_jp = idx2d(i, jp, nx);
            const int id_jm = idx2d(i, jm, nx);

            const State& ref = Uref[id];
            State cell = U[id];

            const double Bx = ref[BX];
            const double By = ref[BY];
            const double Bz = ref[BZ];

            const double dpsidx =
                (Uref[id_ip][PSI] - Uref[id_im][PSI]) / (2.0 * dx);

            const double dpsidy =
                (Uref[id_jp][PSI] - Uref[id_jm][PSI]) / (2.0 * dy);

            const double s = -dt * divB[id];

            // EGLM momentum source:
            //
            //   d(rho*u)/dt = -(divB) B
            cell[MX] += s * Bx;
            cell[MY] += s * By;
            cell[MZ] += s * Bz;

            // EGLM energy source:
            //
            //   dE/dt = -B . grad(psi)
            //
            // In this 2D geometry, dpsi/dz = 0.
            cell[E] -= dt * (Bx * dpsidx + By * dpsidy);

            // Important:
            //   rho is unchanged.
            //   Bx, By, Bz are unchanged by the EGLM source.
            //   psi is unchanged by the EGLM source.
            //
            // B and psi are evolved by the mixed GLM update.

            U[id] = cell;
        }
    }
}

void update_mixed_eglm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const std::vector<State> Uold = U;

    update_mixed_glm_2d(U, params);
    apply_eglm_source_2d(U, Uold, params);
}