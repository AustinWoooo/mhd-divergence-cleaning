#include "powell2d.hpp"

#include <stdexcept>
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

    // Full Powell 8-wave source step for 2.5D ideal MHD.
    //
    // State layout:
    //
    //   U = [rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, psi]
    //
    // Powell source:
    //
    //   dU/dt = -(divB) [0, Bx, By, Bz, ux, uy, uz, u.B, 0]^T
    //
    // Note:
    //   - psi is unchanged.
    //   - This source is non-conservative.
    //   - It transports divergence errors with the local fluid velocity.
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);

            State cell = Uold[id];

            const double rho = cell[RHO];

            if (rho <= TINY_NUMBER) {
                throw std::runtime_error(
                    "apply_powell_source_2d: non-positive density encountered"
                );
            }

            const double ux = cell[MX] / rho;
            const double uy = cell[MY] / rho;
            const double uz = cell[MZ] / rho;

            const double Bx = cell[BX];
            const double By = cell[BY];
            const double Bz = cell[BZ];

            const double u_dot_B = ux * Bx + uy * By + uz * Bz;

            const double s = -dt * divB[id];

            // Momentum source:
            //
            //   d(rho*u)/dt = -(divB) B
            cell[MX] += s * Bx;
            cell[MY] += s * By;
            cell[MZ] += s * Bz;

            // Magnetic-field source:
            //
            //   dB/dt = -(divB) u
            cell[BX] += s * ux;
            cell[BY] += s * uy;
            cell[BZ] += s * uz;

            // Energy source:
            //
            //   dE/dt = -(divB) (u.B)
            cell[E] += s * u_dot_B;

            // GLM scalar is not part of Powell source.
            // cell[PSI] unchanged.

            U[id] = cell;
        }
    }
}