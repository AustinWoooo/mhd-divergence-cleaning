#include "projection2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "glm2d_common.hpp"

namespace {

void subtract_mean(std::vector<double>& a) {
    if (a.empty()) return;

    double mean = 0.0;
    for (double x : a) {
        mean += x;
    }

    mean /= static_cast<double>(a.size());

    for (double& x : a) {
        x -= mean;
    }
}

std::vector<double> solve_periodic_poisson_sor_5pt(
    const std::vector<double>& rhs_input,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (inv_dx2 + inv_dy2);

    std::vector<double> rhs = rhs_input;
    std::vector<double> phi(nx * ny, 0.0);

    // Periodic Poisson equation requires zero-mean RHS.
    subtract_mean(rhs);

    double max_update = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < params.poisson_max_iter; ++iter) {
        max_update = 0.0;

        for (int j = 0; j < ny; ++j) {
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            for (int i = 0; i < nx; ++i) {
                const int ip = periodic_index(i + 1, nx);
                const int im = periodic_index(i - 1, nx);

                const int id = idx2d(i, j, nx);

                const double jacobi_value =
                    (
                        (phi[idx2d(ip, j, nx)] + phi[idx2d(im, j, nx)]) * inv_dx2
                      + (phi[idx2d(i, jp, nx)] + phi[idx2d(i, jm, nx)]) * inv_dy2
                      - rhs[id]
                    ) / denom;

                const double old_value = phi[id];

                const double new_value =
                    (1.0 - params.poisson_omega) * old_value
                  + params.poisson_omega * jacobi_value;

                phi[id] = new_value;

                max_update = std::max(
                    max_update,
                    std::abs(new_value - old_value)
                );
            }
        }

        // Remove the arbitrary constant mode from time to time.
        if (iter % 50 == 0) {
            subtract_mean(phi);
        }

        if (max_update < params.poisson_tol) {
            break;
        }
    }

    subtract_mean(phi);

    return phi;
}

} // namespace

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;

    // Compatible RHS:
    //
    //   Laplacian_5pt(phi) = div_fv(B).
    std::vector<double> rhs =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    // Periodic Poisson equation is solvable only for zero-mean RHS.
    subtract_mean(rhs);

    const std::vector<double> phi =
        solve_periodic_poisson_sor_5pt(rhs, params);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);

            const int id = idx2d(i, j, nx);

            const double dphi_dx =
                (phi[idx2d(ip, j, nx)] - phi[id]) / dx;

            const double dphi_dy =
                (phi[idx2d(i, jp, nx)] - phi[id]) / dy;

            State cell = Uold[id];

            // Compatible projection:
            //
            //   B <- B - grad_fv(phi).
            //
            // With the div_fv operator above:
            //
            //   div_fv(B_new) = div_fv(B_old) - Laplacian_5pt(phi).
            cell[BX] -= dphi_dx;
            cell[BY] -= dphi_dy;

            cell[PSI] = 0.0;

            U[id] = cell;
        }
    }
}
