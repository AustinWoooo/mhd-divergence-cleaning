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

struct PoissonSolveResult {
    std::vector<double> phi;
    ProjectionResult info;
};

ProjectionResult compute_true_poisson_residual(
    const std::vector<double>& phi,
    const std::vector<double>& rhs,
    ProjectionResult info,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
    const double ncell = static_cast<double>(nx * ny);

    double sum_sq = 0.0;
    double max_abs = 0.0;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);
        const int jm = periodic_index(j - 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            const int im = periodic_index(i - 1, nx);
            const int id = idx2d(i, j, nx);

            const double lap =
                (phi[idx2d(ip, j, nx)] - 2.0 * phi[id] + phi[idx2d(im, j, nx)])
                    * inv_dx2
              + (phi[idx2d(i, jp, nx)] - 2.0 * phi[id] + phi[idx2d(i, jm, nx)])
                    * inv_dy2;

            const double residual = lap - rhs[id];
            sum_sq += residual * residual;
            max_abs = std::max(max_abs, std::abs(residual));
        }
    }

    info.true_residual_L2 = std::sqrt(sum_sq / ncell);
    info.true_residual_Linf = max_abs;
    info.final_residual = max_abs;
    return info;
}

PoissonSolveResult solve_periodic_poisson_sor_5pt(
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

    ProjectionResult info{};
    info.converged = false;

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

        info.iterations = iter + 1;
        info.solver_update_residual = max_update;

        if (max_update < params.poisson_tol) {
            info.converged = true;
            break;
        }
    }

    subtract_mean(phi);

    if (params.poisson_max_iter == 0) {
        info.iterations = 0;
        info.solver_update_residual = max_update;
        info.converged = false;
    }

    info = compute_true_poisson_residual(phi, rhs, info, params);
    return {phi, info};
}

} // namespace

// =============================================================================
//  Public interface
// =============================================================================

ProjectionResult solve_projection_phi_2d(
    const std::vector<State>& U,
    const GLM2DParams& params,
    std::vector<double>& phi_out
) {
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;

    std::vector<double> rhs =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    // Periodic Poisson equation is solvable only for zero-mean RHS.
    subtract_mean(rhs);

    const PoissonSolveResult solve =
        solve_periodic_poisson_sor_5pt(rhs, params);

    phi_out = solve.phi;
    return solve.info;
}

void apply_projection_B_correction_2d(
    std::vector<State>& U,
    const std::vector<double>& phi,
    const GLM2DParams& params,
    double theta
) {
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            const int id = idx2d(i, j, nx);

            const double dphi_dx =
                (phi[idx2d(ip, j, nx)] - phi[id]) / dx;

            const double dphi_dy =
                (phi[idx2d(i, jp, nx)] - phi[id]) / dy;

            // Compatible projection:
            //   B <- B - theta * grad_fv(phi).
            U[id][BX] -= theta * dphi_dx;
            U[id][BY] -= theta * dphi_dy;

            U[id][PSI] = 0.0;
        }
    }
}

ProjectionResult apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    std::vector<double> phi;
    const ProjectionResult info = solve_projection_phi_2d(U, params, phi);

    apply_projection_B_correction_2d(U, phi, params, 1.0);

    // Full projection: theta = 1.
    ProjectionResult result = info;
    result.projection_theta = 1.0;
    return result;
}
