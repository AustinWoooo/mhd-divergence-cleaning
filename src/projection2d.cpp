#include "projection2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "glm2d_common.hpp"
#include "mpi_domain.hpp"

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

double distributed_interior_sum(
    const std::vector<double>& a,
    const MPIDomain* domain
) {
    const int ng = domain->ng;
    const int nx_pad = domain->nx_pad();
    double local = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : local)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            local += a[idx2d(i, j, nx_pad)];
        }
    }
    return global_sum(local, domain);
}

double distributed_dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const MPIDomain* domain
) {
    const int ng = domain->ng;
    const int nx_pad = domain->nx_pad();
    double local = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : local)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            const int id = idx2d(i, j, nx_pad);
            local += a[id] * b[id];
        }
    }
    return global_sum(local, domain);
}

void subtract_distributed_mean(
    std::vector<double>& a,
    const MPIDomain* domain
) {
    const double ncell = static_cast<double>(domain->nx_g) * domain->ny_g;
    const double mean = distributed_interior_sum(a, domain) / ncell;
    const int ng = domain->ng;
    const int nx_pad = domain->nx_pad();
#pragma omp parallel for schedule(static)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            a[idx2d(i, j, nx_pad)] -= mean;
        }
    }
}

// Apply the positive-semidefinite operator A = -L on the local interior.
// CG remains in the mean-zero subspace, where periodic -L is positive definite.
void apply_distributed_negative_laplacian(
    std::vector<double>& x,
    std::vector<double>& Ax,
    const GLM2DParams& params,
    const MPIDomain* domain
) {
    exchange_scalar_halos(x, domain);

    const int ng = domain->ng;
    const int nx_pad = domain->nx_pad();
    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);
#pragma omp parallel for schedule(static)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            const int id = idx2d(i, j, nx_pad);
            const double lap =
                (x[id + 1] - 2.0 * x[id] + x[id - 1]) * inv_dx2
              + (x[id + nx_pad] - 2.0 * x[id] + x[id - nx_pad]) * inv_dy2;
            Ax[id] = -lap;
        }
    }
}

PoissonSolveResult solve_periodic_poisson_cg_distributed(
    const std::vector<State>& U,
    const GLM2DParams& params,
    const MPIDomain* domain
) {
    const int ng = domain->ng;
    const int nx_pad = domain->nx_pad();
    const int npad = domain->npad_cells();
    const double ncell = static_cast<double>(domain->nx_g) * domain->ny_g;

    std::vector<double> rhs(npad, 0.0);
    std::vector<double> phi(npad, 0.0);
    std::vector<double> residual(npad, 0.0);
    std::vector<double> direction(npad, 0.0);
    std::vector<double> Adirection(npad, 0.0);

    // The state halos must be current before entry.  Form div_fv(B) only on
    // owned cells, then solve (-L) phi = -div_fv(B).
#pragma omp parallel for schedule(static)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            const int id = idx2d(i, j, nx_pad);
            rhs[id] =
                (U[id][BX] - U[id - 1][BX]) / params.dx
              + (U[id][BY] - U[id - nx_pad][BY]) / params.dy;
        }
    }
    subtract_distributed_mean(rhs, domain);

#pragma omp parallel for schedule(static)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            const int id = idx2d(i, j, nx_pad);
            residual[id] = -rhs[id];
            direction[id] = residual[id];
        }
    }

    ProjectionResult info{};
    info.converged = false;
    double rr = distributed_dot(residual, residual, domain);
    double residual_rms = std::sqrt(rr / ncell);
    if (residual_rms <= params.poisson_tol) {
        info.converged = true;
    }

    for (int iter = 0; iter < params.poisson_max_iter && !info.converged; ++iter) {
        apply_distributed_negative_laplacian(
            direction, Adirection, params, domain);
        const double pAp = distributed_dot(direction, Adirection, domain);

        // pAp is collective, so all ranks take the same failure path.
        if (!(pAp > 0.0) || !std::isfinite(pAp) || !std::isfinite(rr)) {
            break;
        }

        const double alpha = rr / pAp;
#pragma omp parallel for schedule(static)
        for (int j = ng; j < ng + domain->ny_loc; ++j) {
            for (int i = ng; i < ng + domain->nx_loc; ++i) {
                const int id = idx2d(i, j, nx_pad);
                phi[id] += alpha * direction[id];
                residual[id] -= alpha * Adirection[id];
            }
        }

        const double rr_new = distributed_dot(residual, residual, domain);
        residual_rms = std::sqrt(rr_new / ncell);
        info.iterations = iter + 1;
        info.solver_update_residual = residual_rms;

        if (!std::isfinite(residual_rms)) {
            rr = rr_new;
            break;
        }
        if (residual_rms <= params.poisson_tol) {
            rr = rr_new;
            info.converged = true;
            break;
        }

        const double beta = rr_new / rr;
#pragma omp parallel for schedule(static)
        for (int j = ng; j < ng + domain->ny_loc; ++j) {
            for (int i = ng; i < ng + domain->nx_loc; ++i) {
                const int id = idx2d(i, j, nx_pad);
                direction[id] = residual[id] + beta * direction[id];
            }
        }
        rr = rr_new;
    }

    subtract_distributed_mean(phi, domain);

    // Recompute the true residual after the nullspace projection.  Since
    // residual = L(phi) - rhs for A=-L and b=-rhs, this directly matches the
    // serial residual convention.
    std::vector<double> Aphi(npad, 0.0);
    apply_distributed_negative_laplacian(phi, Aphi, params, domain);
    double local_sum_sq = 0.0;
    double local_max_abs = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : local_sum_sq) reduction(max : local_max_abs)
    for (int j = ng; j < ng + domain->ny_loc; ++j) {
        for (int i = ng; i < ng + domain->nx_loc; ++i) {
            const int id = idx2d(i, j, nx_pad);
            const double true_residual = -Aphi[id] - rhs[id];
            local_sum_sq += true_residual * true_residual;
            local_max_abs = std::max(local_max_abs, std::abs(true_residual));
        }
    }
    info.true_residual_L2 =
        std::sqrt(global_sum(local_sum_sq, domain) / ncell);
    info.true_residual_Linf = global_max(local_max_abs, domain);
    info.final_residual = info.true_residual_Linf;
    info.solver_update_residual = info.true_residual_L2;
    info.converged = info.converged
        && std::isfinite(info.true_residual_L2)
        && info.true_residual_L2 <= params.poisson_tol;

    // apply_distributed_negative_laplacian refreshed phi's ghost cells.
    return {phi, info};
}

} // namespace

// =============================================================================
//  Public interface
// =============================================================================

ProjectionResult solve_projection_phi_2d(
    const std::vector<State>& U,
    const GLM2DParams& params,
    std::vector<double>& phi_out,
    const MPIDomain* domain
) {
    if (domain && domain->active) {
        const PoissonSolveResult solve =
            solve_periodic_poisson_cg_distributed(U, params, domain);
        if (!solve.info.converged) {
            // The distributed solve has completed all halo exchanges and
            // collective residual reductions before reaching this branch.
            // Consequently every rank observes the same failure and exits the
            // projection path before applying an unconverged B correction.
            std::ostringstream message;
            message << "MPI elliptic projection CG failed to converge: "
                    << "global_residual_rms="
                    << solve.info.true_residual_L2
                    << ", tolerance=" << params.poisson_tol
                    << ", iterations=" << solve.info.iterations
                    << ", max_iterations=" << params.poisson_max_iter;
            throw std::runtime_error(message.str());
        }
        phi_out = solve.phi;
        return solve.info;
    }

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
    double theta,
    const MPIDomain* domain
) {
    if (domain && domain->active) {
        const int ng = domain->ng;
        const int nx_pad = domain->nx_pad();
#pragma omp parallel for schedule(static)
        for (int j = ng; j < ng + domain->ny_loc; ++j) {
            for (int i = ng; i < ng + domain->nx_loc; ++i) {
                const int id = idx2d(i, j, nx_pad);
                U[id][BX] -= theta * (phi[id + 1] - phi[id]) / params.dx;
                U[id][BY] -=
                    theta * (phi[id + nx_pad] - phi[id]) / params.dy;
                U[id][PSI] = 0.0;
            }
        }
        exchange_halos(U, domain);
        return;
    }

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
    const GLM2DParams& params,
    const MPIDomain* domain
) {
    std::vector<double> phi;
    const ProjectionResult info =
        solve_projection_phi_2d(U, params, phi, domain);

    apply_projection_B_correction_2d(U, phi, params, 1.0, domain);

    // Full projection: theta = 1.
    ProjectionResult result = info;
    result.projection_theta = 1.0;
    return result;
}
