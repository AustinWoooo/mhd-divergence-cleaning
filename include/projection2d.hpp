#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

struct ProjectionResult {
    int iterations = 0;
    double solver_update_residual = 0.0;
    double final_residual = 0.0;
    double true_residual_L2 = 0.0;
    double true_residual_Linf = 0.0;
    bool converged = true;
    // Task E: theta actually used for the B correction.
    // 1.0 = full projection; < 1.0 = relaxed (conserve_total_energy mode only).
    // If theta < 1.0 the projection is NOT exact; divB is reduced by theta.
    double projection_theta = 1.0;
};

// Full projection: solve Poisson for phi, then apply B -= grad_fv(phi).
// Equivalent to solve_projection_phi_2d + apply_projection_B_correction_2d(theta=1).
ProjectionResult apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

// Phase 1: solve the periodic Poisson equation
//   Laplacian_5pt(phi) = div_fv(B)
// and return phi in phi_out.  The returned ProjectionResult contains solver
// convergence info; projection_theta is not set here (set by the caller).
ProjectionResult solve_projection_phi_2d(
    const std::vector<State>& U,
    const GLM2DParams& params,
    std::vector<double>& phi_out
);

// Phase 2: apply the B correction  B <- B - theta * grad_fv(phi); zero PSI.
// Compatible with the discrete div_fv/grad_fv pair used by the Poisson solver.
void apply_projection_B_correction_2d(
    std::vector<State>& U,
    const std::vector<double>& phi,
    const GLM2DParams& params,
    double theta
);
