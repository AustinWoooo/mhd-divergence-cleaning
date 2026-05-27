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
};

// Elliptic projection for the standalone 2D divergence-cleaning framework.
ProjectionResult apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
