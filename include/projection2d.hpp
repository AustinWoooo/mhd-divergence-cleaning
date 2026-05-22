#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Elliptic projection for the standalone 2D divergence-cleaning framework.
void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
