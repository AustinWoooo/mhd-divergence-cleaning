#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

void update_hyperbolic_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
