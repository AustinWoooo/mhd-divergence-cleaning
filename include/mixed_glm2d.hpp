#pragma once

#include "glm2d.hpp"

#include <vector>

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void update_mixed_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);