#pragma once

#include <vector>

#include "glm2d.hpp"

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void update_mixed_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
