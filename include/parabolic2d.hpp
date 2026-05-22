#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
