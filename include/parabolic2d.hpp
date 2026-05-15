#pragma once

#include "glm2d.hpp"

#include <vector>

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
