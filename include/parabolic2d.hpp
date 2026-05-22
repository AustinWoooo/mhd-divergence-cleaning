#pragma once

#include <vector>

#include "glm2d.hpp"

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
