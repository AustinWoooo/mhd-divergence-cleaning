#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Standalone induction-only Powell-like source used by the divergence-cleaning
// toy problem. This is not the full Powell 8-wave MHD source system.
void apply_powell_source_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
