#pragma once

#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

CleaningType parse_cleaning_type_2d(const std::string& name);

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
);

void initialize_divergence_pulse_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void advance_glm_2d_one_step(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
);

void write_glm_2d_snapshot(
    const std::vector<State>& U,
    const GLM2DParams& params,
    const std::string& filename
);

void run_glm_2d_case(
    CleaningType type,
    GLM2DParams params
);
