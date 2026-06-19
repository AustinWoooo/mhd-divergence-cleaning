#pragma once

#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

CleaningType parse_cleaning_type_2d(const std::string& name);

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
);

double max_cleaning_dt(
    CleaningType method,
    double dx,
    double dy,
    const GLM2DParams& params
);

void advance_glm_2d_one_step(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
);
