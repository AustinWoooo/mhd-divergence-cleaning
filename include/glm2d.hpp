#pragma once

#include <string>
#include <vector>

#include "state.hpp"
#include "glm.hpp"
#include "diagnostics.hpp"

struct GLM2DParams {
    int nx = 128;
    int ny = 128;

    double xlen = 1.0;
    double ylen = 1.0;

    double dx = 1.0 / 128.0;
    double dy = 1.0 / 128.0;

    double dt = 1.0e-3;
    double t_end = 0.5;
    double cfl = 0.25;

    double ch = 1.0;
    double cp = 0.2;

    // For Powell-source toy model.
    double powell_vx = 1.0;
    double powell_vy = 0.5;

    // Elliptic projection settings.
    int poisson_max_iter = 10000;
    double poisson_tol = 1.0e-10;

    bool write_snapshot = true;
    std::string out_prefix = "glm_2d";
};

CleaningType parse_cleaning_type_2d(const std::string& name);

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
);

void initialize_divergence_pulse_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void update_hyperbolic_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_powell_source_2d(
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