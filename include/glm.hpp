#pragma once

#include <string>
#include <vector>

#include "state.hpp"

enum class Direction {
    X,
    Y,
    Z
};

struct GLMParams {
    CleaningType type = CleaningType::MIXED_GLM;

    double ch = 1.0;  // hyperbolic cleaning speed
    double cp = 0.1;  // damping parameter for mixed GLM / parabolic cleaning

    double dx = 1.0;
    double dt = 1e-3;
};

struct GLMFlux {
    double FBn;
    double Fpsi;
};

std::string cleaning_name(CleaningType type);

int normal_B_index(Direction dir);

GLMFlux compute_hyperbolic_glm_flux(
    double BnL,
    double psiL,
    double BnR,
    double psiR,
    double ch
);

void add_glm_flux_correction(
    State& flux,
    const State& UL,
    const State& UR,
    Direction dir,
    const GLMParams& params
);

void update_hyperbolic_glm_1d(
    std::vector<State>& U,
    const GLMParams& params
);

void apply_mixed_glm_damping(
    std::vector<State>& U,
    const GLMParams& params
);

void apply_parabolic_cleaning_1d(
    std::vector<State>& U,
    const GLMParams& params
);

double compute_divB_1d(
    const std::vector<State>& U,
    int i,
    double dx
);

void compute_divB_norms_1d(
    const std::vector<State>& U,
    double dx,
    double& L1,
    double& L2,
    double& Linf
);