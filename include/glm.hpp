#pragma once
#include <vector>
#include <string>
#include "state.hpp"

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

GLMFlux compute_hyperbolic_glm_flux(
    double BnL,
    double psiL,
    double BnR,
    double psiR,
    double ch
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