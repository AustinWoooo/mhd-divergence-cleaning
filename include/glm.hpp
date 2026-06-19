#pragma once

#include <string>
#include <vector>

#include "state.hpp"

enum class Direction {
    X,
    Y,
    Z
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
