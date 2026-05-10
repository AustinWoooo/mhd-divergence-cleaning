#pragma once
#include <array>

enum Var {
    RHO = 0,
    MX,
    MY,
    MZ,
    BX,
    BY,
    BZ,
    E,
    PSI,
    NVAR
};

using State = std::array<double, NVAR>;

enum class CleaningType {
    NONE,
    PARABOLIC,
    HYPERBOLIC_GLM,
    MIXED_GLM,
    ELLIPTIC_PROJECTION,
    POWELL_SOURCE,
    MIXED_EGLM
};