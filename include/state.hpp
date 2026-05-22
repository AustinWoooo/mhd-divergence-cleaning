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
    // Standalone induction-only Powell-like source, not full 8-wave MHD Powell.
    POWELL_SOURCE,
    // In the standalone B-psi sandbox this reduces to MIXED_GLM behavior.
    MIXED_EGLM
};
