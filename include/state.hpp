#pragma once
#include <array>

// Conservative variable layout (2.5D MHD, 9 components)
// Ordering: density, momenta, magnetic field, energy, GLM scalar
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

// Floor value used to guard against division by zero across all solvers
constexpr double TINY_NUMBER = 1.0e-20;

enum class CleaningType {
    NONE,
    PARABOLIC,
    HYPERBOLIC_GLM,
    MIXED_GLM,
    ELLIPTIC_PROJECTION,
    // Full operator-split Powell 8-wave source for 2.5D MHD.
    // Non-conservative; transports divB errors with local fluid velocity.
    POWELL_SOURCE,
    // Mixed GLM plus EGLM momentum/energy source terms.
    // Non-conservative; distinct from Powell 8-wave source.
    MIXED_EGLM
};
