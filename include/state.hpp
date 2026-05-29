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
    // Same Powell source formula, source-subcycled to limit |dt * divB|.
    // Non-conservative; keeps the original POWELL_SOURCE behavior unchanged.
    POWELL_SOURCE_SUBCYCLED,
    // Pressure-limited Powell source. Cautionary positivity policy that limits
    // the non-conservative source increment instead of flooring pressure.
    POWELL_SOURCE_LIMITED,
    // Mixed GLM plus EGLM momentum/energy source terms.
    // Non-conservative; distinct from Powell 8-wave source.
    MIXED_EGLM,
    // Galilean-invariant mixed EGLM source terms.
    // Non-conservative; includes momentum, magnetic-field, energy, and psi sources.
    GI_MIXED_EGLM
};
