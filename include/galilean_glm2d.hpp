#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Experimental Galilean-invariant mixed EGLM source terms for the standalone
// 2D cleaning framework, corresponding to Dedner et al. Eq. (38).
//
// This is non-conservative and distinct from the existing MIXED_EGLM source.
//
// Source terms:
//
//   d(rho*u)/dt = -(divB) B
//   dB/dt       = -(divB) u
//   dE/dt       = -(divB)(u.B) - B.grad(psi)
//   dpsi/dt     = -u.grad(psi)
//
// State layout:
//
//   U = [rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, psi]
void apply_gi_eglm_source_2d(
    std::vector<State>& U,
    const std::vector<State>& Uref,
    const GLM2DParams& params
);

// Galilean-invariant mixed EGLM = mixed GLM B-psi update plus the
// non-conservative GI-EGLM source above.
void update_gi_mixed_eglm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
