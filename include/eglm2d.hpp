#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Extended GLM (EGLM) source terms for the 2D cleaning framework.
//
// This is NOT Powell.
// EGLM adds the formal non-conservative extended-GLM momentum and energy source
// terms used by the mixed_eglm method:
//
//   d(rho*u)/dt = -(divB) B
//   dE/dt       = -B . grad(psi)
//
// It does NOT apply the Powell magnetic-field source:
//
//   dB/dt = -(divB) u
//
// State layout:
//
//   U = [rho, rho*ux, rho*uy, rho*uz, Bx, By, Bz, E, psi]
void apply_eglm_source_2d(
    std::vector<State>& U,
    const std::vector<State>& Uref,
    const GLM2DParams& params
);

// Mixed EGLM = mixed GLM B-psi update + EGLM momentum/energy source.
//
// Operator-split order:
//   1. Save Uold.
//   2. Apply mixed GLM update to B and psi, including the policy-selected
//      magnetic-energy repair before any EGLM source terms.
//   3. Apply EGLM source using Uold as reference state.
void update_mixed_eglm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
