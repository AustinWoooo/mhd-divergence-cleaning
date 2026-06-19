#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Galilean-invariant mixed EGLM source terms for the 2D cleaning framework,
// corresponding to the Galilean-invariant extended-GLM source formulation.
//
// This is non-conservative and distinct from MIXED_EGLM because it also applies
// the magnetic-field and psi advection corrections needed by the intended
// Galilean-invariant EGLM formulation.
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

// Galilean-invariant mixed EGLM = mixed GLM B-psi update (including the
// policy-selected magnetic-energy repair) followed by the non-conservative
// GI-EGLM source above. The source's own B/E changes are not repaired again.
void update_gi_mixed_eglm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
