#pragma once

#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

// Full Powell 8-wave source increment for one 2.5D MHD cell:
// dU/dt = -(divB) [0, Bx, By, Bz, ux, uy, uz, u.B, 0]^T.
// The returned State is the finite-time increment dt*S(U, divB).
State compute_powell_source_increment_cell(
    const State& cell,
    double divB,
    double dt
);

// Full operator-split Powell 8-wave source step for 2.5D MHD.
void apply_powell_source_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);
