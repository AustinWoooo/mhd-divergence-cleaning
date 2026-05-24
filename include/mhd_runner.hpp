#pragma once

// =============================================================================
//  include/mhd_runner.hpp
//
//  Integrated 2D ideal-MHD runner that couples:
//    - HLLD Riemann solver  (src/hlld_solver.cpp, namespace MHD)
//    - GLM divergence-cleaning pipeline  (src/glm2d.cpp)
//
//  Time-stepping strategy: operator-split RK2(HLLD) + advance_glm_2d_one_step.
//  The HLLD step advances rho, momenta, energy, and B with ideal-MHD fluxes;
//  the GLM step (selected via CleaningType) corrects divB on top.
//
//  Supported problems: Orszag-Tang vortex and Brio-Wu shock tube (2D strip).
// =============================================================================

#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

struct MHDRunParams {
    // GLM/grid parameters reused as-is by advance_glm_2d_one_step.
    GLM2DParams glm;

    // Ratio of specific heats used by the HLLD solver.
    double gamma = 5.0 / 3.0;

    // Problem selector: "orszag_tang" or "brio_wu".
    std::string problem = "orszag_tang";
};

void initialize_orszag_tang_2d(
    std::vector<State>& U,
    const MHDRunParams& params
);

void initialize_brio_wu_strip_2d(
    std::vector<State>& U,
    const MHDRunParams& params
);

void run_mhd_2d_case(
    CleaningType type,
    MHDRunParams params
);
