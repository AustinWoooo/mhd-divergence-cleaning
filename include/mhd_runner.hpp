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
//  Supported problems: Orszag-Tang vortex, field-loop advection, and
//  divergence advection.
// =============================================================================

#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "mhd_reconstruction.hpp"
#include "state.hpp"

struct MHDRunParams {
    // GLM/grid parameters reused as-is by advance_glm_2d_one_step.
    GLM2DParams glm;

    // Ratio of specific heats used by the HLLD solver.
    double gamma = 5.0 / 3.0;

    // Problem selector: "orszag_tang", "field_loop",
    // or "divergence_advection".
    std::string problem = "orszag_tang";

    // High-resolution shock-capturing controls for the HLLD finite-volume update.
    // PLM + MC gives a second-order MUSCL-HLLD scheme by default; PCM falls back
    // to first-order Godunov for comparison/debugging.
    MHD::Reconstruction reconstruction = MHD::Reconstruction::PLM;
    MHD::SlopeLimiter   limiter        = MHD::SlopeLimiter::MC;

    // Root directory for runner-generated CSV outputs.  The default preserves the
    // historical output layout:
    //   results/mhd_runner/{divergence,snapshots,summaries,failures}
    std::string output_root = "results/mhd_runner";

    // Write time-history diagnostics every N accepted steps.  Step 0, the final
    // step, and failure rows are always written.  Use values > 1 for benchmarks to
    // reduce I/O without changing the numerical update.
    int diagnostic_stride = 1;

    // Benchmark-oriented mode flag used by CLIs/scripts.  The solver does not
    // change physics based on this flag; callers use it to select sparse
    // diagnostics and disabled snapshots.
    bool performance_mode = false;
};

void initialize_orszag_tang_2d(
    std::vector<State>& U,
    const MHDRunParams& params
);

void initialize_field_loop_2d(
    std::vector<State>& U,
    const MHDRunParams& params
);

void initialize_divergence_advection_2d(
    std::vector<State>& U,
    const MHDRunParams& params
);

void run_mhd_2d_case(
    CleaningType type,
    MHDRunParams params
);
