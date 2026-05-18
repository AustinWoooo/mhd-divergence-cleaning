#pragma once

#include "glm2d_common.hpp"
#include "glm2d.hpp"

#include <vector>

enum class ParabolicEnergyPolicy {
    KeepTotalEnergy,     // Do not change E.
    KeepInternalEnergy   // Adjust E by magnetic-energy change.
};

struct CleaningNorms {
    double L1   = 0.0;
    double L2   = 0.0;
    double Linf = 0.0;
};

struct ParabolicCleaningOptions {
    // Stability-aware subcycling for explicit diffusion.
    // Default false keeps the old one-step behavior.
    bool subcycle = false;

    // Only used when subcycle = true.
    double cfl_diffusion = 0.8;

    // If true and dt violates the explicit diffusion stability limit
    // while subcycle = false, throw an error.
    bool abort_if_unstable = false;

    // If true, compute before/after divergence norms and magnetic energy.
    // Default false avoids extra diagnostic cost.
    bool collect_diagnostics = false;

    // Default keeps the old behavior: do not modify total energy E.
    ParabolicEnergyPolicy energy_policy =
        ParabolicEnergyPolicy::KeepTotalEnergy;
};

struct ParabolicCleaningDiagnostics {
    CleaningNorms before;
    CleaningNorms after;

    double normalized_L2_before = 0.0;
    double normalized_L2_after  = 0.0;

    double magnetic_energy_before = 0.0;
    double magnetic_energy_after  = 0.0;

    double max_delta_B = 0.0;

    int n_substeps = 1;
    double dt_substep = 0.0;
    double dt_stability_limit = 0.0;
};

ParabolicCleaningDiagnostics apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params,
    const ParabolicCleaningOptions& options
);

// Backward-compatible wrapper.
// This keeps old code working.
void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);