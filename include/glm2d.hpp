#pragma once

#include <string>
#include <vector>

#include "state.hpp"
#include "glm.hpp"
#include "diagnostics.hpp"

struct GLM2DParams {
    int nx = 128;
    int ny = 128;

    double xlen = 1.0;
    double ylen = 1.0;

    double dx = 1.0 / 128.0;
    double dy = 1.0 / 128.0;

    double dt = 1.0e-3;
    double t_end = 0.5;
    double cfl = 0.25;

    // GLM wave speed and damping parameter.
    //
    // Hyperbolic GLM:
    //   divB errors are transported with speed ch.
    //
    // Mixed GLM:
    //   psi is additionally damped by
    //   exp[-dt * ch^2 / cp^2].
    double ch = 1.0;
    double cp = 0.2;

    // Velocity used only by the standalone Powell-source toy model.
    // This is not the full Powell 8-wave MHD system.
    double powell_vx = 1.0;
    double powell_vy = 0.5;

    // Elliptic projection settings.
    //
    // The projection implemented in glm2d.cpp uses a finite-volume-compatible
    // discrete divergence/gradient pair:
    //
    //   div_fv(B)_ij =
    //       (Bx_ij - Bx_{i-1,j}) / dx
    //     + (By_ij - By_{i,j-1}) / dy
    //
    //   Bx_ij <- Bx_ij - (phi_{i+1,j} - phi_ij) / dx
    //   By_ij <- By_ij - (phi_{i,j+1} - phi_ij) / dy
    //
    // With the 5-point Poisson operator, this makes
    //
    //   div_fv(B - grad_fv phi) = div_fv(B) - Laplacian_5pt(phi).
    int poisson_max_iter = 50000;
    double poisson_tol = 1.0e-12;
    double poisson_omega = 1.7;  // SOR parameter, must satisfy 0 < omega < 2.

    bool write_snapshot = true;
    bool write_initial_snapshot = false;

    std::string out_prefix = "glm_2d";
};

CleaningType parse_cleaning_type_2d(const std::string& name);

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
);

void initialize_divergence_pulse_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void update_hyperbolic_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void apply_powell_source_2d(
    std::vector<State>& U,
    const GLM2DParams& params
);

void advance_glm_2d_one_step(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
);

void write_glm_2d_snapshot(
    const std::vector<State>& U,
    const GLM2DParams& params,
    const std::string& filename
);

void run_glm_2d_case(
    CleaningType type,
    GLM2DParams params
);