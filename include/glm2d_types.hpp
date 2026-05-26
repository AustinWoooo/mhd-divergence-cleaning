#pragma once

#include <string>

enum class CleaningEnergyPolicy {
    ConserveTotalEnergy,
    PreserveThermalPressure
};

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

    CleaningEnergyPolicy energy_policy =
        CleaningEnergyPolicy::ConserveTotalEnergy;

    // Velocity used only by the standalone Powell-like source toy model.
    // This is not the full Powell 8-wave MHD system.
    double powell_vx = 1.0;
    double powell_vy = 0.5;

    // Elliptic projection settings.
    //
    // The projection implemented in projection2d.cpp uses a finite-volume-compatible
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
