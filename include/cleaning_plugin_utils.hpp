#pragma once

#include "state.hpp"

struct GlmInterfaceFlux {
    double Bn_star = 0.0;
    double psi_star = 0.0;
    double F_Bn = 0.0;
    double F_psi = 0.0;
};

int normal_magnetic_index_for_direction(int direction);

GlmInterfaceFlux compute_glm_interface_flux(
    double Bn_L,
    double psi_L,
    double Bn_R,
    double psi_R,
    double ch
);

void add_glm_interface_flux(
    State& flux,
    int direction,
    const GlmInterfaceFlux& glm_flux
);
