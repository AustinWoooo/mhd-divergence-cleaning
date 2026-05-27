#include "cleaning_plugin_utils.hpp"

#include <stdexcept>

int normal_magnetic_index_for_direction(int direction) {
    if (direction == 0) {
        return BX;
    }
    if (direction == 1) {
        return BY;
    }

    throw std::invalid_argument(
        "normal_magnetic_index_for_direction: direction must be 0 or 1"
    );
}

GlmInterfaceFlux compute_glm_interface_flux(
    double Bn_L,
    double psi_L,
    double Bn_R,
    double psi_R,
    double ch
) {
    if (!(ch > 0.0)) {
        throw std::invalid_argument(
            "compute_glm_interface_flux: ch must be positive"
        );
    }

    GlmInterfaceFlux out;

    out.Bn_star =
        0.5 * (Bn_L + Bn_R)
      - 0.5 / ch * (psi_R - psi_L);

    out.psi_star =
        0.5 * (psi_L + psi_R)
      - 0.5 * ch * (Bn_R - Bn_L);

    out.F_Bn = out.psi_star;
    out.F_psi = ch * ch * out.Bn_star;

    return out;
}

void add_glm_interface_flux(
    State& flux,
    int direction,
    const GlmInterfaceFlux& glm_flux
) {
    const int Bn = normal_magnetic_index_for_direction(direction);
    flux[Bn] += glm_flux.F_Bn;
    flux[PSI] += glm_flux.F_psi;
}
