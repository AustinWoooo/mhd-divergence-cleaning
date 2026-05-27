#include "cleaning_plugin.hpp"

#include <cmath>
#include <stdexcept>

void CleaningPlugin::beforeStage(
    std::vector<State>&,
    const CleaningStageContext&
) {}

void CleaningPlugin::modifyInterfaceFlux(
    State&,
    const State&,
    const State&,
    int,
    const CleaningStageContext&
) {}

void CleaningPlugin::applySourceTerms(
    std::vector<State>&,
    const CleaningStageContext&
) {}

void CleaningPlugin::afterStage(
    std::vector<State>&,
    const CleaningStageContext&
) {}

void CleaningPlugin::afterStep(
    std::vector<State>&,
    const CleaningStageContext&
) {}

HyperbolicGlmPlugin::HyperbolicGlmPlugin(double ch)
    : ch_(ch) {
    if (!(ch_ > 0.0)) {
        throw std::invalid_argument("HyperbolicGlmPlugin: ch must be positive");
    }
}

std::string HyperbolicGlmPlugin::name() const {
    return "hyperbolic_glm";
}

bool HyperbolicGlmPlugin::usesPsi() const {
    return true;
}

bool HyperbolicGlmPlugin::modifiesFlux() const {
    return true;
}

void HyperbolicGlmPlugin::modifyInterfaceFlux(
    State& flux,
    const State& UL,
    const State& UR,
    int direction,
    const CleaningStageContext&
) {
    const int Bn = normal_magnetic_index_for_direction(direction);
    const GlmInterfaceFlux glm_flux =
        compute_glm_interface_flux(
            UL[Bn],
            UL[PSI],
            UR[Bn],
            UR[PSI],
            ch_
        );

    add_glm_interface_flux(flux, direction, glm_flux);
}

MixedGlmPlugin::MixedGlmPlugin(double ch, double cp)
    : ch_(ch),
      cp_(cp) {
    if (!(ch_ > 0.0)) {
        throw std::invalid_argument("MixedGlmPlugin: ch must be positive");
    }
    if (!(cp_ > 0.0)) {
        throw std::invalid_argument("MixedGlmPlugin: cp must be positive");
    }
}

std::string MixedGlmPlugin::name() const {
    return "mixed_glm";
}

bool MixedGlmPlugin::usesPsi() const {
    return true;
}

bool MixedGlmPlugin::modifiesFlux() const {
    return true;
}

bool MixedGlmPlugin::hasSourceTerms() const {
    return true;
}

void MixedGlmPlugin::modifyInterfaceFlux(
    State& flux,
    const State& UL,
    const State& UR,
    int direction,
    const CleaningStageContext& ctx
) {
    HyperbolicGlmPlugin hyperbolic(ch_);
    hyperbolic.modifyInterfaceFlux(flux, UL, UR, direction, ctx);
}

void MixedGlmPlugin::applySourceTerms(
    std::vector<State>& U,
    const CleaningStageContext& ctx
) {
    const double damping_rate = ch_ * ch_ / (cp_ * cp_);
    const double factor = std::exp(-ctx.dt_stage * damping_rate);

    for (State& cell : U) {
        cell[PSI] *= factor;
    }
}
