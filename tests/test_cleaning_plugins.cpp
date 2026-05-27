#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "HLLD_mhd_solver.hpp"
#include "cleaning_plugin.hpp"
#include "cleaning_plugin_utils.hpp"

namespace {

bool approx_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

void test_glm_interface_riemann_state() {
    const double ch = 2.0;
    const double Bn_L = 1.2;
    const double Bn_R = 0.8;
    const double psi_L = 0.3;
    const double psi_R = -0.1;

    const GlmInterfaceFlux flux =
        compute_glm_interface_flux(Bn_L, psi_L, Bn_R, psi_R, ch);

    const double expected_Bn_star =
        0.5 * (Bn_L + Bn_R) - 0.5 / ch * (psi_R - psi_L);
    const double expected_psi_star =
        0.5 * (psi_L + psi_R) - 0.5 * ch * (Bn_R - Bn_L);

    assert(approx_equal(flux.Bn_star, expected_Bn_star));
    assert(approx_equal(flux.psi_star, expected_psi_star));
}

void test_glm_flux_values() {
    const double ch = 1.7;
    const double Bn_L = -0.4;
    const double Bn_R = 0.2;
    const double psi_L = 0.5;
    const double psi_R = 0.1;

    const GlmInterfaceFlux flux =
        compute_glm_interface_flux(Bn_L, psi_L, Bn_R, psi_R, ch);

    assert(approx_equal(flux.F_Bn, flux.psi_star));
    assert(approx_equal(flux.F_psi, ch * ch * flux.Bn_star));
}

void test_hyperbolic_glm_plugin_flux_hook() {
    State UL{};
    State UR{};
    State flux{};

    UL[BX] = 1.0;
    UR[BX] = 0.6;
    UL[PSI] = 0.25;
    UR[PSI] = -0.15;

    const double ch = 2.0;
    HyperbolicGlmPlugin plugin(ch);
    CleaningStageContext ctx;

    plugin.modifyInterfaceFlux(flux, UL, UR, 0, ctx);

    const GlmInterfaceFlux expected =
        compute_glm_interface_flux(UL[BX], UL[PSI], UR[BX], UR[PSI], ch);

    assert(approx_equal(flux[BX], expected.F_Bn));
    assert(approx_equal(flux[PSI], expected.F_psi));
    assert(approx_equal(flux[BY], 0.0));
}

void test_mixed_glm_damping_once() {
    const double ch = 1.5;
    const double cp = 0.5;
    const double dt = 0.02;

    MixedGlmPlugin plugin(ch, cp);

    CleaningStageContext ctx;
    ctx.dt_stage = dt;

    std::vector<State> U(3);
    U[0][PSI] = 0.5;
    U[1][PSI] = -0.25;
    U[2][PSI] = 1.25;

    const std::vector<State> Uold = U;
    plugin.applySourceTerms(U, ctx);

    const double factor = std::exp(-dt * ch * ch / (cp * cp));

    for (std::size_t i = 0; i < U.size(); ++i) {
        assert(approx_equal(U[i][PSI], Uold[i][PSI] * factor));
    }
}

void test_plugin_metadata() {
    HyperbolicGlmPlugin hyperbolic(1.0);
    MixedGlmPlugin mixed(1.0, 0.2);

    assert(hyperbolic.name() == "hyperbolic_glm");
    assert(hyperbolic.usesPsi());
    assert(hyperbolic.modifiesFlux());
    assert(!hyperbolic.hasSourceTerms());
    assert(!hyperbolic.hasCorrectionStep());

    assert(mixed.name() == "mixed_glm");
    assert(mixed.usesPsi());
    assert(mixed.modifiesFlux());
    assert(mixed.hasSourceTerms());
    assert(!mixed.hasCorrectionStep());
}

void test_raw_diagnostics_compatibility() {
    const double gamma = 5.0 / 3.0;

    State U{};
    U[RHO] = 1.0;
    U[MX] = 0.0;
    U[MY] = 0.0;
    U[MZ] = 0.0;
    U[BX] = 1.0;
    U[BY] = 0.0;
    U[BZ] = 0.0;
    U[E] = 0.1;
    U[PSI] = 0.0;

    const MHD::PrimState raw =
        MHD::PrimState::from_conserved_raw(U, gamma);

    MHD::PrimitiveRecoveryStatus status;
    const MHD::PrimState checked =
        MHD::PrimState::from_conserved_checked(U, gamma, &status);

    assert(raw.p < 0.0);
    assert(!approx_equal(raw.p, TINY_NUMBER));
    assert(status.pressure_floored);
    assert(approx_equal(checked.p, TINY_NUMBER));
}

} // namespace

int main() {
    test_glm_interface_riemann_state();
    test_glm_flux_values();
    test_hyperbolic_glm_plugin_flux_hook();
    test_mixed_glm_damping_once();
    test_plugin_metadata();
    test_raw_diagnostics_compatibility();

    std::cout << "Cleaning plugin assertions passed.\n";
    return 0;
}
