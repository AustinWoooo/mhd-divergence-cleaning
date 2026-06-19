#include "glm.hpp"

// Shared GLM helpers used by the 2D cleaning kernels and the runner.  (The
// standalone 1D GLM toy solver that used to live here was removed together with
// tests/test_glm_1d.cpp, its only caller.)

std::string cleaning_name(CleaningType type) {
    switch (type) {
        case CleaningType::NONE:
            return "none";
        case CleaningType::PARABOLIC:
            return "parabolic";
        case CleaningType::HYPERBOLIC_GLM:
            return "hyperbolic_glm";
        case CleaningType::MIXED_GLM:
            return "mixed_glm";
        case CleaningType::ELLIPTIC_PROJECTION:
            return "elliptic_projection";
        case CleaningType::POWELL_SOURCE:
            return "powell_source";
        case CleaningType::MIXED_EGLM:
            return "mixed_eglm";
        case CleaningType::GI_MIXED_EGLM:
            return "gi_mixed_eglm";
        default:
            return "unknown";
    }
}

GLMFlux compute_hyperbolic_glm_flux(
    double BnL,
    double psiL,
    double BnR,
    double psiR,
    double ch
) {
    GLMFlux F{};

    // Physical flux:
    // F(Bn)  = psi
    // F(psi) = ch^2 Bn
    const double FBn_L  = psiL;
    const double FBn_R  = psiR;
    const double Fpsi_L = ch * ch * BnL;
    const double Fpsi_R = ch * ch * BnR;

    // Rusanov flux for the 2x2 GLM subsystem
    F.FBn =
        0.5 * (FBn_L + FBn_R)
      - 0.5 * ch * (BnR - BnL);

    F.Fpsi =
        0.5 * (Fpsi_L + Fpsi_R)
      - 0.5 * ch * (psiR - psiL);

    return F;
}
