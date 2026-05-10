#include "glm.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

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

void update_hyperbolic_glm_1d(
    std::vector<State>& U,
    const GLMParams& params
) {
    const int N = static_cast<int>(U.size());
    const double dx = params.dx;
    const double dt = params.dt;
    const double ch = params.ch;

    std::vector<GLMFlux> flux(N + 1);

    // Periodic boundary for toy test.
    for (int iface = 0; iface <= N; ++iface) {
        int iL = (iface - 1 + N) % N;
        int iR = iface % N;

        flux[iface] = compute_hyperbolic_glm_flux(
            U[iL][BX],
            U[iL][PSI],
            U[iR][BX],
            U[iR][PSI],
            ch
        );
    }

    std::vector<State> Uold = U;

    for (int i = 0; i < N; ++i) {
        const int iLface = i;
        const int iRface = i + 1;

        U[i][BX] =
            Uold[i][BX]
          - dt / dx * (flux[iRface].FBn - flux[iLface].FBn);

        U[i][PSI] =
            Uold[i][PSI]
          - dt / dx * (flux[iRface].Fpsi - flux[iLface].Fpsi);
    }
}

void apply_mixed_glm_damping(
    std::vector<State>& U,
    const GLMParams& params
) {
    const double ch = params.ch;
    const double cp = params.cp;
    const double dt = params.dt;

    const double factor = std::exp(-dt * ch * ch / (cp * cp));

    for (auto& cell : U) {
        cell[PSI] *= factor;
    }
}

void apply_parabolic_cleaning_1d(
    std::vector<State>& U,
    const GLMParams& params
) {
    const int N = static_cast<int>(U.size());
    const double dx = params.dx;
    const double dt = params.dt;
    const double cp = params.cp;

    std::vector<double> divB(N, 0.0);
    std::vector<double> gradDivB(N, 0.0);

    for (int i = 0; i < N; ++i) {
        int ip = (i + 1) % N;
        int im = (i - 1 + N) % N;

        divB[i] = (U[ip][BX] - U[im][BX]) / (2.0 * dx);
    }

    for (int i = 0; i < N; ++i) {
        int ip = (i + 1) % N;
        int im = (i - 1 + N) % N;

        gradDivB[i] = (divB[ip] - divB[im]) / (2.0 * dx);
    }

    for (int i = 0; i < N; ++i) {
        U[i][BX] += dt * cp * cp * gradDivB[i];
    }
}

double compute_divB_1d(
    const std::vector<State>& U,
    int i,
    double dx
) {
    const int N = static_cast<int>(U.size());
    const int ip = (i + 1) % N;
    const int im = (i - 1 + N) % N;

    return (U[ip][BX] - U[im][BX]) / (2.0 * dx);
}

void compute_divB_norms_1d(
    const std::vector<State>& U,
    double dx,
    double& L1,
    double& L2,
    double& Linf
) {
    const int N = static_cast<int>(U.size());

    L1 = 0.0;
    L2 = 0.0;
    Linf = 0.0;

    for (int i = 0; i < N; ++i) {
        const double d = compute_divB_1d(U, i, dx);

        L1 += std::abs(d);
        L2 += d * d;
        Linf = std::max(Linf, std::abs(d));
    }

    L1 /= static_cast<double>(N);
    L2 = std::sqrt(L2 / static_cast<double>(N));
}