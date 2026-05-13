#include "glm.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

int normal_B_index(Direction dir) {
    switch (dir) {
        case Direction::X:
            return BX;
        case Direction::Y:
            return BY;
        case Direction::Z:
            return BZ;
        default:
            throw std::runtime_error("Unknown direction");
    }
}

void add_glm_flux_correction(
    State& flux,
    const State& UL,
    const State& UR,
    Direction dir,
    const GLMParams& params
) {
    if (
        params.type != CleaningType::HYPERBOLIC_GLM &&
        params.type != CleaningType::MIXED_GLM &&
        params.type != CleaningType::MIXED_EGLM
    ) {
        return;
    }

    const int Bn = normal_B_index(dir);

    GLMFlux Fglm = compute_hyperbolic_glm_flux(
        UL[Bn],
        UL[PSI],
        UR[Bn],
        UR[PSI],
        params.ch
    );

    flux[Bn]  += Fglm.FBn;
    flux[PSI] += Fglm.Fpsi;
}

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

void apply_parabolic_cleaning_1d(
    std::vector<State>& U,
    const GLMParams& params
) {
    const int N = static_cast<int>(U.size());
    const double dx = params.dx;
    const double dt = params.dt;
    const double D = params.cp * params.cp;

    if (N < 3) {
        throw std::runtime_error("apply_parabolic_cleaning_1d requires N >= 3");
    }
    if (dx <= 0.0) {
        throw std::runtime_error("dx must be positive");
    }
    if (dt < 0.0) {
        throw std::runtime_error("dt must be non-negative");
    }
    if (params.cp < 0.0) {
        throw std::runtime_error("cp must be non-negative");
    }

    std::vector<State> Uold = U;

    for (int i = 0; i < N; ++i) {
        const int ip = (i + 1) % N;
        const int im = (i - 1 + N) % N;

        const double lapBx =
            (Uold[ip][BX] - 2.0 * Uold[i][BX] + Uold[im][BX]) / (dx * dx);

        U[i][BX] = Uold[i][BX] + dt * D * lapBx;
    }
}

void apply_mixed_glm_damping(
    std::vector<State>& U,
    const GLMParams& params
) {
    const double ch = params.ch;
    const double cp = params.cp;
    const double dt = params.dt;

    if (cp <= 0.0) {
        throw std::runtime_error("cp must be positive for mixed GLM damping");
    }
    if (ch < 0.0) {
        throw std::runtime_error("ch must be non-negative");
    }
    if (dt < 0.0) {
        throw std::runtime_error("dt must be non-negative");
    }

    const double factor = std::exp(-dt * ch * ch / (cp * cp));

    for (auto& cell : U) {
        cell[PSI] *= factor;
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