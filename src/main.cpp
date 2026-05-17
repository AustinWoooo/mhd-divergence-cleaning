// =============================================================================
//  src/main.cpp
//  Brio-Wu Shock Tube — 1D HLLD + GLM comparison
//
//  Runs three scenarios:
//    1. Reference  — clean IC, no div-B error
//    2. No GLM     — sinusoidal Bx perturbation, no cleaning
//    3. With GLM   — same perturbation, hyperbolic-GLM cleaning active
//
//  Output: results/brio_wu_reference.csv
//          results/brio_wu_noGLM.csv
//          results/brio_wu_GLM.csv
//
//  Build:
//    g++ -std=c++17 -O2 -Iinclude src/hlld.cpp src/main.cpp -o brio_wu
//    ./brio_wu
// =============================================================================

#include "HLLD_mhd_solver.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

using namespace MHD;

// -----------------------------------------------------------------------------
//  GLM (Dedner 2002) Rusanov correction for the (Bx, psi) subsystem.
//  Overwrites F[BX] and F[PSI] at every interface.
// -----------------------------------------------------------------------------
static void glm_flux_correction(std::vector<std::vector<double>>& F,
                                 const std::vector<std::vector<double>>& U,
                                 double ch, int N)
{
    for (int face = 0; face <= N; ++face) {
        double BxL  = (face == 0) ? U[0][IB1]   : U[face - 1][IB1];
        double BxR  = (face == N) ? U[N-1][IB1]  : U[face][IB1];
        double psiL = (face == 0) ? U[0][IPSI]  : U[face - 1][IPSI];
        double psiR = (face == N) ? U[N-1][IPSI] : U[face][IPSI];

        F[face][IB1]  = 0.5*(psiL + psiR) - 0.5*ch*(BxR  - BxL);
        F[face][IPSI] = 0.5*ch*ch*(BxL + BxR) - 0.5*ch*(psiR - psiL);
    }
}

// -----------------------------------------------------------------------------
//  Single simulation run
// -----------------------------------------------------------------------------
struct SimConfig {
    bool   div_b_pert;  // add sinusoidal Bx perturbation
    bool   glm;         // apply GLM cleaning
    double ch;          // GLM wave speed (used only when glm=true)
};

static void simulate(const SimConfig& cfg, const std::string& outfile,
                     int N, double gamma, double t_end, double cfl)
{
    const double x_left = -0.5, x_right = 0.5;
    const double dx = (x_right - x_left) / N;

    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
        x[i] = x_left + (i + 0.5) * dx;

    // Brio-Wu initial conditions (discontinuity at x = 0)
    std::vector<std::vector<double>> U(N);
    for (int i = 0; i < N; ++i) {
        PrimState W = (x[i] < 0.0)
            ? PrimState(1.0,   0, 0, 0, 1.0,  0.75,  1.0, 0)
            : PrimState(0.125, 0, 0, 0, 0.1,  0.75, -1.0, 0);
        if (cfg.div_b_pert)
            W.Bx += 0.05 * std::sin(4.0 * M_PI * x[i]);
        U[i] = W.to_conserved(gamma);
    }

    double t = 0.0;
    while (t < t_end - 1e-12) {
        // CFL: GLM wave ch must also be resolved
        double max_sp = 0.0;
        for (int i = 0; i < N; ++i) {
            PrimState W = PrimState::from_conserved(U[i], gamma);
            double sp = std::abs(W.u) + fast_magnetosonic_speed(W, gamma);
            if (cfg.glm) sp = std::max(sp, cfg.ch);
            max_sp = std::max(max_sp, sp);
        }
        double dt = std::min(cfl * dx / max_sp, t_end - t);

        // HLLD fluxes (outflow BC: ghost = edge cell)
        std::vector<std::vector<double>> F(N + 1);
        {
            PrimState WL = PrimState::from_conserved(U[0], gamma);
            F[0] = compute_flux(WL, WL, 0, gamma);
        }
        for (int i = 0; i < N - 1; ++i) {
            PrimState WL = PrimState::from_conserved(U[i],     gamma);
            PrimState WR = PrimState::from_conserved(U[i + 1], gamma);
            F[i + 1] = compute_flux(WL, WR, 0, gamma);
        }
        {
            PrimState WR = PrimState::from_conserved(U[N - 1], gamma);
            F[N] = compute_flux(WR, WR, 0, gamma);
        }

        // Replace (Bx, psi) fluxes with GLM correction
        if (cfg.glm)
            glm_flux_correction(F, U, cfg.ch, N);

        // Conservative update
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < NVAR; ++k)
                U[i][k] -= (dt / dx) * (F[i + 1][k] - F[i][k]);

        // GLM: mixed exponential damping of psi (Dedner 2002, eq. 24)
        if (cfg.glm) {
            double cp    = cfg.ch / std::sqrt(2.0);
            double decay = std::exp(-dt * cfg.ch * cfg.ch / (cp * cp));
            for (int i = 0; i < N; ++i)
                U[i][IPSI] *= decay;
        }

        t += dt;
    }

    // Write CSV
    std::ofstream out(outfile);
    out << "x,rho,u,v,w,p,Bx,By,Bz\n";
    out << std::scientific << std::setprecision(10);
    for (int i = 0; i < N; ++i) {
        PrimState W = PrimState::from_conserved(U[i], gamma);
        out << x[i]  << ","
            << W.rho << "," << W.u << "," << W.v << "," << W.w << ","
            << W.p   << "," << W.Bx << "," << W.By << "," << W.Bz << "\n";
    }
}

// -----------------------------------------------------------------------------
//  Main
// -----------------------------------------------------------------------------
int main()
{
    const int    N     = 400;
    const double gamma = 2.0;
    const double t_end = 0.1;
    const double cfl   = 0.4;
    const double ch    = 2.0;  // GLM wave speed

    std::filesystem::create_directories("results");

    const std::string sep(58, '=');
    std::cout << sep << "\n";
    std::cout << " Brio-Wu Shock Tube — HLLD + GLM Comparison\n";
    std::cout << " N=" << N << "  gamma=" << gamma
              << "  t_end=" << t_end << "  CFL=" << cfl
              << "  ch=" << ch << "\n";
    std::cout << sep << "\n";

    struct Run { std::string label, file; SimConfig cfg; };
    std::vector<Run> runs = {
        {"Reference (clean IC)",       "results/brio_wu_reference.csv", {false, false, ch}},
        {"No GLM  (div-B perturbed)",  "results/brio_wu_noGLM.csv",     {true,  false, ch}},
        {"With GLM (div-B perturbed)", "results/brio_wu_GLM.csv",       {true,  true,  ch}},
    };

    for (auto& r : runs) {
        std::cout << " [run] " << r.label << "\n";
        simulate(r.cfg, r.file, N, gamma, t_end, cfl);
        std::cout << "        → " << r.file << "\n";
    }

    std::cout << sep << "\n";
    return 0;
}
