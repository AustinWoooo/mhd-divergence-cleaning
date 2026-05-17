// =============================================================================
//  src/orszag_tang.cpp
//  Orszag-Tang Vortex — 2D HLLD + GLM comparison
//
//  IC (Dahlburg & Picone 1989 / Stone et al. 2008):
//    Domain  : [0,1]×[0,1] periodic
//    rho     = 25/(36 pi),   p  = 5/(12 pi)
//    u       = -sin(2 pi y), v  =  sin(2 pi x)
//    Bx      = -B0 sin(2 pi y),  By = B0 sin(4 pi x),  B0 = 1/sqrt(4 pi)
//
//  Runs two scenarios:
//    1. No GLM  — HLLD only
//    2. With GLM — HLLD + hyperbolic-GLM (Dedner 2002)
//
//  Output: results/orszag_tang/noGLM.csv
//          results/orszag_tang/withGLM.csv
//
//  Build: see CMakeLists.txt target 'orszag_tang'
// =============================================================================

#include "HLLD_mhd_solver.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <string>
#include <algorithm>

using namespace MHD;

// Flat index: outer loop i (x), inner loop j (y)
static inline int idx2(int i, int j, int Ny) { return i * Ny + j; }

// -------------------------------------------------------------------------
//  Max signal speed across all cells for CFL condition
// -------------------------------------------------------------------------
static double max_signal(const std::vector<std::vector<double>>& U,
                         double gamma, bool glm, double ch)
{
    double sp = 0.0;
    for (const auto& u : U) {
        PrimState W = PrimState::from_conserved(u, gamma);
        double rho = std::max(W.rho, TINY_NUMBER);
        double a2  = gamma * W.p / rho;
        double b2  = (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz) / rho;
        double bn2_x = W.Bx*W.Bx / rho;
        double bn2_y = W.By*W.By / rho;
        double term  = a2 + b2;
        double cf_x  = std::sqrt(std::max(0.5*(term + std::sqrt(std::max(term*term - 4*a2*bn2_x, 0.0))), 0.0));
        double cf_y  = std::sqrt(std::max(0.5*(term + std::sqrt(std::max(term*term - 4*a2*bn2_y, 0.0))), 0.0));
        double s = std::max(std::abs(W.u) + cf_x, std::abs(W.v) + cf_y);
        if (glm) s = std::max(s, ch);
        sp = std::max(sp, s);
    }
    return sp;
}

// -------------------------------------------------------------------------
//  Compute RHS = -div(F) for 2D finite-volume update (periodic BC)
//
//  Convention:
//    Fx[idx2(i,j)] = x-flux at right face of cell (i,j)
//                    i.e., between cell (i,j) and ((i+1)%Nx, j)
//    Fy[idx2(i,j)] = y-flux at top  face of cell (i,j)
//                    i.e., between cell (i,j) and (i, (j+1)%Ny)
//
//  Update: dU[i,j]/dt = -( Fx[i,j] - Fx[(i-1)%Nx, j] ) / dx
//                       -( Fy[i,j] - Fy[i, (j-1)%Ny] ) / dy
// -------------------------------------------------------------------------
static std::vector<std::vector<double>> compute_rhs(
    const std::vector<std::vector<double>>& U,
    int Nx, int Ny, double dx, double dy,
    double gamma, bool glm, double ch)
{
    int NC = Nx * Ny;
    std::vector<std::vector<double>> Fx(NC, std::vector<double>(NVAR, 0.0));
    std::vector<std::vector<double>> Fy(NC, std::vector<double>(NVAR, 0.0));

    // X-fluxes
    for (int i = 0; i < Nx; ++i) {
        int ip1 = (i + 1) % Nx;
        for (int j = 0; j < Ny; ++j) {
            int cL = idx2(i,   j, Ny);
            int cR = idx2(ip1, j, Ny);
            PrimState WL = PrimState::from_conserved(U[cL], gamma);
            PrimState WR = PrimState::from_conserved(U[cR], gamma);
            Fx[cL] = compute_flux(WL, WR, 0, gamma);
            if (glm) {
                double BxL  = U[cL][IB1], BxR  = U[cR][IB1];
                double psiL = U[cL][IPSI], psiR = U[cR][IPSI];
                Fx[cL][IB1]  = 0.5*(psiL + psiR)       - 0.5*ch*(BxR  - BxL);
                Fx[cL][IPSI] = 0.5*ch*ch*(BxL + BxR)   - 0.5*ch*(psiR - psiL);
            }
        }
    }

    // Y-fluxes
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            int jp1 = (j + 1) % Ny;
            int cL = idx2(i, j,   Ny);
            int cR = idx2(i, jp1, Ny);
            PrimState WL = PrimState::from_conserved(U[cL], gamma);
            PrimState WR = PrimState::from_conserved(U[cR], gamma);
            Fy[cL] = compute_flux(WL, WR, 1, gamma);
            if (glm) {
                double ByL  = U[cL][IB2], ByR  = U[cR][IB2];
                double psiL = U[cL][IPSI], psiR = U[cR][IPSI];
                Fy[cL][IB2]  = 0.5*(psiL + psiR)       - 0.5*ch*(ByR  - ByL);
                Fy[cL][IPSI] = 0.5*ch*ch*(ByL + ByR)   - 0.5*ch*(psiR - psiL);
            }
        }
    }

    // RHS = -div(F)
    double inv_dx = 1.0 / dx, inv_dy = 1.0 / dy;
    std::vector<std::vector<double>> RHS(NC, std::vector<double>(NVAR, 0.0));
    for (int i = 0; i < Nx; ++i) {
        int im1 = (i - 1 + Nx) % Nx;
        for (int j = 0; j < Ny; ++j) {
            int jm1 = (j - 1 + Ny) % Ny;
            int c = idx2(i, j, Ny);
            for (int k = 0; k < NVAR; ++k) {
                RHS[c][k] -= inv_dx * (Fx[c][k] - Fx[idx2(im1, j,   Ny)][k])
                           + inv_dy * (Fy[c][k] - Fy[idx2(i,   jm1, Ny)][k]);
            }
        }
    }
    return RHS;
}

// -------------------------------------------------------------------------
//  RK2 (Heun's method) + operator-split GLM exponential damping
// -------------------------------------------------------------------------
static void rk2_step(std::vector<std::vector<double>>& U,
                     int Nx, int Ny, double dx, double dy,
                     double gamma, bool glm, double ch, double dt)
{
    int NC = Nx * Ny;

    // Stage 1: U* = U + dt * L(U)
    auto R1 = compute_rhs(U, Nx, Ny, dx, dy, gamma, glm, ch);
    std::vector<std::vector<double>> Us(NC, std::vector<double>(NVAR));
    for (int c = 0; c < NC; ++c)
        for (int k = 0; k < NVAR; ++k)
            Us[c][k] = U[c][k] + dt * R1[c][k];

    // Stage 2: U^{n+1} = 0.5*(U + U* + dt*L(U*))
    auto R2 = compute_rhs(Us, Nx, Ny, dx, dy, gamma, glm, ch);
    for (int c = 0; c < NC; ++c)
        for (int k = 0; k < NVAR; ++k)
            U[c][k] = 0.5 * (U[c][k] + Us[c][k] + dt * R2[c][k]);

    // GLM: exact exponential damping (Dedner 2002 eq. 24, cp = ch/sqrt(2))
    if (glm) {
        double decay = std::exp(-2.0 * dt);  // -dt * ch²/cp² = -dt*2
        for (int c = 0; c < NC; ++c)
            U[c][IPSI] *= decay;
    }
}

// -------------------------------------------------------------------------
//  Single simulation run
// -------------------------------------------------------------------------
struct OTConfig {
    bool        glm;
    std::string outfile;
};

static void simulate(const OTConfig& cfg,
                     int N, double gamma, double t_end, double cfl)
{
    const int    Nx = N, Ny = N;
    const double dx = 1.0 / Nx, dy = 1.0 / Ny;

    // Orszag-Tang initial conditions
    const double B0   = 1.0 / std::sqrt(4.0 * M_PI);
    const double rho0 = 25.0 / (36.0 * M_PI);
    const double p0   = 5.0  / (12.0 * M_PI);

    int NC = Nx * Ny;
    std::vector<std::vector<double>> U(NC, std::vector<double>(NVAR, 0.0));
    for (int i = 0; i < Nx; ++i) {
        double x = (i + 0.5) * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = (j + 0.5) * dy;
            PrimState W(rho0,
                        -std::sin(2.0*M_PI*y),
                         std::sin(2.0*M_PI*x),
                        0.0, p0,
                        -B0 * std::sin(2.0*M_PI*y),
                         B0 * std::sin(4.0*M_PI*x),
                        0.0);
            U[idx2(i, j, Ny)] = W.to_conserved(gamma);
        }
    }

    // GLM wave speed: max initial signal speed
    double ch = max_signal(U, gamma, false, 0.0);

    double t = 0.0;
    int    step = 0;
    while (t < t_end - 1e-12) {
        double sp = max_signal(U, gamma, cfg.glm, ch);
        double dt = std::min(cfl * std::min(dx, dy) / sp, t_end - t);
        rk2_step(U, Nx, Ny, dx, dy, gamma, cfg.glm, ch, dt);
        t += dt;
        ++step;
        if (step % 200 == 0)
            std::cout << "   step=" << step << "  t=" << std::fixed
                      << std::setprecision(4) << t << "\n";
    }
    std::cout << "  Done: " << step << " steps, t=" << t << "\n";

    // Write CSV: outer loop i (x), inner loop j (y)
    std::filesystem::create_directories(
        std::filesystem::path(cfg.outfile).parent_path());
    std::ofstream out(cfg.outfile);
    out << "x,y,rho,u,v,w,p,Bx,By,Bz,psi,divB\n";
    out << std::scientific << std::setprecision(8);
    for (int i = 0; i < Nx; ++i) {
        double x = (i + 0.5) * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = (j + 0.5) * dy;
            PrimState W = PrimState::from_conserved(U[idx2(i, j, Ny)], gamma);
            // div(B) via centered finite differences (periodic)
            double Bxp = U[idx2((i+1)%Nx,      j,      Ny)][IB1];
            double Bxm = U[idx2((i-1+Nx)%Nx,   j,      Ny)][IB1];
            double Byp = U[idx2(i,              (j+1)%Ny,  Ny)][IB2];
            double Bym = U[idx2(i,              (j-1+Ny)%Ny, Ny)][IB2];
            double divB = (Bxp - Bxm) / (2.0*dx) + (Byp - Bym) / (2.0*dy);
            out << x     << "," << y     << ","
                << W.rho << "," << W.u   << "," << W.v   << "," << W.w << ","
                << W.p   << "," << W.Bx  << "," << W.By  << "," << W.Bz << ","
                << W.psi << "," << divB  << "\n";
        }
    }
}

// -------------------------------------------------------------------------
//  Main
// -------------------------------------------------------------------------
int main()
{
    const int    N     = 128;
    const double gamma = 5.0 / 3.0;
    const double t_end = 0.5;
    const double cfl   = 0.4;

    const std::string sep(58, '=');
    std::cout << sep << "\n";
    std::cout << " Orszag-Tang Vortex — 2D HLLD + GLM Comparison\n";
    std::cout << " N=" << N << "x" << N
              << "  gamma=" << gamma
              << "  t_end=" << t_end
              << "  CFL=" << cfl << "\n";
    std::cout << sep << "\n";

    std::vector<OTConfig> runs = {
        {false, "results/orszag_tang/noGLM.csv"},
        {true,  "results/orszag_tang/withGLM.csv"},
    };
    std::vector<std::string> labels = {
        "No GLM  (HLLD only)",
        "With GLM (HLLD + Dedner 2002)",
    };

    for (int r = 0; r < (int)runs.size(); ++r) {
        std::cout << " [run] " << labels[r] << "\n";
        simulate(runs[r], N, gamma, t_end, cfl);
        std::cout << "        -> " << runs[r].outfile << "\n";
    }

    std::cout << sep << "\n";
    return 0;
}
