// =============================================================================
//  src/main_glm.cpp
//  1D Ideal MHD simulation with GLM divergence cleaning via HLLD solver
//
//  This driver integrates the HLLD Riemann solver with GLM flux correction
//  and psi-damping source term for divergence-free magnetic field evolution.
//
//  Key features:
//    - 9 conserved variables: [rho, mx, my, mz, E, Bx, By, Bz, psi]
//    - GLM hyperbolic flux correction applied during Riemann solve
//    - Psi damping via mixed GLM (exponential decay, operator splitting)
//    - Output tracks divB norms for quantitative comparison
//
//  Build (from the project root):
//    cmake -B build && cd build && make -j4
//    ./main_glm
//
// =============================================================================

#include "../include/HLLD_mhd_solver.hpp"
#include "../include/glm.hpp"
#include "../include/state.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace MHD;

// =============================================================================
// Problem Setup: Brio-Wu shock tube with localized Bx bump
// =============================================================================

void initialize_brio_wu_glm(
    std::vector<State>& W,
    const std::vector<std::array<double, 9>>& U_init,
    double dx,
    double gamma
) {
    const int N = static_cast<int>(W.size());

    for (int i = 0; i < N; ++i) {
        // Convert conserved to primitive
        W[i] = State::from_conserved(
            std::vector<double>(U_init[i].begin(), U_init[i].end()),
            gamma
        );
    }
}

void initialize_conserved_with_glm(
    std::vector<std::array<double, 9>>& U,
    double dx
) {
    const int N = static_cast<int>(U.size());

    for (int i = 0; i < N; ++i) {
        double x = (i + 0.5) * dx;

        // Initialize all 9 components
        // Variables: [rho, mx, my, mz, E, Bx, By, Bz, psi]
        U[i].fill(0.0);

        // Density and energy (uniform)
        U[i][IDN] = 1.0;
        U[i][IEN] = 1.0;

        // Localized Bx bump: creates div(B) = dBx/dx
        // This bump will be cleaned by the GLM algorithm
        double Bx = 1.0 + 0.1 * std::exp(-200.0 * (x - 0.5) * (x - 0.5));
        U[i][IB1] = Bx;

        // Other field components and velocities are zero
        U[i][IM1] = 0.0;
        U[i][IM2] = 0.0;
        U[i][IM3] = 0.0;
        U[i][IB2] = 0.0;
        U[i][IB3] = 0.0;

        // GLM scalar: initialized to zero
        U[i][IPSI] = 0.0;
    }
}

// =============================================================================
// Diagnostics: compute divB norms
// =============================================================================

double compute_divB_cell(
    const std::vector<std::array<double, 9>>& U,
    int i,
    double dx
) {
    const int N = static_cast<int>(U.size());
    const int im = (i - 1 + N) % N;
    const int ip = (i + 1) % N;

    // Central difference: divB = dBx/dx
    return (U[ip][IB1] - U[im][IB1]) / (2.0 * dx);
}

void compute_divB_norms(
    const std::vector<std::array<double, 9>>& U,
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
        double divB = compute_divB_cell(U, i, dx);
        L1 += std::abs(divB);
        L2 += divB * divB;
        Linf = std::max(Linf, std::abs(divB));
    }

    L1 /= static_cast<double>(N);
    L2 = std::sqrt(L2 / static_cast<double>(N));
}

// =============================================================================
// Time stepping: full MHD+GLM update
//
// Per timestep, we:
//  1. Compute HLLD flux with GLM correction
//  2. Apply conserved update via flux divergence
//  3. Apply GLM psi-damping source term
//
// =============================================================================

void step_mhd_glm_1d(
    std::vector<std::array<double, 9>>& U,
    double dx,
    double dt,
    double gamma,
    double ch,
    double cp
) {
    const int N = static_cast<int>(U.size());

    // =========================================================================
    // Step 1: Convert conserved to primitive
    // =========================================================================
    std::vector<State> W(N);
    for (int i = 0; i < N; ++i) {
        W[i] = State::from_conserved(
            std::vector<double>(U[i].begin(), U[i].end()),
            gamma
        );
    }

    // =========================================================================
    // Step 2: Compute fluxes at all cell interfaces with GLM correction
    // =========================================================================
    std::vector<std::vector<double>> flux(N + 1, std::vector<double>(NVAR));

    for (int iface = 0; iface <= N; ++iface) {
        int iL = (iface - 1 + N) % N;  // left cell
        int iR = iface % N;            // right cell

        // HLLD flux (without GLM psi flux)
        // Note: HLLD returns F[IPSI] = 0.0
        flux[iface] = hlld_flux_normal(W[iL], W[iR], gamma);

        // Add GLM hyperbolic flux correction
        // GLM flux: F_Bn = psi, F_psi = ch^2 * Bn
        // Using Rusanov flux: F = 0.5*(F_L + F_R) - 0.5*ch*(U_R - U_L)

        double BnL = U[iL][IB1];
        double BnR = U[iR][IB1];
        double psiL = U[iL][IPSI];
        double psiR = U[iR][IPSI];

        // Hyperbolic GLM flux (2x2 subsystem)
        double FBn = 0.5 * (psiL + psiR) - 0.5 * ch * (BnR - BnL);
        double Fpsi = 0.5 * (ch * ch * BnL + ch * ch * BnR)
                    - 0.5 * ch * (psiR - psiL);

        flux[iface][IB1] = FBn;
        flux[iface][IPSI] = Fpsi;
    }

    // =========================================================================
    // Step 3: Conservative update via flux divergence
    // =========================================================================
    std::vector<std::array<double, 9>> Unew = U;

    for (int i = 0; i < N; ++i) {
        int left_face  = i;
        int right_face = i + 1;

        for (int v = 0; v < NVAR; ++v) {
            Unew[i][v] = U[i][v]
                       - (dt / dx) * (flux[right_face][v] - flux[left_face][v]);
        }
    }

    // =========================================================================
    // Step 4: Apply mixed GLM psi-damping source term (operator splitting)
    //
    // The psi damping is:  d(psi)/dt = -(ch^2 / cp^2) * psi
    // Exact solution:      psi(t+dt) = psi(t) * exp(-ch^2*dt / cp^2)
    // ==========================================================================
    double damping_factor = std::exp(-ch * ch * dt / (cp * cp));

    for (int i = 0; i < N; ++i) {
        Unew[i][IPSI] *= damping_factor;
    }

    // Update state
    U = Unew;
}

// =============================================================================
// Main: 1D MHD-GLM test problem
// =============================================================================

int main() {
    const std::string sep(70, '=');

    std::cout << sep << "\n";
    std::cout << " 1D Ideal MHD with GLM Divergence Cleaning\n";
    std::cout << " Solver  : HLLD Riemann Solver (Miyoshi & Kusano 2005)\n";
    std::cout << " Cleaning: Generalized Lagrange Multiplier (GLM) method\n";
    std::cout << " Problem : Brio-Wu shock tube with localized Bx bump\n";
    std::cout << sep << "\n";

    // =========================================================================
    // Grid and Physical Parameters
    // =========================================================================
    const int N = 400;                   // number of cells
    const double xlen = 1.0;             // domain length
    const double dx = xlen / N;          // cell spacing

    const double gamma = 5.0 / 3.0;      // adiabatic index

    // =========================================================================
    // GLM Parameters (mixed GLM: hyperbolic + parabolic damping)
    // =========================================================================
    const double ch = 1.0;               // hyperbolic cleaning speed
    const double cp = 0.2;               // damping parameter

    // CFL-based timestep
    // For GLM hyperbolic: dt_cfl = CFL * dx / ch
    const double CFL = 0.4;
    const double dt = CFL * dx / ch;

    const double t_end = 0.5;
    const int nstep = static_cast<int>(std::ceil(t_end / dt));

    std::cout << " Grid parameters:\n";
    std::cout << "   N        = " << N << " cells\n";
    std::cout << "   dx       = " << std::scientific << std::setprecision(3) << dx << "\n";
    std::cout << " GLM parameters:\n";
    std::cout << "   ch       = " << ch << "\n";
    std::cout << "   cp       = " << cp << "\n";
    std::cout << "   dt       = " << dt << " (CFL = " << CFL << ")\n";
    std::cout << "   nstep    = " << nstep << "\n";
    std::cout << "   t_end    = " << std::fixed << std::setprecision(3) << t_end << "\n";
    std::cout << sep << "\n";

    // =========================================================================
    // Initialize: conserved variables (9 components, PSI = 0)
    // =========================================================================
    std::vector<std::array<double, 9>> U(N);
    initialize_conserved_with_glm(U, dx);

    // =========================================================================
    // Output file: divB diagnostics at each timestep
    // =========================================================================
    const std::string outdir = "results/divergence";
    const std::string outname = outdir + "/glm_1d_mixed_glm.csv";

    std::ofstream fout(outname);
    if (!fout) {
        std::cerr << "Error: could not open output file " << outname << "\n";
        return 1;
    }

    fout << std::scientific << std::setprecision(6);
    fout << "step,time,L1_divB,L2_divB,Linf_divB\n";

    // =========================================================================
    // Time integration loop
    // =========================================================================
    std::cout << " Running " << nstep << " steps...\n";

    for (int step = 0; step <= nstep; ++step) {
        double time = step * dt;

        // Compute and write divB norms
        double L1_divB, L2_divB, Linf_divB;
        compute_divB_norms(U, dx, L1_divB, L2_divB, Linf_divB);

        fout << step << "," << time << "," << L1_divB << ","
             << L2_divB << "," << Linf_divB << "\n";

        // Print progress every 20 steps
        if (step % 20 == 0) {
            std::cout << "   step " << std::setw(5) << step
                      << " / " << nstep
                      << " (t = " << std::fixed << std::setprecision(3) << time << ")"
                      << " L2(divB) = " << std::scientific << L2_divB << "\n";
        }

        if (step == nstep) break;

        // Perform one MHD+GLM timestep
        step_mhd_glm_1d(U, dx, dt, gamma, ch, cp);
    }

    fout.close();
    std::cout << " Wrote " << outname << "\n";
    std::cout << sep << "\n";
    std::cout << " Simulation completed successfully.\n";
    std::cout << sep << "\n";

    return 0;
}
