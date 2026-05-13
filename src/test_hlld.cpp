// =============================================================================
//  main.cpp
//  Test driver for the HLLD Riemann Solver -- standard MHD shock-tube cases
//
//  Test cases:
//    1. Brio-Wu shock tube (Brio & Wu 1988) -- classic MHD benchmark, gamma=2
//    2. Rotation consistency -- same problem in the Y direction
//    3. Degenerate Bx=0 -- HLLD should reduce gracefully to HLLC-like
//    4. Supersonic flow  -- solver must return F_L unchanged
//    5. Identical L/R states -- HLLD flux must equal the exact physical flux
// =============================================================================

#include "../include/HLLD_mhd_solver.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace MHD;

// Print all 8 flux components with labels
void print_flux(const std::string& title, const std::vector<double>& F) {
    const std::vector<std::string> names = {
        "F[rho]    ", "F[rho*u]  ", "F[rho*v]  ", "F[rho*w]  ",
        "F[E]      ", "F[Bx]     ", "F[By]     ", "F[Bz]     "
    };
    std::cout << "\n--- " << title << " ---\n";
    std::cout << std::scientific << std::setprecision(8);
    for (int i = 0; i < NVAR; ++i) {
        std::cout << "  " << names[i] << " = " << std::setw(16) << F[i] << "\n";
    }
}

void print_state(const std::string& title, const State& W) {
    std::cout << "  " << title
              << ": rho=" << W.rho
              << ", u=" << W.u  << ", v=" << W.v  << ", w=" << W.w
              << ", p="  << W.p
              << ", Bx=" << W.Bx << ", By=" << W.By << ", Bz=" << W.Bz << "\n";
}

int main() {
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "=================================================\n";
    std::cout << " 2D Ideal MHD HLLD Riemann Solver - Test Driver\n";
    std::cout << "=================================================\n";

    // -------------------------------------------------------------------------
    //  Test 1: Brio-Wu shock tube (X-direction)
    //    Left : rho=1.0,   p=1.0,  Bx=0.75, By= 1.0
    //    Right: rho=0.125, p=0.1,  Bx=0.75, By=-1.0
    //    All velocities and Bz = 0.  gamma=2.0 (standard Brio-Wu setting)
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Test 1] Brio-Wu Shock Tube (X-direction, gamma=2.0)\n";
        State WL(1.0,   0.0, 0.0, 0.0,  1.0,  0.75,  1.0, 0.0);
        State WR(0.125, 0.0, 0.0, 0.0,  0.1,  0.75, -1.0, 0.0);
        print_state("L", WL);
        print_state("R", WR);

        double gamma = 2.0;
        auto F = compute_flux(WL, WR, /*direction=*/0, gamma);
        print_flux("Numerical Flux (X)", F);

        // F[Bx] must be zero: normal B is frozen (div B = 0)
        std::cout << "\n  Sanity check: F[Bx] should be 0  -> F[Bx] = "
                  << F[IB1] << "\n";
    }

    // -------------------------------------------------------------------------
    //  Test 2: Brio-Wu in the Y direction (rotation consistency check)
    //    Rotate Test 1 by 90 degrees: swap (u,Bx) <-> (v,By).
    //    F[rho] must match Test 1; normal-B flux F[By] must be zero.
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Test 2] Brio-Wu in Y-direction (rotation consistency)\n";
        // Test 1 rotated 90 deg: (u=0,v=0,Bx=0.75,By=±1) -> (v=0,u=0,By=0.75,Bx=∓1)
        State WL(1.0,   0.0, 0.0, 0.0,  1.0, -1.0, 0.75, 0.0);
        State WR(0.125, 0.0, 0.0, 0.0,  0.1,  1.0, 0.75, 0.0);
        print_state("L", WL);
        print_state("R", WR);

        double gamma = 2.0;
        auto F = compute_flux(WL, WR, /*direction=*/1, gamma);
        print_flux("Numerical Flux (Y)", F);

        std::cout << "\n  Sanity check: F[By] should be 0  -> F[By] = "
                  << F[IB2] << "\n";
        std::cout << "  Compare F[rho](Y) with F[rho](X) from Test 1 -- should match.\n";
    }

    // -------------------------------------------------------------------------
    //  Test 3: Supersonic flow to the right (SL >= 0 => return F_L)
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Test 3] Supersonic flow to the right (should return F_L)\n";
        State WL(1.0, 10.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0);
        State WR(1.0, 10.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0);
        double gamma = 5.0/3.0;
        auto F = compute_flux(WL, WR, 0, gamma);
        // Expected F[rho] = rho*u = 1*10 = 10
        print_flux("Numerical Flux", F);
        std::cout << "\n  Expected F[rho] = 10  -> got " << F[IDN] << "\n";
    }

    // -------------------------------------------------------------------------
    //  Test 4: Degenerate case -- Bx = 0, no Alfven wave
    //    HLLD must reduce to HLLC-like without producing NaN or Inf.
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Test 4] Degenerate case: Bx = 0 (no Alfven wave)\n";
        State WL(1.0,   0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        State WR(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.0);
        double gamma = 5.0/3.0;
        auto F = compute_flux(WL, WR, 0, gamma);
        print_flux("Numerical Flux", F);
        std::cout << "\n  HLLD should degrade gracefully without NaN.\n";
        bool nan_found = false;
        for (auto v : F) if (std::isnan(v) || std::isinf(v)) nan_found = true;
        std::cout << "  NaN/Inf check: "
                  << (nan_found ? "FAILED (NaN detected!)" : "PASSED") << "\n";
    }

    // -------------------------------------------------------------------------
    //  Test 5: Identical L/R states -- HLLD must reproduce the exact F(W)
    // -------------------------------------------------------------------------
    {
        std::cout << "\n[Test 5] Identical L/R states (should equal F(W))\n";
        State W(1.0, 0.5, -0.3, 0.2, 1.0, 0.4, 0.6, 0.1);
        double gamma = 5.0/3.0;
        auto F = compute_flux(W, W, 0, gamma);
        auto Fexact = physical_flux(W, gamma);
        print_flux("HLLD Flux", F);
        print_flux("Exact F(W)", Fexact);
        double max_diff = 0.0;
        for (int i = 0; i < NVAR; ++i)
            max_diff = std::max(max_diff, std::abs(F[i] - Fexact[i]));
        std::cout << "\n  Max |F_HLLD - F_exact| = " << max_diff << "\n";
        std::cout << "  Consistency check: "
                  << (max_diff < 1e-10 ? "PASSED" : "FAILED") << "\n";
    }

    std::cout << "\n=================================================\n";
    std::cout << " All tests completed.\n";
    std::cout << "=================================================\n";
    return 0;
}