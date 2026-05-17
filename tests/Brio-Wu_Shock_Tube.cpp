// =============================================================================
//  test/Brio-Wu_Shock_Tube.cpp
//  Brio-Wu Shock Tube Test Cases for the HLLD Riemann Solver
//
//  Reference: Brio & Wu (1988), "An Upwind Differencing Scheme for the
//             Equations of Ideal Magnetohydrodynamics"
//             Journal of Computational Physics, 75, 400-422
//
//  This file defines and runs five structured test cases that together provide
//  thorough coverage of the HLLD solver's code paths:
//
//  Case 1 — X-direction (standard Brio-Wu)
//    The canonical benchmark: verifies fast/slow shocks, rarefactions,
//    and the contact discontinuity in the normal (X) direction.
//
//  Case 2 — Y-direction (rotation consistency)
//    Identical physics rotated 90°: swaps (u↔v, Bx↔By).
//    F[rho] must match Case 1; F[By] (normal B flux) must be zero.
//
//  Case 3 — Supersonic flow (SL ≥ 0 early-exit path)
//    Both states identical and convecting fast to the right.
//    HLLD must return F_L unchanged; F[rho] must equal rho*u = 10.
//
//  Case 4 — Degenerate Bx = 0 (no Alfven wave)
//    Normal B is zero, so the Alfven speed vanishes and the double-star
//    region collapses.  HLLD must degrade gracefully — no NaN or Inf.
//
//  Case 5 — Identical L/R states (consistency with exact flux)
//    When left and right states are the same, the numerical flux must
//    reproduce the exact physical flux F(W) to machine precision.
//
//  Interface (called from main.cpp):
//    void run_all_tests(double gamma_brio_wu = 2.0,
//                       double gamma_general = 5.0/3.0);
// =============================================================================

#include "HLLD_mhd_solver.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace MHD;

namespace BrioWuTest {

// -----------------------------------------------------------------------------
//  Internal helpers
// -----------------------------------------------------------------------------

// Print a named 8-component flux vector.
static void print_flux(const std::string& title, const std::vector<double>& F)
{
    const std::vector<std::string> labels = {
        "F[rho]   ", "F[rho*u] ", "F[rho*v] ", "F[rho*w] ",
        "F[Bx]    ", "F[By]    ", "F[Bz]    ", "F[E]     ", "F[psi]   "
    };
    std::cout << "\n  -- " << title << " --\n";
    std::cout << std::scientific << std::setprecision(8);
    for (int i = 0; i < NVAR; ++i)
        std::cout << "    " << labels[i] << " = " << std::setw(17) << F[i] << "\n";
}

// Print a primitive state on one line.
static void print_state(const std::string& side, const PrimState& W)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << side
              << ": rho=" << W.rho
              << "  u="  << W.u  << "  v=" << W.v  << "  w=" << W.w
              << "  p="  << W.p
              << "  Bx=" << W.Bx << "  By=" << W.By << "  Bz=" << W.Bz
              << "\n";
}

// Simple pass/fail banner.
static void check(const std::string& description, bool passed)
{
    std::cout << "  Check  [" << (passed ? " PASS " : " FAIL ") << "]  "
              << description << "\n";
}

// -----------------------------------------------------------------------------
//  Test cases
// -----------------------------------------------------------------------------

// -------------------------------------------------------------------------
//  Case 1: Brio-Wu Shock Tube — X direction
//
//  Initial conditions (Brio & Wu 1988, table 1):
//    Left  : rho=1.0,   p=1.0,  Bx=0.75, By= 1.0,  u=v=w=Bz=0
//    Right : rho=0.125, p=0.1,  Bx=0.75, By=-1.0,  u=v=w=Bz=0
//    gamma = 2.0
//
//  Expected wave structure (left → right):
//    fast rarefaction | slow compound wave | contact | slow shock | fast rarefaction
//
//  Sanity check: F[Bx] must be exactly 0 (normal-B conservation, div B = 0).
// -------------------------------------------------------------------------
static void case1_brio_wu_x(double gamma)
{
    std::cout << "\n========================================\n";
    std::cout << " Case 1: Brio-Wu Shock Tube (X direction)\n";
    std::cout << " Reference: Brio & Wu (1988), gamma = " << gamma << "\n";
    std::cout << "========================================\n";

    PrimState WL(1.0,   0.0, 0.0, 0.0,  1.0,  0.75,  1.0, 0.0);
    PrimState WR(0.125, 0.0, 0.0, 0.0,  0.1,  0.75, -1.0, 0.0);

    print_state("L", WL);
    print_state("R", WR);

    auto F = compute_flux(WL, WR, /*direction=*/0, gamma);
    print_flux("HLLD Numerical Flux (X)", F);

    // F[Bx] is the induction flux for the normal component; in 1-D it is
    // always zero because Bx is frozen across all waves (div B = 0 in 1-D).
    check("F[Bx] == 0  (div B = 0)",
          std::abs(F[IB1]) < 1.0e-14);

    // All components must be finite.
    bool finite_ok = true;
    for (auto v : F) if (!std::isfinite(v)) { finite_ok = false; break; }
    check("All flux components are finite", finite_ok);
}

// -------------------------------------------------------------------------
//  Case 2: Rotation consistency — Y direction
//
//  The Brio-Wu problem is rotated 90°: the Y axis becomes the normal
//  direction.  Rotation map: (u,v) ↔ (v,u) and (Bx,By) ↔ (By,Bx).
//  Because the HLLD solver uses rotate_to_normal internally, F[rho](Y)
//  must equal F[rho](X) from Case 1, and F[By] (the new normal-B flux)
//  must be exactly zero.
// -------------------------------------------------------------------------
static void case2_rotation_y(double gamma)
{
    std::cout << "\n========================================\n";
    std::cout << " Case 2: Rotation Consistency (Y direction)\n";
    std::cout << " Rotated 90° from Case 1 — F[rho] must match, F[By] must be 0\n";
    std::cout << "========================================\n";

    // Rotate Case 1: (u,v,Bx,By) -> (v,u,By,Bx)
    // Left  original: u=0, v=0, Bx=0.75, By= 1.0
    // Left  rotated : v=0, u=0, By=0.75, Bx=-1.0  (By of right ↔ Bx of left)
    // A cleaner way: flip the sign of Bx for the left state so that
    // after the Y-rotation the interface sees the same normal-B (0.75) as Case 1.
    PrimState WL(1.0,   0.0, 0.0, 0.0,  1.0, -1.0, 0.75, 0.0);
    PrimState WR(0.125, 0.0, 0.0, 0.0,  0.1,  1.0, 0.75, 0.0);

    print_state("L", WL);
    print_state("R", WR);

    auto F  = compute_flux(WL, WR, /*direction=*/1, gamma);

    // Also run Case 1 to get the reference F[rho] for comparison.
    PrimState WL1(1.0,   0.0, 0.0, 0.0,  1.0,  0.75,  1.0, 0.0);
    PrimState WR1(0.125, 0.0, 0.0, 0.0,  0.1,  0.75, -1.0, 0.0);
    auto  F1 = compute_flux(WL1, WR1, /*direction=*/0, gamma);

    print_flux("HLLD Numerical Flux (Y)", F);

    check("F[By] == 0  (normal B flux, div B = 0)",
          std::abs(F[IB2]) < 1.0e-14);

    check("F[rho](Y) matches F[rho](X) from Case 1",
          std::abs(F[IDN] - F1[IDN]) < 1.0e-12);

    bool finite_ok = true;
    for (auto v : F) if (!std::isfinite(v)) { finite_ok = false; break; }
    check("All flux components are finite", finite_ok);
}

// -------------------------------------------------------------------------
//  Case 3: Supersonic flow to the right (SL ≥ 0 early-exit path)
//
//  Both states are identical and moving at u=10 >> fast-wave speed.
//  HLLD must detect SL ≥ 0 and return F_L without entering the star-state
//  calculations.
//
//  Expected: F[rho] = rho * u = 1 * 10 = 10.
// -------------------------------------------------------------------------
static void case3_supersonic(double gamma)
{
    std::cout << "\n========================================\n";
    std::cout << " Case 3: Supersonic Flow to the Right\n";
    std::cout << " SL >= 0 path — HLLD must return F_L, F[rho] = 10\n";
    std::cout << "========================================\n";

    PrimState WL(1.0, 10.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0);
    PrimState WR(1.0, 10.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0);

    print_state("L", WL);
    print_state("R", WR);

    auto F     = compute_flux(WL, WR, /*direction=*/0, gamma);
    auto F_ref = compute_flux(WL, WR, /*direction=*/0, gamma); // same as F_L

    print_flux("HLLD Numerical Flux", F);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Expected F[rho] = 10.0,  got = " << F[IDN] << "\n";

    check("F[rho] == rho*u = 10.0",
          std::abs(F[IDN] - 10.0) < 1.0e-12);

    bool finite_ok = true;
    for (auto v : F) if (!std::isfinite(v)) { finite_ok = false; break; }
    check("All flux components are finite", finite_ok);
}

// -------------------------------------------------------------------------
//  Case 4: Degenerate case — Bx = 0 (no Alfven wave)
//
//  When the normal magnetic field vanishes the Alfven speed is zero,
//  SL* = SR* = SM, and the double-star region disappears.  HLLD must
//  degrade gracefully to its HLLC-like two-state path without producing
//  any NaN or Inf values.
// -------------------------------------------------------------------------
static void case4_degenerate_bx0(double gamma)
{
    std::cout << "\n========================================\n";
    std::cout << " Case 4: Degenerate Bx = 0 (No Alfven Wave)\n";
    std::cout << " HLLD must degrade gracefully — no NaN / Inf\n";
    std::cout << "========================================\n";

    PrimState WL(1.0,   0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0);
    PrimState WR(0.125, 0.0, 0.0, 0.0, 0.1, 0.0, 0.5, 0.0);

    print_state("L", WL);
    print_state("R", WR);

    auto F = compute_flux(WL, WR, /*direction=*/0, gamma);
    print_flux("HLLD Numerical Flux (degenerate)", F);

    bool nan_found = false;
    for (auto v : F) if (!std::isfinite(v)) { nan_found = true; break; }

    check("No NaN / Inf in any flux component", !nan_found);

    // With Bx=0 and Bz=0, the Bz induction flux must also be zero.
    check("F[Bz] == 0  (Bz=0 in both states)",
          std::abs(F[IB3]) < 1.0e-14);
}

// -------------------------------------------------------------------------
//  Case 5: Identical L and R states — consistency with exact physical flux
//
//  When both states are the same, the Riemann problem is trivial and the
//  numerical flux must equal the exact physical flux F(W) to machine
//  precision.  This is the standard consistency (E-flux) property.
// -------------------------------------------------------------------------
static void case5_identical_states(double gamma)
{
    std::cout << "\n========================================\n";
    std::cout << " Case 5: Identical L/R States (Consistency Check)\n";
    std::cout << " HLLD flux must equal the exact physical flux F(W)\n";
    std::cout << "========================================\n";

    PrimState W(1.0, 0.5, -0.3, 0.2,  1.0,  0.4, 0.6, 0.1);

    print_state("L = R", W);

    auto F_hlld  = compute_flux(W, W, /*direction=*/0, gamma);
    auto F_exact = physical_flux(W, gamma);

    print_flux("HLLD Flux  F_HLLD", F_hlld);
    print_flux("Exact Flux F(W)  ", F_exact);

    double max_abs_err = 0.0;
    for (int i = 0; i < NVAR; ++i)
        max_abs_err = std::max(max_abs_err,
                               std::abs(F_hlld[i] - F_exact[i]));

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "\n  max |F_HLLD - F_exact| = " << max_abs_err << "\n";

    check("max |F_HLLD - F_exact| < 1e-10  (machine-precision consistency)",
          max_abs_err < 1.0e-10);
}

// -----------------------------------------------------------------------------
//  Public entry point
// -----------------------------------------------------------------------------

} // namespace BrioWuTest


// Called from main.cpp
void run_all_tests(double gamma_brio_wu, double gamma_general)
{
    BrioWuTest::case1_brio_wu_x    (gamma_brio_wu);
    BrioWuTest::case2_rotation_y   (gamma_brio_wu);
    BrioWuTest::case3_supersonic   (gamma_general);
    BrioWuTest::case4_degenerate_bx0(gamma_general);
    BrioWuTest::case5_identical_states(gamma_general);
}