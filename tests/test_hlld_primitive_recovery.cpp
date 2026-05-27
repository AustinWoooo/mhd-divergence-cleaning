#include <cassert>
#include <cmath>
#include <iostream>

#include "HLLD_mhd_solver.hpp"

namespace {

bool approx_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

void test_raw_recovery_valid_state() {
    const double gamma = 5.0 / 3.0;

    MHD::PrimState W0(
        1.2,
        0.3,
       -0.2,
        0.1,
        0.7,
        0.4,
       -0.1,
        0.2,
        0.0
    );

    const State U = W0.to_conserved(gamma);
    const MHD::PrimState W = MHD::PrimState::from_conserved_raw(U, gamma);

    assert(approx_equal(W.rho, W0.rho));
    assert(approx_equal(W.u, W0.u));
    assert(approx_equal(W.v, W0.v));
    assert(approx_equal(W.w, W0.w));
    assert(approx_equal(W.p, W0.p));
    assert(W.p > 0.0);
}

void test_raw_recovery_keeps_negative_pressure() {
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
    const MHD::PrimState default_recovery =
        MHD::PrimState::from_conserved(U, gamma);

    assert(raw.p < 0.0);
    assert(raw.p != TINY_NUMBER);
    assert(default_recovery.p < 0.0);
    assert(approx_equal(default_recovery.p, raw.p));
}

void test_checked_recovery_reports_pressure_floor() {
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

    MHD::PrimitiveRecoveryStatus status;
    const MHD::PrimState checked =
        MHD::PrimState::from_conserved_checked(U, gamma, &status);

    assert(!status.valid);
    assert(!status.rho_floored);
    assert(status.pressure_floored);
    assert(approx_equal(status.raw_rho, 1.0));
    assert(status.raw_pressure < 0.0);
    assert(approx_equal(status.used_rho, 1.0));
    assert(approx_equal(status.used_pressure, TINY_NUMBER));
    assert(approx_equal(checked.p, TINY_NUMBER));
}

void test_diagnostics_contract_uses_raw_default() {
    const double gamma = 5.0 / 3.0;

    State U{};
    U[RHO] = 1.0;
    U[BX] = 1.0;
    U[E] = 0.1;

    const MHD::PrimState W = MHD::PrimState::from_conserved(U, gamma);

    // mhd_runner diagnostics call the raw recovery path through state_to_prim.
    // This assertion protects that contract by requiring the default recovery
    // to expose, not hide, negative pressure.
    assert(W.p < 0.0);
}

// Task B: verify the pressure-preserving energy correction formula
//
//   E_new = E_old + 0.5 * (|B_new|^2 - |B_old|^2)
//
// After this correction, raw pressure must be identical to the pre-B-change
// value, to floating-point roundoff.  The formula is:
//
//   p_new = (γ-1)(E_new - KE - ME_new)
//         = (γ-1)(E_old + ME_new - ME_old - KE - ME_new)
//         = (γ-1)(E_old - KE - ME_old) = p_old.   QED
void test_pressure_preserving_B_correction() {
    const double gamma = 5.0 / 3.0;

    // Construct a cell with known positive pressure.
    MHD::PrimState W0(
        1.3,    // rho
        0.4,    // u
       -0.2,    // v
        0.1,    // w
        0.85,   // p   (positive)
        0.5,    // Bx
       -0.3,    // By
        0.2,    // Bz
        0.0     // psi
    );

    State Uold = W0.to_conserved(gamma);

    // Verify initial pressure is what we set.
    {
        const MHD::PrimState Wcheck =
            MHD::PrimState::from_conserved_raw(Uold, gamma);
        assert(approx_equal(Wcheck.p, W0.p, 1.0e-12));
        assert(Wcheck.p > 0.0);
    }

    // Change B (simulate what parabolic or projection cleaning does to B).
    State U = Uold;
    U[BX] = 0.8;   // was 0.5
    U[BY] = 0.1;   // was -0.3
    U[BZ] = -0.4;  // was 0.2
    // Note: density, momenta, PSI are unchanged.

    // Apply pressure-preserving energy correction:
    //   E_new = E_old + 0.5 * (|B_new|^2 - |B_old|^2)
    const double old_me =
        0.5 * (Uold[BX]*Uold[BX] + Uold[BY]*Uold[BY] + Uold[BZ]*Uold[BZ]);
    const double new_me =
        0.5 * (U[BX]*U[BX] + U[BY]*U[BY] + U[BZ]*U[BZ]);
    U[E] = Uold[E] + (new_me - old_me);

    // Verify raw pressure is unchanged to roundoff.
    const MHD::PrimState W_new =
        MHD::PrimState::from_conserved_raw(U, gamma);

    // Tolerance: a few ulps.  The formula involves only adds/subtracts so
    // error is O(eps_machine * |p|).
    const double tol = 1.0e-12;
    assert(approx_equal(W_new.p, W0.p, tol));
    assert(W_new.p > 0.0);

    // Verify rho and velocities are unchanged (B update should not touch them).
    assert(approx_equal(W_new.rho, W0.rho));
    assert(approx_equal(W_new.u, W0.u));
    assert(approx_equal(W_new.v, W0.v));
    assert(approx_equal(W_new.w, W0.w));

    // Verify B was actually changed.
    assert(!approx_equal(W_new.Bx, W0.Bx));
    assert(!approx_equal(W_new.By, W0.By));
}

// Also test with a cell that has large magnetic energy relative to thermal.
// This exercises the near-cancellation regime where the formula must be exact.
void test_pressure_preserving_high_Bmag() {
    const double gamma = 5.0 / 3.0;

    MHD::PrimState W0(
        1.0,
        0.0, 0.0, 0.0,
        0.05,   // low thermal pressure
        5.0,    // large Bx (ME = 12.5)
        0.0,
        0.0,
        0.0
    );

    State Uold = W0.to_conserved(gamma);

    // Slightly perturb B — as a projection might do.
    State U = Uold;
    U[BX] = 5.02;  // small correction to large B

    const double old_me =
        0.5 * (Uold[BX]*Uold[BX] + Uold[BY]*Uold[BY] + Uold[BZ]*Uold[BZ]);
    const double new_me =
        0.5 * (U[BX]*U[BX] + U[BY]*U[BY] + U[BZ]*U[BZ]);
    U[E] = Uold[E] + (new_me - old_me);

    const MHD::PrimState W_new =
        MHD::PrimState::from_conserved_raw(U, gamma);

    // Tolerance: O(eps_machine * |ME|) because large ME means larger roundoff
    // in E relative to p.
    const double tol = 1.0e-10;
    assert(approx_equal(W_new.p, W0.p, tol));
    assert(W_new.p > 0.0);
}

} // namespace

int main() {
    test_raw_recovery_valid_state();
    test_raw_recovery_keeps_negative_pressure();
    test_checked_recovery_reports_pressure_floor();
    test_diagnostics_contract_uses_raw_default();
    test_pressure_preserving_B_correction();
    test_pressure_preserving_high_Bmag();

    std::cout << "HLLD primitive recovery assertions passed.\n";
    return 0;
}
