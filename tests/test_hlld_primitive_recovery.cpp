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

} // namespace

int main() {
    test_raw_recovery_valid_state();
    test_raw_recovery_keeps_negative_pressure();
    test_checked_recovery_reports_pressure_floor();
    test_diagnostics_contract_uses_raw_default();

    std::cout << "HLLD primitive recovery assertions passed.\n";
    return 0;
}
