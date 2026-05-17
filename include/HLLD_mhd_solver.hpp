// =============================================================================
//  include/HLLD_mhd_solver.hpp
//  2D Ideal MHD HLLD Approximate Riemann Solver
//
//  Reference: Miyoshi & Kusano, JCP 208 (2005) 315-344
//             "A multi-state HLL approximate Riemann solver for ideal
//              magnetohydrodynamics"
//
//  Design notes:
//    1. 2.5D layout -- spatial fluxes in X/Y only, but velocity and magnetic
//       field retain their Z components to correctly resolve Alfven waves.
//    2. Direction switching via index rotation: a single HLLD kernel handles
//       both X and Y fluxes (standard practice in Athena / PLUTO / FLASH).
//    3. Declarations only -- implementations are in src/hlld.cpp.
// =============================================================================

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "state.hpp"

namespace MHD {

// -----------------------------------------------------------------------------
//  Conservative variable index aliases
// -----------------------------------------------------------------------------
constexpr int IDN  = RHO;
constexpr int IM1  = MX;
constexpr int IM2  = MY;
constexpr int IM3  = MZ;
constexpr int IB1  = BX;
constexpr int IB2  = BY;
constexpr int IB3  = BZ;
constexpr int IEN  = E;
constexpr int IPSI = PSI;

// -----------------------------------------------------------------------------
//  PrimState -- primitive variables W = [rho, u, v, w, p, Bx, By, Bz, psi]
// -----------------------------------------------------------------------------
struct PrimState {
    double rho;
    double u, v, w;
    double p;
    double Bx, By, Bz;
    double psi;

    PrimState() : rho(0), u(0), v(0), w(0), p(0), Bx(0), By(0), Bz(0), psi(0) {}

    PrimState(double r, double uu, double vv, double ww,
              double pp, double bx, double by, double bz, double ps = 0.0)
        : rho(r), u(uu), v(vv), w(ww), p(pp), Bx(bx), By(by), Bz(bz), psi(ps) {}

    std::vector<double> to_conserved(double gamma) const;
    static PrimState from_conserved(const std::vector<double>& U, double gamma);
};

// -----------------------------------------------------------------------------
//  Free function declarations
// -----------------------------------------------------------------------------

PrimState rotate_to_normal(const PrimState& W, int direction);

std::vector<double> rotate_flux_back(const std::vector<double>& F, int direction);

double fast_magnetosonic_speed(const PrimState& W, double gamma);

std::vector<double> physical_flux(const PrimState& W, double gamma);

std::vector<double> hlld_flux_normal(const PrimState& WL, const PrimState& WR, double gamma);

// Public interface: rotate to normal frame -> HLLD -> rotate flux back.
// direction = 0 (X-flux) or 1 (Y-flux).
std::vector<double> compute_flux(const PrimState& W_L, const PrimState& W_R,
                                 int direction, double gamma);

} // namespace MHD
