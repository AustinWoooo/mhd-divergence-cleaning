// =============================================================================
//  include/mhd_reconstruction.hpp
//  High-resolution shock-capturing spatial reconstruction for the 2D HLLD
//  finite-volume update.
//
//  The HLLD Riemann solver (HLLD_mhd_solver.hpp) only defines the numerical flux
//  given a left/right interface state. On its own, feeding it the cell averages
//  gives a first-order Godunov scheme. To obtain a genuine high-resolution
//  shock-capturing (HRSC) scheme we reconstruct the primitive variables to the
//  cell faces with a piecewise-linear MUSCL slope and a TVD slope limiter, then
//  hand the reconstructed left/right states to HLLD. Combined with the runner's
//  SSP-RK2 (Heun) time integration this is second-order in space and time and
//  retains monotonicity near shocks.
//
//  Positivity: every limiter here is TVD, i.e. the limited slope is bounded so
//  that the face extrapolation W +/- 0.5*slope always lies within the range of
//  the cell and its upwind/downwind neighbour averages. Therefore, whenever the
//  cell-averaged density and pressure are positive, the reconstructed face
//  density and pressure are positive as well -- no extra clamping is required.
// =============================================================================

#pragma once

#include <algorithm>
#include <cmath>

#include "HLLD_mhd_solver.hpp"

namespace MHD {

// Spatial reconstruction order used to build interface states.
enum class Reconstruction {
    PCM,  // piecewise-constant (cell averages)  -> first-order Godunov
    PLM   // piecewise-linear MUSCL              -> second-order HRSC
};

// TVD slope limiters, in increasing order of sharpness (and decreasing
// dissipation): MINMOD < VANLEER < MC (monotonized central).
enum class SlopeLimiter {
    MINMOD,
    VANLEER,
    MC
};

// -----------------------------------------------------------------------------
//  Scalar limited slope.
//
//  Inputs are the one-sided undivided differences
//      dL = W_i - W_{i-1},   dR = W_{i+1} - W_i
//  and the return value is the limited *cell-centred* slope s (such that the
//  reconstruction at the cell faces is W_i +/- 0.5*s). At a local extremum
//  (dL*dR <= 0) every TVD limiter returns zero slope, dropping the cell to
//  first order and guaranteeing no new extrema are created.
// -----------------------------------------------------------------------------
inline double limited_slope(double dL, double dR, SlopeLimiter lim) {
    if (!(dL * dR > 0.0)) {
        return 0.0;  // local extremum (or a zero/non-finite difference)
    }
    switch (lim) {
        case SlopeLimiter::MINMOD:
            return (std::abs(dL) < std::abs(dR)) ? dL : dR;
        case SlopeLimiter::VANLEER:
            // Harmonic mean of the one-sided slopes: 2 dL dR / (dL + dR).
            // dL, dR share a sign here, so the denominator cannot vanish.
            return 2.0 * dL * dR / (dL + dR);
        case SlopeLimiter::MC: {
            // Monotonized central: min of the central slope and twice each
            // one-sided slope, carrying the common sign.
            const double central = 0.5 * (dL + dR);
            const double s = std::min(std::abs(central),
                                      std::min(2.0 * std::abs(dL),
                                               2.0 * std::abs(dR)));
            return std::copysign(s, central);
        }
    }
    return 0.0;
}

// Per-component limited slope of a primitive state from its two neighbours.
inline PrimState limited_slope_prim(const PrimState& Wm,
                                    const PrimState& W0,
                                    const PrimState& Wp,
                                    SlopeLimiter lim) {
    return PrimState(
        limited_slope(W0.rho - Wm.rho, Wp.rho - W0.rho, lim),
        limited_slope(W0.u   - Wm.u,   Wp.u   - W0.u,   lim),
        limited_slope(W0.v   - Wm.v,   Wp.v   - W0.v,   lim),
        limited_slope(W0.w   - Wm.w,   Wp.w   - W0.w,   lim),
        limited_slope(W0.p   - Wm.p,   Wp.p   - W0.p,   lim),
        limited_slope(W0.Bx  - Wm.Bx,  Wp.Bx  - W0.Bx,  lim),
        limited_slope(W0.By  - Wm.By,  Wp.By  - W0.By,  lim),
        limited_slope(W0.Bz  - Wm.Bz,  Wp.Bz  - W0.Bz,  lim),
        limited_slope(W0.psi - Wm.psi, Wp.psi - W0.psi, lim)
    );
}

// Extrapolate a cell's primitive state to one of its faces.
//   sign = +1 -> right/top face value  (W + 0.5*slope)
//   sign = -1 -> left/bottom face value (W - 0.5*slope)
// With a zero slope (PCM) this returns the cell average unchanged.
inline PrimState reconstruct_face(const PrimState& W,
                                  const PrimState& slope,
                                  double sign) {
    const double h = 0.5 * sign;
    return PrimState(
        W.rho + h * slope.rho,
        W.u   + h * slope.u,
        W.v   + h * slope.v,
        W.w   + h * slope.w,
        W.p   + h * slope.p,
        W.Bx  + h * slope.Bx,
        W.By  + h * slope.By,
        W.Bz  + h * slope.Bz,
        W.psi + h * slope.psi
    );
}

} // namespace MHD
