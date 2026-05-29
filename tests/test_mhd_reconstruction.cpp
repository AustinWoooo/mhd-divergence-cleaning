// =============================================================================
//  tests/test_mhd_reconstruction.cpp
//
//  Unit tests for the high-resolution shock-capturing reconstruction used by
//  the HLLD finite-volume update (include/mhd_reconstruction.hpp).
//
//  Properties verified:
//    1. Limiters are exact on linear data (recover the underlying slope).
//    2. Limiters vanish at local extrema (TVD / monotonicity).
//    3. Reconstructed face values are bounded by the neighbouring cell
//       averages, which guarantees positivity of reconstructed rho and p.
//    4. PCM reconstruction returns the cell averages unchanged.
// =============================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mhd_reconstruction.hpp"

using namespace MHD;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

void check_close(double a, double b, double tol, const std::string& msg) {
    if (!(std::abs(a - b) <= tol)) {
        std::cerr << "FAIL: " << msg << "  (got " << a << ", expected " << b
                  << ", tol " << tol << ")\n";
        ++g_failures;
    }
}

const SlopeLimiter ALL_LIMITERS[] = {
    SlopeLimiter::MINMOD,
    SlopeLimiter::VANLEER,
    SlopeLimiter::MC
};

// On smooth linear data with equal one-sided differences d, every TVD limiter
// must recover the full slope d (second-order accuracy in smooth regions).
void test_linear_exactness() {
    const double d = 0.37;
    for (SlopeLimiter lim : ALL_LIMITERS) {
        const double s = limited_slope(d, d, lim);
        check_close(s, d, 1.0e-14, "limiter not exact on linear data");
    }
}

// At a local extremum the one-sided differences have opposite signs; every TVD
// limiter must return zero slope so no new extremum is created.
void test_extremum_clipping() {
    for (SlopeLimiter lim : ALL_LIMITERS) {
        check_close(limited_slope(+1.0, -1.0, lim), 0.0, 0.0,
                    "limiter does not clip a maximum");
        check_close(limited_slope(-1.0, +1.0, lim), 0.0, 0.0,
                    "limiter does not clip a minimum");
        check_close(limited_slope(0.0, 1.0, lim), 0.0, 0.0,
                    "limiter does not clip a flat-left stencil");
    }
}

// The reconstructed face value W +/- 0.5*slope must stay within the range of
// the cell averages, which is what protects positivity of rho and p.
void test_face_values_bounded() {
    const double samples[] = {-3.0, -0.5, 0.0, 0.4, 1.0, 2.5, 10.0};
    for (double wm : samples) {
        for (double w0 : samples) {
            for (double wp : samples) {
                for (SlopeLimiter lim : ALL_LIMITERS) {
                    const double s = limited_slope(w0 - wm, wp - w0, lim);
                    const double face_r = w0 + 0.5 * s;  // right face
                    const double face_l = w0 - 0.5 * s;  // left face
                    const double lo = std::min(std::min(wm, w0), wp);
                    const double hi = std::max(std::max(wm, w0), wp);
                    const double eps = 1.0e-12 * (1.0 + std::abs(hi));
                    check(face_r >= lo - eps && face_r <= hi + eps,
                          "right face value out of neighbour bounds");
                    check(face_l >= lo - eps && face_l <= hi + eps,
                          "left face value out of neighbour bounds");
                }
            }
        }
    }
}

// Positive cell averages must yield positive reconstructed face density/pressure
// across a full primitive reconstruction.
void test_primitive_positivity() {
    const PrimState Wm(0.8, -0.2, 0.1, 0.0, 0.5,  0.3, -0.1, 0.0);
    const PrimState W0(1.0,  0.0, 0.0, 0.0, 1.0,  0.2,  0.0, 0.0);
    const PrimState Wp(1.3,  0.4, 0.2, 0.0, 1.7, -0.1,  0.2, 0.0);

    for (SlopeLimiter lim : ALL_LIMITERS) {
        const PrimState s = limited_slope_prim(Wm, W0, Wp, lim);
        const PrimState fR = reconstruct_face(W0, s, +1.0);
        const PrimState fL = reconstruct_face(W0, s, -1.0);
        check(fR.rho > 0.0 && fL.rho > 0.0,
              "reconstructed density non-positive");
        check(fR.p > 0.0 && fL.p > 0.0,
              "reconstructed pressure non-positive");
    }
}

// PCM (zero slope) must reproduce the cell average on both faces.
void test_pcm_is_identity() {
    const PrimState W0(1.0, 0.3, -0.2, 0.1, 2.0, 0.5, 0.4, -0.3, 0.0);
    const PrimState zero;  // default-constructed: all zeros
    const PrimState fR = reconstruct_face(W0, zero, +1.0);
    const PrimState fL = reconstruct_face(W0, zero, -1.0);
    check_close(fR.rho, W0.rho, 0.0, "PCM altered density (right)");
    check_close(fL.p,   W0.p,   0.0, "PCM altered pressure (left)");
    check_close(fR.By,  W0.By,  0.0, "PCM altered By (right)");
}

} // namespace

int main() {
    test_linear_exactness();
    test_extremum_clipping();
    test_face_values_bounded();
    test_primitive_positivity();
    test_pcm_is_identity();

    if (g_failures == 0) {
        std::cout << "test_mhd_reconstruction: all checks passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << "test_mhd_reconstruction: " << g_failures
              << " check(s) failed.\n";
    return EXIT_FAILURE;
}
