#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "eglm2d.hpp"
#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_common.hpp"

namespace {

bool approx_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

void test_eglm_source_one_step() {
    GLM2DParams params;
    params.nx = 4;
    params.ny = 4;
    params.dx = 1.0;
    params.dy = 1.0;
    params.dt = 0.1;

    const int nx = params.nx;
    const int ny = params.ny;

    std::vector<State> Uold(nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            State cell{};
            cell.fill(0.0);

            cell[RHO] = 1.0 + 0.1 * i + 0.05 * j;
            cell[MX] = 0.3 + 0.02 * i;
            cell[MY] = -0.4 + 0.03 * j;
            cell[MZ] = 0.2 + 0.01 * (i + j);

            cell[BX] = 1.0 + 0.2 * i + 0.1 * j;
            cell[BY] = -0.5 + 0.3 * i - 0.05 * j;
            cell[BZ] = 0.25 + 0.07 * i + 0.11 * j;

            cell[E] = 2.0 + 0.2 * i + 0.15 * j;
            cell[PSI] = 0.4 * i * i - 0.2 * j + 0.1 * i * j;

            Uold[idx2d(i, j, nx)] = cell;
        }
    }

    std::vector<State> U = Uold;

    apply_eglm_source_2d(U, Uold, params);

    const std::vector<double> divB =
        compute_fv_divB_field_2d(Uold, nx, ny, params.dx, params.dy);

    const int i = 2;
    const int j = 1;
    const int id = idx2d(i, j, nx);

    const int ip = periodic_index(i + 1, nx);
    const int im = periodic_index(i - 1, nx);
    const int jp = periodic_index(j + 1, ny);
    const int jm = periodic_index(j - 1, ny);

    const double dpsidx =
        (Uold[idx2d(ip, j, nx)][PSI] - Uold[idx2d(im, j, nx)][PSI])
      / (2.0 * params.dx);

    const double dpsidy =
        (Uold[idx2d(i, jp, nx)][PSI] - Uold[idx2d(i, jm, nx)][PSI])
      / (2.0 * params.dy);

    const double Bx = Uold[id][BX];
    const double By = Uold[id][BY];
    const double Bz = Uold[id][BZ];

    assert(approx_equal(U[id][MX], Uold[id][MX] - params.dt * divB[id] * Bx));
    assert(approx_equal(U[id][MY], Uold[id][MY] - params.dt * divB[id] * By));
    assert(approx_equal(U[id][MZ], Uold[id][MZ] - params.dt * divB[id] * Bz));
    assert(approx_equal(
        U[id][E],
        Uold[id][E] - params.dt * (Bx * dpsidx + By * dpsidy)
    ));

    assert(approx_equal(U[id][BX], Uold[id][BX]));
    assert(approx_equal(U[id][BY], Uold[id][BY]));
    assert(approx_equal(U[id][BZ], Uold[id][BZ]));
    assert(approx_equal(U[id][PSI], Uold[id][PSI]));

    std::cout << "EGLM one-step source assertion passed.\n";
}

} // namespace

int main() {
    test_eglm_source_one_step();

    GLM2DParams params;

    // ============================================================
    // Hard-coded 2D GLM test configuration
    // ============================================================

    params.nx = 128;
    params.ny = 128;

    params.xlen = 1.0;
    params.ylen = 1.0;

    params.t_end = 0.5;

    params.ch = 1.0;
    params.cp = 0.2;

    params.cfl = 0.25;

    params.powell_vx = 1.0;
    params.powell_vy = 0.5;

    params.poisson_max_iter = 10000;
    params.poisson_tol = 1.0e-10;

    params.write_snapshot = true;
    params.out_prefix = "glm_2d";

    // ============================================================
    // Select cleaning methods to run
    // ============================================================

    const std::vector<CleaningType> cases = {
        CleaningType::NONE,
        CleaningType::HYPERBOLIC_GLM,
        CleaningType::MIXED_GLM,
        CleaningType::PARABOLIC,
        CleaningType::ELLIPTIC_PROJECTION,
        CleaningType::POWELL_SOURCE,
        CleaningType::MIXED_EGLM
    };

    // ============================================================
    // Run all selected cases
    // ============================================================

    for (CleaningType type : cases) {
        std::cout << "========================================\n";
        std::cout << "Running 2D GLM case: "
                  << cleaning_name(type)
                  << "\n";
        std::cout << "========================================\n";

        run_glm_2d_case(type, params);
    }

    std::cout << "All 2D GLM cleaning cases finished.\n";

    return 0;
}
