#include "diagnostics.hpp"
#include "glm2d_common.hpp"

#include <algorithm>
#include <cmath>

double compute_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
) {
    const int ip = (i + 1) % nx;
    const int im = (i - 1 + nx) % nx;
    const int jp = (j + 1) % ny;
    const int jm = (j - 1 + ny) % ny;

    const double dBx_dx =
        (U[idx2d(ip, j, nx)][BX] - U[idx2d(im, j, nx)][BX]) / (2.0 * dx);

    const double dBy_dy =
        (U[idx2d(i, jp, nx)][BY] - U[idx2d(i, jm, nx)][BY]) / (2.0 * dy);

    return dBx_dx + dBy_dy;
}

double compute_normalized_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
) {
    const State& cell = U[idx2d(i, j, nx)];

    const double divB = compute_divB_cell_2d(U, nx, ny, i, j, dx, dy);

    const double Bmag = std::sqrt(
        cell[BX] * cell[BX]
      + cell[BY] * cell[BY]
      + cell[BZ] * cell[BZ]
    );

    const double dx_eff = std::min(dx, dy);
    return dx_eff * std::abs(divB) / (Bmag + DIVB_NORM_EPS);
}

DivBNorms compute_divB_norms_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
) {
    DivBNorms norms;

    const int N = nx * ny;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double divB =
                compute_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            const double normDivB =
                compute_normalized_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            const double abs_divB = std::abs(divB);

            norms.L1 += abs_divB;
            norms.L2 += divB * divB;
            norms.Linf = std::max(norms.Linf, abs_divB);

            norms.L1_norm += normDivB;
            norms.L2_norm += normDivB * normDivB;
            norms.Linf_norm = std::max(norms.Linf_norm, normDivB);
        }
    }

    norms.L1 /= static_cast<double>(N);
    norms.L2 = std::sqrt(norms.L2 / static_cast<double>(N));

    norms.L1_norm /= static_cast<double>(N);
    norms.L2_norm = std::sqrt(norms.L2_norm / static_cast<double>(N));

    return norms;
}
