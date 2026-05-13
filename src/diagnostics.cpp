#include "diagnostics.hpp"

#include <cmath>
#include <algorithm>

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
        int jp = (j + 1) % ny;
        int jm = (j - 1 + ny) % ny;

        for (int i = 0; i < nx; ++i) {
            int ip = (i + 1) % nx;
            int im = (i - 1 + nx) % nx;

            const double dBx_dx =
                (U[idx2d(ip, j, nx)][BX] - U[idx2d(im, j, nx)][BX]) / (2.0 * dx);

            const double dBy_dy =
                (U[idx2d(i, jp, nx)][BY] - U[idx2d(i, jm, nx)][BY]) / (2.0 * dy);

            const double divB = dBx_dx + dBy_dy;
            const double abs_divB = std::abs(divB);

            norms.L1 += abs_divB;
            norms.L2 += divB * divB;
            norms.Linf = std::max(norms.Linf, abs_divB);
        }
    }

    norms.L1 /= static_cast<double>(N);
    norms.L2 = std::sqrt(norms.L2 / static_cast<double>(N));

    return norms;
}