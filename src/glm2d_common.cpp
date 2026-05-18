#include "glm2d_common.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int periodic_index(int i, int n) {
    int r = i % n;
    if (r < 0) {
        r += n;
    }
    return r;
}

void ensure_parent_directory(const std::string& filename) {
    const fs::path path(filename);
    const fs::path parent = path.parent_path();

    if (!parent.empty()) {
        fs::create_directories(parent);
    }
}

double analytic_divergence_pulse_2d(double x, double y) {
    const double dx0 = x - 0.5;
    const double dy0 = y - 0.5;

    const double r2 = dx0 * dx0 + dy0 * dy0;
    const double bump = std::exp(-100.0 * r2);

    // Initial field:
    //   Bx = 1 + 0.10 exp(-100 r^2)
    //   By =     0.05 exp(-100 r^2)
    //
    // Therefore:
    //   dBx/dx = -20 (x - 0.5) exp(-100 r^2)
    //   dBy/dy = -10 (y - 0.5) exp(-100 r^2)
    return (-20.0 * dx0 - 10.0 * dy0) * bump;
}

double compute_fv_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
) {
    const int im = periodic_index(i - 1, nx);
    const int jm = periodic_index(j - 1, ny);

    const State& C  = U[idx2d(i,  j,  nx)];
    const State& XL = U[idx2d(im, j,  nx)];
    const State& YL = U[idx2d(i,  jm, nx)];

    const double dBx_dx = (C[BX] - XL[BX]) / dx;
    const double dBy_dy = (C[BY] - YL[BY]) / dy;

    return dBx_dx + dBy_dy;
}

double compute_fv_normalized_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
) {
    const State& C = U[idx2d(i, j, nx)];

    const double divB =
        compute_fv_divB_cell_2d(U, nx, ny, i, j, dx, dy);

    const double Bmag = std::sqrt(
        C[BX] * C[BX]
      + C[BY] * C[BY]
      + C[BZ] * C[BZ]
    );

    const double h = std::min(dx, dy);
    const double eps = 1.0e-30;

    return std::abs(divB) * h / (Bmag + eps);
}

LocalDivBNorms compute_fv_divB_norms_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
) {
    LocalDivBNorms out{};

    const double ncell = static_cast<double>(nx * ny);

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;

    double sum_abs_norm = 0.0;
    double sum_sq_norm = 0.0;
    double max_abs_norm = 0.0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double divB =
                compute_fv_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            const double normDivB =
                compute_fv_normalized_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            const double abs_divB = std::abs(divB);
            const double abs_normDivB = std::abs(normDivB);

            sum_abs += abs_divB;
            sum_sq += divB * divB;
            max_abs = std::max(max_abs, abs_divB);

            sum_abs_norm += abs_normDivB;
            sum_sq_norm += normDivB * normDivB;
            max_abs_norm = std::max(max_abs_norm, abs_normDivB);
        }
    }

    out.L1 = sum_abs / ncell;
    out.L2 = std::sqrt(sum_sq / ncell);
    out.Linf = max_abs;

    out.L1_norm = sum_abs_norm / ncell;
    out.L2_norm = std::sqrt(sum_sq_norm / ncell);
    out.Linf_norm = max_abs_norm;

    return out;
}

std::vector<double> compute_fv_divB_field_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
) {
    std::vector<double> divB(nx * ny, 0.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            divB[idx2d(i, j, nx)] =
                compute_fv_divB_cell_2d(U, nx, ny, i, j, dx, dy);
        }
    }

    return divB;
}