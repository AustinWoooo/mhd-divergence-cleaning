#include "projection2d.hpp"
#include "glm2d_common.hpp"

#include <omp.h> 
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void subtract_mean(std::vector<double>& a) {
    if (a.empty()) return;

    double sum = 0.0;
    const int n = static_cast<int>(a.size());

    // 使用 OpenMP reduction 來平行計算總和
    #pragma omp parallel for reduction(+:sum)
    for (int k = 0; k < n; ++k) {
        sum += a[k];
    }

    const double mean = sum / static_cast<double>(n);

    // 平行扣除平均值
    #pragma omp parallel for
    for (int k = 0; k < n; ++k) {
        a[k] -= mean;
    }
}

std::vector<double> solve_periodic_poisson_sor_5pt_red_black(
    const std::vector<double>& rhs_input,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (inv_dx2 + inv_dy2);

    std::vector<double> rhs = rhs_input;
    std::vector<double> phi(nx * ny, 0.0);

    subtract_mean(rhs);

    double max_update = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < params.poisson_max_iter; ++iter) {
        double max_update_red = 0.0;
        double max_update_black = 0.0;

        // ==========================================
        // 階段一：更新所有紅格 (Red Phase) : i + j 為偶數
        // ==========================================
        #pragma omp parallel for reduction(max:max_update_red)
        for (int j = 0; j < ny; ++j) {
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            for (int i = 0; i < nx; ++i) {
                if ((i + j) % 2 != 0) continue; // 跳過黑格

                const int ip = periodic_index(i + 1, nx);
                const int im = periodic_index(i - 1, nx);
                const int id = idx2d(i, j, nx);

                const double jacobi_value =
                    (
                        (phi[idx2d(ip, j, nx)] + phi[idx2d(im, j, nx)]) * inv_dx2
                      + (phi[idx2d(i, jp, nx)] + phi[idx2d(i, jm, nx)]) * inv_dy2
                      - rhs[id]
                    ) / denom;

                const double old_value = phi[id];
                const double new_value =
                    (1.0 - params.poisson_omega) * old_value
                  + params.poisson_omega * jacobi_value;

                phi[id] = new_value;
                max_update_red = std::max(max_update_red, std::abs(new_value - old_value));
            }
        }

        // ==========================================
        // 階段二：更新所有黑格 (Black Phase) : i + j 為奇數
        // ==========================================
        #pragma omp parallel for reduction(max:max_update_black)
        for (int j = 0; j < ny; ++j) {
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            for (int i = 0; i < nx; ++i) {
                if ((i + j) % 2 == 0) continue; // 跳過紅格

                const int ip = periodic_index(i + 1, nx);
                const int im = periodic_index(i - 1, nx);
                const int id = idx2d(i, j, nx);

                const double jacobi_value =
                    (
                        (phi[idx2d(ip, j, nx)] + phi[idx2d(im, j, nx)]) * inv_dx2
                      + (phi[idx2d(i, jp, nx)] + phi[idx2d(i, jm, nx)]) * inv_dy2
                      - rhs[id]
                    ) / denom;

                const double old_value = phi[id];
                const double new_value =
                    (1.0 - params.poisson_omega) * old_value
                  + params.poisson_omega * jacobi_value;

                phi[id] = new_value;
                max_update_black = std::max(max_update_black, std::abs(new_value - old_value));
            }
        }

        max_update = std::max(max_update_red, max_update_black);

        // 定期移除常數模式
        if (iter % 50 == 0) {
            subtract_mean(phi);
        }

        if (max_update < params.poisson_tol) {
            break;
        }
    }

    subtract_mean(phi);
    return phi;
}

} // namespace

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;

    std::vector<double> rhs =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    subtract_mean(rhs);
    if (nx % 2 != 0 || ny % 2 != 0) {
    throw std::runtime_error(
        "Red-black SOR with periodic boundaries requires even nx and even ny."
    );
}
    // 呼叫更新為紅黑排序法的求解器
    const std::vector<double> phi =
        solve_periodic_poisson_sor_5pt_red_black(rhs, params);

    // 注意：移除了 Uold，使用原地更新 (In-place update) 節省記憶體與頻寬
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            const int id = idx2d(i, j, nx);

            const double dphi_dx = (phi[idx2d(ip, j, nx)] - phi[id]) / dx;
            const double dphi_dy = (phi[idx2d(i, jp, nx)] - phi[id]) / dy;

            // 各個執行緒獨立修改各自負責的網格，保證安全
            U[id][BX] -= dphi_dx;
            U[id][BY] -= dphi_dy;
            U[id][PSI] = 0.0;
        }
    }
}