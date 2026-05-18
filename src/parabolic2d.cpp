#include "parabolic2d.hpp"
#include "glm2d_common.hpp"
#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void validate_params(const GLM2DParams& params) {
    if (params.nx <= 0 || params.ny <= 0) {
        throw std::invalid_argument("Parabolic cleaning: nx and ny must be positive.");
    }
    if (!(params.dx > 0.0) || !(params.dy > 0.0)) {
        throw std::invalid_argument("Parabolic cleaning: dx and dy must be positive.");
    }
    if (!(params.dt >= 0.0)) {
        throw std::invalid_argument("Parabolic cleaning: dt must be non-negative.");
    }
    if (!(params.cp >= 0.0)) {
        throw std::invalid_argument("Parabolic cleaning: cp must be non-negative.");
    }
}

double magnetic_energy_2d(const std::vector<State>& U) {
    double eB = 0.0;

    for (const auto& s : U) {
        const double bx = s[BX];
        const double by = s[BY];
        const double bz = s[BZ];

        eB += 0.5 * (bx * bx + by * by + bz * bz);
    }

    return eB;
}

double magnetic_rms_2d(const std::vector<State>& U) {
    if (U.empty()) return 0.0;

    double sum = 0.0;

    for (const auto& s : U) {
        const double bx = s[BX];
        const double by = s[BY];
        const double bz = s[BZ];

        sum += bx * bx + by * by + bz * bz;
    }

    return std::sqrt(sum / static_cast<double>(U.size()));
}

CleaningNorms compute_norms(const std::vector<double>& a) {
    CleaningNorms n;

    if (a.empty()) return n;

    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double max_abs = 0.0;

    for (const double x : a) {
        const double ax = std::abs(x);
        sum_abs += ax;
        sum_sq  += x * x;
        max_abs = std::max(max_abs, ax);
    }

    const double invN = 1.0 / static_cast<double>(a.size());

    n.L1   = sum_abs * invN;
    n.L2   = std::sqrt(sum_sq * invN);
    n.Linf = max_abs;

    return n;
}

double normalized_divB_L2(
    const std::vector<State>& U,
    const CleaningNorms& divB_norms,
    const GLM2DParams& params
) {
    const double b_rms = magnetic_rms_2d(U);
    const double h_min = std::min(params.dx, params.dy);

    // Characteristic scale: |B| / length.
    const double scale = b_rms / h_min;

    if (scale == 0.0) return divB_norms.L2;

    return divB_norms.L2 / scale;
}

double explicit_diffusion_dt_limit(
    const GLM2DParams& params,
    const ParabolicCleaningOptions& options
) {
    const double cp2 = params.cp * params.cp;

    if (cp2 == 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double inv_dx2 = 1.0 / (params.dx * params.dx);
    const double inv_dy2 = 1.0 / (params.dy * params.dy);

    // 2D explicit diffusion condition:
    // cp^2 dt (1/dx^2 + 1/dy^2) <= 1/2.
    const double dt_limit = 0.5 / (cp2 * (inv_dx2 + inv_dy2));

    return options.cfl_diffusion * dt_limit;
}

void apply_one_parabolic_substep(
    std::vector<State>& U,
    const GLM2DParams& params,
    const double dt_sub,
    const ParabolicEnergyPolicy energy_policy,
    double& max_delta_B
) {
    const int nx = params.nx;
    const int ny = params.ny;
    const double dx = params.dx;
    const double dy = params.dy;
    const double cp2 = params.cp * params.cp;

    // Important:
    // This must be discretely consistent with the forward gradient below.
    // Ideally:
    // divB(i,j) = [Bx(i,j)-Bx(i-1,j)]/dx + [By(i,j)-By(i,j-1)]/dy.
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    #pragma omp parallel for collapse(2) reduction(max:max_delta_B) schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            const int jp = periodic_index(j + 1, ny);

            const int id = idx2d(i, j, nx);

            const double grad_divB_x =
                (divB[idx2d(ip, j, nx)] - divB[id]) / dx;

            const double grad_divB_y =
                (divB[idx2d(i, jp, nx)] - divB[id]) / dy;

            const double bx_old = U[id][BX];
            const double by_old = U[id][BY];

            const double bx_new =
                bx_old + dt_sub * cp2 * grad_divB_x;

            const double by_new =
                by_old + dt_sub * cp2 * grad_divB_y;

            U[id][BX] = bx_new;
            U[id][BY] = by_new;

            if (energy_policy == ParabolicEnergyPolicy::KeepInternalEnergy) {
                const double dE_mag =
                    0.5 * (bx_new * bx_new + by_new * by_new
                         - bx_old * bx_old - by_old * by_old);

                U[id][E] += dE_mag;
            }

            const double dBx = bx_new - bx_old;
            const double dBy = by_new - by_old;
            const double dB  = std::sqrt(dBx * dBx + dBy * dBy);

            max_delta_B = std::max(max_delta_B, dB);
        }
    }
}

} // namespace

ParabolicCleaningDiagnostics apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params,
    const ParabolicCleaningOptions& options
) {
    validate_params(params);

    if (static_cast<int>(U.size()) != params.nx * params.ny) {
        throw std::invalid_argument(
            "Parabolic cleaning: U.size() != nx * ny."
        );
    }

    ParabolicCleaningDiagnostics diag;

    if (options.collect_diagnostics) {
        const std::vector<double> divB_before =
            compute_fv_divB_field_2d(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy
            );

        diag.before = compute_norms(divB_before);
        diag.normalized_L2_before =
            normalized_divB_L2(U, diag.before, params);

        diag.magnetic_energy_before = magnetic_energy_2d(U);
    }

    const double dt_limit =
    explicit_diffusion_dt_limit(params, options);

    diag.dt_stability_limit = dt_limit;
    if (params.dt > dt_limit && !options.subcycle) {
        if (options.abort_if_unstable) {
            throw std::runtime_error(
                "Parabolic cleaning: dt violates explicit diffusion stability limit."
            );
        }
    }

    int n_substeps = 1;

    if (options.subcycle && params.dt > dt_limit) {
        n_substeps = static_cast<int>(std::ceil(params.dt / dt_limit));
        n_substeps = std::max(n_substeps, 1);
    }

    const double dt_sub =
        (n_substeps > 0) ? params.dt / static_cast<double>(n_substeps) : 0.0;

    diag.n_substeps = n_substeps;
    diag.dt_substep = dt_sub;

    double max_delta_B = 0.0;

    for (int step = 0; step < n_substeps; ++step) {
        apply_one_parabolic_substep(
            U,
            params,
            dt_sub,
            options.energy_policy,
            max_delta_B
        );
    }

    diag.max_delta_B = max_delta_B;

    if (options.collect_diagnostics) {
        const std::vector<double> divB_after =
            compute_fv_divB_field_2d(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy
            );

        diag.after = compute_norms(divB_after);
        diag.normalized_L2_after =
            normalized_divB_L2(U, diag.after, params);

        diag.magnetic_energy_after = magnetic_energy_2d(U);
    }

    return diag;
}

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    ParabolicCleaningOptions options;
    (void)apply_parabolic_cleaning_2d(U, params, options);
}