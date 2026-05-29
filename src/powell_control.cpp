#include "powell_control.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "glm2d_common.hpp"
#include "powell2d.hpp"

namespace {

double kinetic_energy_density_control(const State& cell) {
    const double rho = cell[RHO];
    if (!(rho > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return 0.5 * (
        cell[MX] * cell[MX]
      + cell[MY] * cell[MY]
      + cell[MZ] * cell[MZ]
    ) / rho;
}

double magnetic_energy_density_control(const State& cell) {
    return 0.5 * (
        cell[BX] * cell[BX]
      + cell[BY] * cell[BY]
      + cell[BZ] * cell[BZ]
    );
}

struct PowellMinPhysical {
    double min_pressure = std::numeric_limits<double>::infinity();
    double min_density = std::numeric_limits<double>::infinity();
};

PowellMinPhysical compute_min_physical_control(
    const std::vector<State>& U,
    double gamma
) {
    PowellMinPhysical out;
    for (const State& cell : U) {
        const double rho = cell[RHO];
        if (std::isfinite(rho)) {
            out.min_density = std::min(out.min_density, rho);
        }
        if (std::isfinite(rho) && rho > 0.0) {
            const double ke = kinetic_energy_density_control(cell);
            const double me = magnetic_energy_density_control(cell);
            const double internal = cell[E] - ke - me;
            const double p = (gamma - 1.0) * internal;
            if (std::isfinite(p)) {
                out.min_pressure = std::min(out.min_pressure, p);
            } else {
                out.min_pressure = std::min(
                    out.min_pressure,
                    -std::numeric_limits<double>::infinity()
                );
            }
        }
    }
    return out;
}

PowellControlBadState scan_physical_state_control(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy,
    double gamma,
    const std::string& stage
) {
    PowellControlBadState out;
    out.stage = stage;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);
            const State& cell = U[id];

            const double rho = cell[RHO];
            const double ke = kinetic_energy_density_control(cell);
            const double me = magnetic_energy_density_control(cell);
            const double internal = cell[E] - ke - me;
            const double pressure = (gamma - 1.0) * internal;

            std::string reason;
            if (!std::isfinite(rho)) {
                reason = "nonfinite_state";
            } else if (rho <= 0.0) {
                reason = "non_positive_density";
            } else if (!std::isfinite(pressure)) {
                reason = "nonfinite_state";
            } else if (pressure <= 0.0) {
                reason = "non_positive_pressure";
            }

            if (!reason.empty()) {
                out.found = true;
                out.stage = stage;
                out.reason = reason;
                out.i = i;
                out.j = j;
                out.rho = rho;
                out.pressure = pressure;
                out.total_energy = cell[E];
                out.kinetic_energy = ke;
                out.magnetic_energy = me;
                out.internal_energy = internal;
                out.Bx = cell[BX];
                out.By = cell[BY];
                out.Bz = cell[BZ];
                out.psi = cell[PSI];
                out.divB = compute_fv_divB_cell_2d(U, nx, ny, i, j, dx, dy);
                return out;
            }
        }
    }

    return out;
}

State add_scaled_increment_control(
    const State& base,
    const State& dU,
    double theta
) {
    State out = base;
    for (int n = 0; n < NVAR; ++n) {
        out[n] += theta * dU[n];
    }
    return out;
}

bool pressure_limiter_accepts_state_control(
    const State& cell,
    double gamma,
    double p_floor,
    double rho_floor
) {
    const double rho = cell[RHO];
    if (!std::isfinite(rho) || rho <= rho_floor) {
        return false;
    }

    const double ke = kinetic_energy_density_control(cell);
    const double me = magnetic_energy_density_control(cell);
    const double p = (gamma - 1.0) * (cell[E] - ke - me);

    return std::isfinite(p) && p >= p_floor;
}

} // namespace

int compute_powell_subcycle_count(
    const std::vector<State>& U,
    const GLM2DParams& params,
    double dt,
    double source_cfl
) {
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, params.nx, params.ny, params.dx, params.dy);

    double max_s = 0.0;
    for (double d : divB) {
        if (std::isfinite(d)) {
            max_s = std::max(max_s, std::abs(dt * d));
        }
    }

    return std::max(
        1,
        static_cast<int>(std::ceil(max_s / source_cfl))
    );
}

void apply_powell_subcycled_2d(
    std::vector<State>& U,
    GLM2DParams params,
    double dt_mhd,
    double gamma,
    double source_cfl
) {
    (void)gamma;
    const int nsub =
        compute_powell_subcycle_count(U, params, dt_mhd, source_cfl);
    params.dt = dt_mhd / static_cast<double>(nsub);
    for (int sub = 0; sub < nsub; ++sub) {
        apply_powell_source_2d(U, params);
    }
}

PowellControlResult apply_powell_subcycled_2d(
    std::vector<State>& U,
    GLM2DParams params,
    double dt_mhd,
    double gamma,
    double source_cfl,
    double start_time
) {
    PowellControlResult result;
    const int nsub =
        compute_powell_subcycle_count(U, params, dt_mhd, source_cfl);
    result.subcycles = nsub;

    const double dt_sub = dt_mhd / static_cast<double>(nsub);
    for (int sub = 0; sub < nsub; ++sub) {
        params.dt = dt_sub;
        const std::vector<State> Ubefore = U;

        apply_powell_source_2d(U, params);

        const PowellMinPhysical mp = compute_min_physical_control(U, gamma);
        result.min_pressure_after_source = std::min(
            result.min_pressure_after_source,
            mp.min_pressure
        );
        result.min_density_after_source = std::min(
            result.min_density_after_source,
            mp.min_density
        );

        const PowellControlBadState bad =
            scan_physical_state_control(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy,
                gamma,
                "after_powell_subcycle_update"
            );

        if (bad.found) {
            if (bad.reason == "non_positive_pressure") {
                const int id = idx2d(bad.i, bad.j, params.nx);
                result.first_bad_cell.found = true;
                result.first_bad_cell.i = bad.i;
                result.first_bad_cell.j = bad.j;
                result.first_bad_cell.time =
                    start_time + dt_sub * static_cast<double>(sub + 1);
                result.first_bad_cell.dt = dt_sub;
                result.first_bad_cell.before = Ubefore[id];
                result.first_bad_cell.after = U[id];
                result.first_bad_cell.divB =
                    compute_fv_divB_cell_2d(
                        Ubefore,
                        params.nx,
                        params.ny,
                        bad.i,
                        bad.j,
                        params.dx,
                        params.dy
                    );
            }

            result.bad_state = bad;
            result.failure_time =
                start_time + dt_sub * static_cast<double>(sub + 1);
            result.failure_stage = "after_powell_subcycle_update";
            return result;
        }
    }

    return result;
}

double find_pressure_safe_theta(
    const State& base,
    const State& dU,
    double gamma,
    double p_floor,
    double rho_floor
) {
    if (!pressure_limiter_accepts_state_control(
            base,
            gamma,
            p_floor,
            rho_floor
        )) {
        return 0.0;
    }

    const State full = add_scaled_increment_control(base, dU, 1.0);
    if (pressure_limiter_accepts_state_control(
            full,
            gamma,
            p_floor,
            rho_floor
        )) {
        return 1.0;
    }

    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 64; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const State trial = add_scaled_increment_control(base, dU, mid);
        if (pressure_limiter_accepts_state_control(
                trial,
                gamma,
                p_floor,
                rho_floor
            )) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    return lo;
}

PowellControlResult apply_powell_limited_2d(
    std::vector<State>& U,
    const GLM2DParams& params,
    double dt_mhd,
    double gamma,
    double p_floor,
    double rho_floor,
    double start_time,
    bool capture_first_limited_cell
) {
    PowellControlResult result;
    result.subcycles = 1;

    const PowellControlBadState before_bad =
        scan_physical_state_control(
            U,
            params.nx,
            params.ny,
            params.dx,
            params.dy,
            gamma,
            "before_powell_limited_source"
        );

    if (before_bad.found) {
        result.bad_state = before_bad;
        result.failure_time = start_time;
        result.failure_stage = "before_powell_limited_source";
        return result;
    }

    const std::vector<State> Ubefore = U;
    const PowellMinPhysical before_min =
        compute_min_physical_control(Ubefore, gamma);
    result.min_pressure_before_source = before_min.min_pressure;
    result.min_density_before_source = before_min.min_density;

    const std::vector<double> divB =
        compute_fv_divB_field_2d(Ubefore, params.nx, params.ny, params.dx, params.dy);

    for (double d : divB) {
        if (std::isfinite(d)) {
            result.max_abs_divB = std::max(result.max_abs_divB, std::abs(d));
        }
    }

    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const int id = idx2d(i, j, params.nx);
            const State& before = Ubefore[id];
            const State dU =
                compute_powell_source_increment_cell(before, divB[id], dt_mhd);

            double theta = 1.0;
            const State full = add_scaled_increment_control(before, dU, 1.0);
            if (!pressure_limiter_accepts_state_control(
                    full,
                    gamma,
                    p_floor,
                    rho_floor
                )) {
                theta = find_pressure_safe_theta(
                    before,
                    dU,
                    gamma,
                    p_floor,
                    rho_floor
                );
                ++result.total_limited_cells;
                result.theta_min = std::min(result.theta_min, theta);
                result.theta_sum_limited += theta;
            }

            const State after = add_scaled_increment_control(before, dU, theta);
            U[id] = after;

            if (theta < 1.0 &&
                capture_first_limited_cell &&
                !result.first_limited_cell.found) {
                result.first_limited_cell.found = true;
                result.first_limited_cell.i = i;
                result.first_limited_cell.j = j;
                result.first_limited_cell.time = start_time + dt_mhd;
                result.first_limited_cell.dt = dt_mhd;
                result.first_limited_cell.before = before;
                result.first_limited_cell.after = after;
                result.first_limited_cell.full_delta = dU;
                result.first_limited_cell.divB = divB[id];
                result.first_limited_cell.theta = theta;
            }
        }
    }

    const PowellMinPhysical after_min = compute_min_physical_control(U, gamma);
    result.min_pressure_after_source = after_min.min_pressure;
    result.min_density_after_source = after_min.min_density;
    result.limiter_activations = (result.total_limited_cells > 0) ? 1 : 0;

    const PowellControlBadState bad =
        scan_physical_state_control(
            U,
            params.nx,
            params.ny,
            params.dx,
            params.dy,
            gamma,
            "after_powell_limited_update"
        );

    if (bad.found) {
        result.bad_state = bad;
        result.failure_time = start_time + dt_mhd;
        result.failure_stage = "after_powell_limited_update";
    }

    return result;
}
