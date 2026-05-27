// =============================================================================
//  src/mhd_runner.cpp
//
//  Integrated HLLD + GLM-cleaning runner.
//
//  Reused pieces (no re-implementation):
//    MHD::compute_flux        — HLLD Riemann solver (src/hlld_solver.cpp)
//    MHD::PrimState           — primitive/conserved conversions
//    advance_glm_2d_one_step  — full GLM cleaning dispatch (src/glm2d.cpp)
//    compute_fv_divB_norms_2d — divB diagnostic (src/glm2d_common.cpp)
//    cleaning_name            — type-to-string (src/glm.cpp)
// =============================================================================

#include "mhd_runner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "HLLD_mhd_solver.hpp"
#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_common.hpp"
#include "projection2d.hpp"

namespace fs = std::filesystem;
using namespace MHD;

namespace {

// -----------------------------------------------------------------------------
//  State / primitive adapters.
// -----------------------------------------------------------------------------
//  Use raw primitive recovery here so diagnostics and failure detection see the
//  physical state produced by the conservative update, not a repaired one.

inline PrimState state_to_prim(const State& s, double gamma) {
    return PrimState::from_conserved_raw(s, gamma);
}

inline State prim_to_state(const PrimState& W, double gamma) {
    return W.to_conserved(gamma);
}

// Minimum theta for the relaxed projection limiter (Task E).
// Below this, the full projection cannot be made physical and we record the
// failure.  Do NOT claim exact projection when theta < 1.
constexpr double MIN_PROJECTION_THETA = 1.0 / 64.0;

// -----------------------------------------------------------------------------
//  Stage-level positivity diagnostics (Task A)
// -----------------------------------------------------------------------------

struct MinPhysical {
    double min_pressure = std::numeric_limits<double>::infinity();
    double min_density  = std::numeric_limits<double>::infinity();
};

// Task A: min pressure and density at each named stage.
struct StagePressureMins {
    double min_pressure_before_hydro       = std::numeric_limits<double>::infinity();
    double min_density_before_hydro        = std::numeric_limits<double>::infinity();
    double min_pressure_after_hydro        = std::numeric_limits<double>::infinity();
    double min_density_after_hydro         = std::numeric_limits<double>::infinity();
    double min_pressure_after_cleaning_B   = std::numeric_limits<double>::infinity();
    double min_density_after_cleaning_B    = std::numeric_limits<double>::infinity();
    double min_pressure_after_energy_repair= std::numeric_limits<double>::infinity();
    double min_density_after_energy_repair = std::numeric_limits<double>::infinity();
    double min_pressure_after_full_step    = std::numeric_limits<double>::infinity();
    double min_density_after_full_step     = std::numeric_limits<double>::infinity();
    // The earliest stage at which min_pressure first went non-positive.
    std::string failure_stage;
};

struct MHDRunDiagnostics {
    double total_mass = 0.0;
    double total_momentum_x = 0.0;
    double total_momentum_y = 0.0;
    double total_momentum_z = 0.0;
    double total_energy = 0.0;
    double min_density = std::numeric_limits<double>::infinity();
    double min_pressure = std::numeric_limits<double>::infinity();
    bool has_nonfinite = false;
    bool has_negative_density = false;
    bool has_negative_pressure = false;
};

struct BadStateRecord {
    bool found = false;
    std::string stage;
    std::string reason;
    int i = -1;
    int j = -1;
    double rho = 0.0;
    double pressure = 0.0;
    double total_energy = 0.0;
    double kinetic_energy = 0.0;
    double magnetic_energy = 0.0;
    double internal_energy = 0.0;
    double Bx = 0.0;
    double By = 0.0;
    double Bz = 0.0;
    double psi = 0.0;
    double divB = 0.0;
};

struct CleaningAdvanceStats {
    int subcycles = 0;
    int projection_iterations = 0;
    double projection_solver_update_residual =
        std::numeric_limits<double>::quiet_NaN();
    double projection_final_residual =
        std::numeric_limits<double>::quiet_NaN();
    double projection_true_residual =
        std::numeric_limits<double>::quiet_NaN();
    bool projection_converged = true;
    bool projection_used = false;
    BadStateRecord bad_state;
    double failure_time = std::numeric_limits<double>::quiet_NaN();
    // Task E: theta used for relaxed projection (1.0 = full, < 1 = partial).
    double projection_theta = 1.0;
    // Task A: per-stage minimums recorded during the cleaning substep.
    StagePressureMins stage_mins;
};

std::string cleaning_energy_policy_name(CleaningEnergyPolicy policy) {
    switch (policy) {
        case CleaningEnergyPolicy::ConserveTotalEnergy:
            return "conserve_total_energy";
        case CleaningEnergyPolicy::PreserveThermalPressure:
            return "preserve_thermal_pressure";
        default:
            return "unknown";
    }
}

double kinetic_energy_density(const State& cell) {
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

double magnetic_energy_density(const State& cell) {
    return 0.5 * (
        cell[BX] * cell[BX]
      + cell[BY] * cell[BY]
      + cell[BZ] * cell[BZ]
    );
}

// Task A helper: compute min raw pressure and min density from a state vector.
MinPhysical compute_min_physical(
    const std::vector<State>& U,
    double gamma
) {
    MinPhysical out;
    for (const State& cell : U) {
        const double rho = cell[RHO];
        if (std::isfinite(rho)) {
            out.min_density = std::min(out.min_density, rho);
        }
        if (std::isfinite(rho) && rho > 0.0) {
            const double ke = kinetic_energy_density(cell);
            const double me = magnetic_energy_density(cell);
            const double internal = cell[E] - ke - me;
            const double p = (gamma - 1.0) * internal;
            if (std::isfinite(p)) {
                out.min_pressure = std::min(out.min_pressure, p);
            } else {
                // non-finite pressure counts as a negative flag
                out.min_pressure = std::min(
                    out.min_pressure,
                    -std::numeric_limits<double>::infinity()
                );
            }
        }
    }
    return out;
}

BadStateRecord scan_physical_state(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy,
    double gamma,
    const std::string& stage
) {
    BadStateRecord out;
    out.stage = stage;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);
            const State& cell = U[id];

            const double rho = cell[RHO];
            const double ke = kinetic_energy_density(cell);
            const double me = magnetic_energy_density(cell);
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

// Determine the failure_stage string from the StagePressureMins record.
// Returns "none" if all minimums are positive (no failure).
std::string determine_failure_stage(const StagePressureMins& sm) {
    if (sm.min_pressure_before_hydro <= 0.0) return "before_hydro_step";
    if (sm.min_pressure_after_hydro <= 0.0)  return "after_hydro_step";
    if (sm.min_pressure_after_cleaning_B <= 0.0)    return "after_cleaning_B_update";
    if (sm.min_pressure_after_energy_repair <= 0.0) return "after_cleaning_energy_repair";
    if (sm.min_pressure_after_full_step <= 0.0)     return "after_full_step";
    return "none";
}

void write_cleaning_failure_csv(
    const MHDRunParams& params,
    const std::string& method,
    int step,
    double time,
    const BadStateRecord& bad,
    const StagePressureMins& stage_mins,
    double projection_theta,
    int retry_count,
    double min_dt_used
) {
    const std::string filename =
        "results/mhd_runner/failures/"
      + params.glm.out_prefix
      + "_"
      + method
      + "_"
      + cleaning_energy_policy_name(params.glm.energy_policy)
      + "_failure.csv";
    ensure_parent_directory(filename);

    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Failed to open cleaning failure file: " + filename);
    }

    // Task A: include stage columns in the failure CSV.
    fout << "problem,method,stage,step,time,reason,i,j,"
         << "rho,pressure,total_energy,kinetic_energy,magnetic_energy,"
         << "internal_energy,Bx,By,Bz,psi,divB,energy_policy,"
         << "failure_stage,"
         << "min_pressure_before_hydro,min_pressure_after_hydro,"
         << "min_pressure_after_cleaning_B,min_pressure_after_energy_repair,"
         << "min_pressure_after_full_step,"
         << "projection_theta,retry_count,min_dt_used\n";

    const std::string fs_name =
        stage_mins.failure_stage.empty()
        ? determine_failure_stage(stage_mins)
        : stage_mins.failure_stage;

    auto write_double = [&](double v) {
        if (std::isfinite(v)) fout << v; else fout << "";
    };

    fout << params.problem << ","
         << method << ","
         << bad.stage << ","
         << step << ","
         << time << ","
         << bad.reason << ","
         << bad.i << ","
         << bad.j << ","
         << bad.rho << ","
         << bad.pressure << ","
         << bad.total_energy << ","
         << bad.kinetic_energy << ","
         << bad.magnetic_energy << ","
         << bad.internal_energy << ","
         << bad.Bx << ","
         << bad.By << ","
         << bad.Bz << ","
         << bad.psi << ","
         << bad.divB << ","
         << cleaning_energy_policy_name(params.glm.energy_policy) << ","
         << fs_name << ",";
    write_double(stage_mins.min_pressure_before_hydro);
    fout << ",";
    write_double(stage_mins.min_pressure_after_hydro);
    fout << ",";
    write_double(stage_mins.min_pressure_after_cleaning_B);
    fout << ",";
    write_double(stage_mins.min_pressure_after_energy_repair);
    fout << ",";
    write_double(stage_mins.min_pressure_after_full_step);
    fout << "," << projection_theta
         << "," << retry_count
         << ",";
    write_double(min_dt_used);
    fout << "\n";
}

MHDRunDiagnostics compute_mhd_run_diagnostics(
    const std::vector<State>& U,
    double gamma,
    double cell_area
) {
    MHDRunDiagnostics out;

    for (const State& cell : U) {
        for (double value : cell) {
            if (!std::isfinite(value)) {
                out.has_nonfinite = true;
            }
        }

        const double rho = cell[RHO];
        const double energy = cell[E];

        out.total_mass += rho * cell_area;
        out.total_momentum_x += cell[MX] * cell_area;
        out.total_momentum_y += cell[MY] * cell_area;
        out.total_momentum_z += cell[MZ] * cell_area;
        out.total_energy += energy * cell_area;

        if (std::isfinite(rho)) {
            out.min_density = std::min(out.min_density, rho);
            if (rho < 0.0) {
                out.has_negative_density = true;
            }
        }

        const PrimState W = state_to_prim(cell, gamma);
        if (!std::isfinite(W.p)) {
            out.has_nonfinite = true;
        } else {
            out.min_pressure = std::min(out.min_pressure, W.p);
            if (W.p < 0.0) {
                out.has_negative_pressure = true;
            }
        }
    }

    return out;
}

bool has_mhd_run_failure(const MHDRunDiagnostics& diag) {
    return diag.has_nonfinite
        || diag.has_negative_density
        || diag.has_negative_pressure;
}

void write_optional_double(std::ostream& out, double value) {
    if (std::isfinite(value)) {
        out << value;
    }
}

bool benefits_from_hydro_retry(CleaningType type) {
    return type == CleaningType::PARABOLIC
        || type == CleaningType::ELLIPTIC_PROJECTION;
}

void print_mhd_run_failure_warning(
    const MHDRunParams& params,
    const std::string& cleaning,
    int step,
    double time,
    double dt,
    const MHDRunDiagnostics& diag
) {
    std::cerr << "WARNING: stopping MHD run after invalid state detected"
              << "  problem=" << params.problem
              << "  cleaning=" << cleaning
              << "  step=" << step
              << "  time=" << time
              << "  dt=" << dt
              << "  has_nonfinite=" << (diag.has_nonfinite ? 1 : 0)
              << "  has_negative_density="
              << (diag.has_negative_density ? 1 : 0)
              << "  has_negative_pressure="
              << (diag.has_negative_pressure ? 1 : 0)
              << "\n";
}

bool pressure_preserving_policy_applies(CleaningType type) {
    return type == CleaningType::PARABOLIC
        || type == CleaningType::ELLIPTIC_PROJECTION;
}

void preserve_thermal_pressure_after_cleaning(
    std::vector<State>& U,
    const std::vector<State>& Uold
) {
    const int ncell = static_cast<int>(U.size());
    for (int id = 0; id < ncell; ++id) {
        const double old_me = magnetic_energy_density(Uold[id]);
        const double new_me = magnetic_energy_density(U[id]);
        U[id][E] = Uold[id][E] + (new_me - old_me);
    }
}

// Task E: Elliptic projection with a relaxed limiter for conserve_total_energy.
//
// Solves Poisson once, then tries B_new = B_old - theta * grad(phi) with
// theta halved until the state is physical or theta < MIN_PROJECTION_THETA.
// Records the theta used in the returned ProjectionResult.
//
// Only call this for ELLIPTIC_PROJECTION + ConserveTotalEnergy.
// Do NOT use the returned result to claim exact div-cleaning if theta < 1.
ProjectionResult apply_elliptic_projection_theta_limited(
    std::vector<State>& U,
    const GLM2DParams& params,
    double gamma
) {
    std::vector<double> phi;
    const ProjectionResult info = solve_projection_phi_2d(U, params, phi);

    const std::vector<State> Upre = U;

    double theta = 1.0;
    bool found_physical = false;

    while (theta >= MIN_PROJECTION_THETA) {
        U = Upre;
        apply_projection_B_correction_2d(U, phi, params, theta);

        // Check that ALL cells have positive raw pressure.
        const MinPhysical mp = compute_min_physical(U, gamma);
        if (mp.min_pressure > 0.0) {
            found_physical = true;
            break;
        }

        theta *= 0.5;
    }

    if (!found_physical) {
        // Apply minimum available theta even though state is still non-physical.
        // The caller will detect this via the returned theta and bad-state scan.
        U = Upre;
        apply_projection_B_correction_2d(U, phi, params, theta);
    }

    ProjectionResult result = info;
    result.projection_theta = theta;
    return result;
}

ProjectionResult apply_cleaning_update_for_runner(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params,
    double gamma  // needed for theta-limiter (Task E)
) {
    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        if (params.energy_policy == CleaningEnergyPolicy::ConserveTotalEnergy) {
            // Task E: use relaxed projection limiter.
            return apply_elliptic_projection_theta_limited(U, params, gamma);
        }
        return apply_elliptic_projection_2d(U, params);
    }

    advance_glm_2d_one_step(U, type, params);
    return {};
}

CleaningAdvanceStats apply_cleaning_with_subcycles(
    std::vector<State>& U,
    CleaningType type,
    GLM2DParams params,
    const MHDRunParams& run_params,
    const std::string& method_name,
    int step,
    double time,
    double dt_mhd,
    double gamma,
    bool print_parabolic_subcycling
) {
    CleaningAdvanceStats stats;

    if (type == CleaningType::NONE || dt_mhd <= 0.0) {
        return stats;
    }

    const double dt_clean =
        max_cleaning_dt(type, params.dx, params.dy, params);

    int nsub = 1;
    if (std::isfinite(dt_clean) && dt_clean > 0.0 && dt_mhd > dt_clean) {
        nsub = static_cast<int>(std::ceil(dt_mhd / dt_clean));
    }

    stats.subcycles = nsub;

    if (type == CleaningType::PARABOLIC &&
        nsub > 1 &&
        print_parabolic_subcycling) {
        std::cout << "  [" << method_name << "] cleaning subcycles="
                  << nsub << " at step=" << step << "\n";
    }

    const double dt_sub = dt_mhd / static_cast<double>(nsub);

    for (int sub = 0; sub < nsub; ++sub) {
        params.dt = dt_sub;

        std::vector<State> Uold;
        const bool repair_energy =
            params.energy_policy == CleaningEnergyPolicy::PreserveThermalPressure
         && pressure_preserving_policy_applies(type);

        if (repair_energy) {
            Uold = U;
        }

        const ProjectionResult projection =
            apply_cleaning_update_for_runner(U, type, params, gamma);

        if (projection.iterations > 0 || type == CleaningType::ELLIPTIC_PROJECTION) {
            stats.projection_used = true;
            stats.projection_iterations += projection.iterations;
            stats.projection_solver_update_residual =
                projection.solver_update_residual;
            stats.projection_final_residual = projection.final_residual;
            stats.projection_true_residual = projection.true_residual_Linf;
            stats.projection_converged =
                stats.projection_converged && projection.converged;
            // Task E: record the theta used (may be < 1 for ConserveTotalEnergy).
            stats.projection_theta = projection.projection_theta;
            if (projection.projection_theta < 1.0) {
                std::cout << "  [" << method_name << "] relaxed projection"
                          << "  theta=" << projection.projection_theta
                          << "  step=" << step << "\n";
            }
        }

        // Task A: record min physical after B correction (before energy repair).
        {
            const MinPhysical mp = compute_min_physical(U, gamma);
            stats.stage_mins.min_pressure_after_cleaning_B =
                std::min(stats.stage_mins.min_pressure_after_cleaning_B,
                         mp.min_pressure);
            stats.stage_mins.min_density_after_cleaning_B =
                std::min(stats.stage_mins.min_density_after_cleaning_B,
                         mp.min_density);
        }

        if (repair_energy) {
            preserve_thermal_pressure_after_cleaning(U, Uold);
        }

        // Task A: record min physical after energy repair (or no-op if no repair).
        {
            const MinPhysical mp = compute_min_physical(U, gamma);
            stats.stage_mins.min_pressure_after_energy_repair =
                std::min(stats.stage_mins.min_pressure_after_energy_repair,
                         mp.min_pressure);
            stats.stage_mins.min_density_after_energy_repair =
                std::min(stats.stage_mins.min_density_after_energy_repair,
                         mp.min_density);
        }

        const BadStateRecord bad =
            scan_physical_state(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy,
                gamma,
                "after_cleaning"
            );

        if (bad.found) {
            stats.bad_state = bad;
            stats.failure_time =
                time + dt_sub * static_cast<double>(sub + 1);
            return stats;
        }
    }

    return stats;
}

// -----------------------------------------------------------------------------
//  Max signal speed for the CFL condition.
// -----------------------------------------------------------------------------

double max_signal_speed_2d(
    const std::vector<State>& U,
    double gamma,
    double ch,
    bool include_ch
) {
    double smax = 0.0;
    for (const State& s : U) {
        const PrimState W = state_to_prim(s, gamma);
        const double rho = std::max(W.rho, TINY_NUMBER);
        const double a2  = gamma * std::max(W.p, 0.0) / rho;
        const double b2  = (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz) / rho;
        const double bn2_x = W.Bx * W.Bx / rho;
        const double bn2_y = W.By * W.By / rho;
        const double term  = a2 + b2;
        const double disc_x = std::max(term * term - 4.0 * a2 * bn2_x, 0.0);
        const double disc_y = std::max(term * term - 4.0 * a2 * bn2_y, 0.0);
        const double cf_x = std::sqrt(std::max(0.5 * (term + std::sqrt(disc_x)), 0.0));
        const double cf_y = std::sqrt(std::max(0.5 * (term + std::sqrt(disc_y)), 0.0));
        double s_local = std::max(std::abs(W.u) + cf_x, std::abs(W.v) + cf_y);
        if (include_ch) s_local = std::max(s_local, ch);
        smax = std::max(smax, s_local);
    }
    return smax;
}

// -----------------------------------------------------------------------------
//  Finite-volume RHS = -div(F) on a periodic grid.
//  Calls MHD::compute_flux at every interface; does not re-implement HLLD.
// -----------------------------------------------------------------------------
//  Face-flux storage convention:
//    flux_x[idx2d(i, j, nx)] = numerical flux between cells (i-1, j) and (i, j)
//    flux_y[idx2d(i, j, nx)] = numerical flux between cells (i, j-1) and (i, j)

std::vector<State> compute_rhs_hlld_2d(
    const std::vector<State>& U,
    int nx, int ny,
    double dx, double dy,
    double gamma
) {
    const int ncell = nx * ny;

    std::vector<State> flux_x(ncell);
    std::vector<State> flux_y(ncell);

    // X-direction Riemann problems.
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int iL = periodic_index(i - 1, nx);
            const PrimState WL = state_to_prim(U[idx2d(iL, j, nx)], gamma);
            const PrimState WR = state_to_prim(U[idx2d(i,  j, nx)], gamma);
            flux_x[idx2d(i, j, nx)] = compute_flux(WL, WR, /*direction=*/0, gamma);
        }
    }

    // Y-direction Riemann problems.
    for (int j = 0; j < ny; ++j) {
        const int jL = periodic_index(j - 1, ny);
        for (int i = 0; i < nx; ++i) {
            const PrimState WL = state_to_prim(U[idx2d(i, jL, nx)], gamma);
            const PrimState WR = state_to_prim(U[idx2d(i, j,  nx)], gamma);
            flux_y[idx2d(i, j, nx)] = compute_flux(WL, WR, /*direction=*/1, gamma);
        }
    }

    // Assemble RHS:
    //   RHS[i,j] = -(F_x[i+1,j] - F_x[i,j]) / dx
    //              -(F_y[i,j+1] - F_y[i,j]) / dy
    std::vector<State> RHS(ncell);
    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);
        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            State r;
            r.fill(0.0);
            const State& fxL = flux_x[idx2d(i,  j,  nx)];
            const State& fxR = flux_x[idx2d(ip, j,  nx)];
            const State& fyL = flux_y[idx2d(i,  j,  nx)];
            const State& fyR = flux_y[idx2d(i,  jp, nx)];
            for (int k = 0; k < NVAR; ++k) {
                r[k] = -(fxR[k] - fxL[k]) / dx
                       -(fyR[k] - fyL[k]) / dy;
            }
            RHS[idx2d(i, j, nx)] = r;
        }
    }
    return RHS;
}

// -----------------------------------------------------------------------------
//  Heun (RK2) update with HLLD fluxes only. Cleaning is applied separately.
//
//  Task C: when stage_glm_params != nullptr (project_each_stage), apply
//  elliptic projection (and optional pressure-preserving energy repair) to the
//  predictor state Us before computing the second-stage RHS.
// -----------------------------------------------------------------------------

void rk2_step_hlld_2d(
    std::vector<State>& U,
    int nx, int ny,
    double dx, double dy,
    double gamma, double dt,
    const GLM2DParams* stage_glm_params   // non-null enables stage projection
) {
    const int ncell = nx * ny;

    const std::vector<State> R1 = compute_rhs_hlld_2d(U, nx, ny, dx, dy, gamma);

    std::vector<State> Us(ncell);
    for (int id = 0; id < ncell; ++id) {
        for (int k = 0; k < NVAR; ++k) {
            Us[id][k] = U[id][k] + dt * R1[id][k];
        }
    }

    // Task C: optional projection on the intermediate predictor state.
    if (stage_glm_params != nullptr) {
        GLM2DParams proj_params = *stage_glm_params;
        proj_params.dt = dt;

        const bool repair =
            (proj_params.energy_policy ==
             CleaningEnergyPolicy::PreserveThermalPressure);

        std::vector<State> Us_pre;
        if (repair) {
            Us_pre = Us;
        }

        // For the stage projection, always apply the full theta=1 projection.
        // (The theta limiter is applied at the end-of-step projection only.)
        apply_elliptic_projection_2d(Us, proj_params);

        if (repair) {
            for (int id = 0; id < ncell; ++id) {
                const double old_me =
                    0.5 * (Us_pre[id][BX]*Us_pre[id][BX]
                         + Us_pre[id][BY]*Us_pre[id][BY]
                         + Us_pre[id][BZ]*Us_pre[id][BZ]);
                const double new_me =
                    0.5 * (Us[id][BX]*Us[id][BX]
                         + Us[id][BY]*Us[id][BY]
                         + Us[id][BZ]*Us[id][BZ]);
                Us[id][E] = Us_pre[id][E] + (new_me - old_me);
            }
        }
    }

    const std::vector<State> R2 = compute_rhs_hlld_2d(Us, nx, ny, dx, dy, gamma);

    for (int id = 0; id < ncell; ++id) {
        for (int k = 0; k < NVAR; ++k) {
            U[id][k] = 0.5 * (U[id][k] + Us[id][k] + dt * R2[id][k]);
        }
    }
}

// Overload without stage projection (backward-compatible default).
void rk2_step_hlld_2d(
    std::vector<State>& U,
    int nx, int ny,
    double dx, double dy,
    double gamma, double dt
) {
    rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt, nullptr);
}

// -----------------------------------------------------------------------------
//  Snapshot writer: dumps primitive variables plus divB diagnostics.
// -----------------------------------------------------------------------------

void write_mhd_2d_snapshot(
    const std::vector<State>& U,
    const MHDRunParams& params,
    const std::string& filename
) {
    ensure_parent_directory(filename);

    std::ofstream fout(filename);
    if (!fout) {
        throw std::runtime_error("Failed to open snapshot file: " + filename);
    }

    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;

    fout << "i,j,x,y,rho,u,v,w,p,Bx,By,Bz,psi,divB_fv,Bmag\n";

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double y = (j + 0.5) * dy;

            const PrimState W = state_to_prim(U[idx2d(i, j, nx)], gamma);

            const double divB =
                compute_fv_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            const double Bmag = std::sqrt(
                W.Bx * W.Bx + W.By * W.By + W.Bz * W.Bz
            );

            fout << i << "," << j << ","
                 << x << "," << y << ","
                 << W.rho << "," << W.u << "," << W.v << "," << W.w << ","
                 << W.p << ","
                 << W.Bx << "," << W.By << "," << W.Bz << ","
                 << W.psi << ","
                 << divB << "," << Bmag << "\n";
        }
    }
}

void write_mhd_diagnostic_row(
    std::ofstream& diag,
    int step,
    double time,
    double dt,
    const LocalDivBNorms& norms,
    const MHDRunDiagnostics& run_diag,
    int cleaning_subcycles_step,
    int projection_iterations_step,
    double projection_solver_update_residual,
    double projection_final_residual,
    double projection_true_residual,
    bool projection_converged
) {
    diag << step << "," << time << "," << dt << ","
         << norms.L1 << "," << norms.L2 << "," << norms.Linf << ","
         << norms.L1_norm << "," << norms.L2_norm << "," << norms.Linf_norm
         << ","
         << run_diag.total_mass << ","
         << run_diag.total_momentum_x << ","
         << run_diag.total_momentum_y << ","
         << run_diag.total_momentum_z << ","
         << run_diag.total_energy << ","
         << run_diag.min_density << ","
         << run_diag.min_pressure << ","
         << (run_diag.has_nonfinite ? 1 : 0) << ","
         << (run_diag.has_negative_density ? 1 : 0) << ","
         << (run_diag.has_negative_pressure ? 1 : 0) << ","
         << cleaning_subcycles_step << ","
         << projection_iterations_step << ",";
    write_optional_double(diag, projection_solver_update_residual);
    diag << ",";
    write_optional_double(diag, projection_final_residual);
    diag << ",";
    write_optional_double(diag, projection_true_residual);
    diag << ","
         << (projection_converged ? 1 : 0) << "\n";
}

void write_mhd_run_summary(
    const MHDRunParams& params,
    const std::string& method,
    bool final_time_reached,
    double failure_time,
    const std::string& failure_reason,
    const LocalDivBNorms& final_norms,
    double energy_initial,
    const MHDRunDiagnostics& final_diag,
    long long cleaning_subcycles_total,
    long long projection_iterations_total,
    CleaningEnergyPolicy energy_policy,
    double projection_true_residual,
    const StagePressureMins& stage_mins,
    double projection_theta,
    int retry_count,
    double min_dt_used
) {
    const std::string filename =
        "results/mhd_runner/summaries/"
      + params.glm.out_prefix
      + "_"
      + method
      + "_summary.csv";

    ensure_parent_directory(filename);

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open summary file: " + filename);
    }

    const double energy_final = final_diag.total_energy;
    const double energy_drift =
        (energy_final - energy_initial)
      / std::max(std::abs(energy_initial), 1.0e-30);

    const std::string status =
        final_time_reached ? "finished" : "failed";

    // Task A: add stage pressure columns to summary.
    out << "problem,method,status,final_time_reached,"
        << "failure_time,failure_reason,"
        << "final_L1_fv,final_L2_fv,final_Linf_fv,"
        << "energy_initial,energy_final,energy_drift,"
        << "min_raw_pressure,min_pressure,min_density,"
        << "energy_policy,projection_true_residual,"
        << "cleaning_subcycles_total,projection_iterations_total,"
        << "failure_stage,"
        << "min_pressure_before_hydro,min_pressure_after_hydro,"
        << "min_pressure_after_cleaning_B,min_pressure_after_energy_repair,"
        << "min_pressure_after_full_step,"
        << "projection_theta,retry_count,min_dt_used\n";

    const std::string fs_name =
        stage_mins.failure_stage.empty()
        ? determine_failure_stage(stage_mins)
        : stage_mins.failure_stage;

    out << params.problem << ","
         << method << ","
         << status << ","
         << (final_time_reached ? 1 : 0) << ",";
    write_optional_double(out, failure_time);
    out << ","
         << failure_reason << ","
         << final_norms.L1 << ","
         << final_norms.L2 << ","
         << final_norms.Linf << ","
         << energy_initial << ","
         << energy_final << ","
         << energy_drift << ","
         << final_diag.min_pressure << ","
         << final_diag.min_pressure << ","
         << final_diag.min_density << ","
         << cleaning_energy_policy_name(energy_policy) << ",";
    write_optional_double(out, projection_true_residual);
    out << ","
         << cleaning_subcycles_total << ","
         << projection_iterations_total << ","
         << fs_name << ",";
    write_optional_double(out, stage_mins.min_pressure_before_hydro);
    out << ",";
    write_optional_double(out, stage_mins.min_pressure_after_hydro);
    out << ",";
    write_optional_double(out, stage_mins.min_pressure_after_cleaning_B);
    out << ",";
    write_optional_double(out, stage_mins.min_pressure_after_energy_repair);
    out << ",";
    write_optional_double(out, stage_mins.min_pressure_after_full_step);
    out << "," << projection_theta
        << "," << retry_count
        << ",";
    write_optional_double(out, min_dt_used);
    out << "\n";
}

} // namespace

// =============================================================================
//  Initial conditions
// =============================================================================

void initialize_orszag_tang_2d(
    std::vector<State>& U,
    const MHDRunParams& params
) {
    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error("initialize_orszag_tang_2d: U size mismatch.");
    }

    const double pi = M_PI;
    const double B0   = 1.0 / std::sqrt(4.0 * pi);
    const double rho0 = 25.0 / (36.0 * pi);
    const double p0   = 5.0  / (12.0 * pi);

    for (int j = 0; j < ny; ++j) {
        const double y = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;

            const PrimState W(
                rho0,
                -std::sin(2.0 * pi * y),
                 std::sin(2.0 * pi * x),
                 0.0,
                 p0,
                -B0 * std::sin(2.0 * pi * y),
                 B0 * std::sin(4.0 * pi * x),
                 0.0,
                 0.0
            );

            U[idx2d(i, j, nx)] = prim_to_state(W, gamma);
        }
    }
}

void initialize_brio_wu_strip_2d(
    std::vector<State>& U,
    const MHDRunParams& params
) {
    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double gamma = params.gamma;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error("initialize_brio_wu_strip_2d: U size mismatch.");
    }

    // Brio & Wu (1988) 1D shock tube, replicated along y to make a 2D strip.
    // Discontinuity at x = xlen/2. Standard convention: gamma = 2.
    const double x_mid = 0.5 * params.glm.xlen;

    const PrimState WL(1.0,   0.0, 0.0, 0.0,  1.0,  0.75,  1.0, 0.0, 0.0);
    const PrimState WR(0.125, 0.0, 0.0, 0.0,  0.1,  0.75, -1.0, 0.0, 0.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const PrimState& W = (x < x_mid) ? WL : WR;
            U[idx2d(i, j, nx)] = prim_to_state(W, gamma);
        }
    }
}

void initialize_field_loop_2d(
    std::vector<State>& U,
    const MHDRunParams& params
) {
    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error("initialize_field_loop_2d: U size mismatch.");
    }

    const double xc = 0.5;
    const double yc = 0.5;
    const double radius = 0.15;
    const double A0 = 1.0e-3;

    for (int j = 0; j < ny; ++j) {
        const double y = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double rx = x - xc;
            const double ry = y - yc;
            const double r = std::sqrt(rx * rx + ry * ry);

            double Bx = 0.0;
            double By = 0.0;
            if (r > 0.0 && r < radius) {
                Bx = -A0 * ry / r;
                By =  A0 * rx / r;
            }

            const PrimState W(
                1.0,
                1.0,
                1.0,
                0.0,
                1.0,
                Bx,
                By,
                0.0,
                0.0
            );

            U[idx2d(i, j, nx)] = prim_to_state(W, gamma);
        }
    }
}

void initialize_divergence_advection_2d(
    std::vector<State>& U,
    const MHDRunParams& params
) {
    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error(
            "initialize_divergence_advection_2d: U size mismatch."
        );
    }

    const double xc = 0.35;
    const double yc = 0.5;
    const double alpha = 100.0;

    for (int j = 0; j < ny; ++j) {
        const double y = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double rx = x - xc;
            const double ry = y - yc;
            const double r2 = rx * rx + ry * ry;
            const double g = std::exp(-alpha * r2);

            const PrimState W(
                1.0,
                1.0,
                0.5,
                0.0,
                1.0,
                1.0 + 0.05 * g,
                0.025 * g,
                0.0,
                0.0
            );

            U[idx2d(i, j, nx)] = prim_to_state(W, gamma);
        }
    }
}

// =============================================================================
//  Main runner: RK2(HLLD) + GLM cleaning, with divB diagnostics.
// =============================================================================

void run_mhd_2d_case(
    CleaningType type,
    MHDRunParams params
) {
    // Derive grid spacing from domain size and resolution.
    params.glm.dx = params.glm.xlen / static_cast<double>(params.glm.nx);
    params.glm.dy = params.glm.ylen / static_cast<double>(params.glm.ny);

    const int    nx    = params.glm.nx;
    const int    ny    = params.glm.ny;
    const double dx    = params.glm.dx;
    const double dy    = params.glm.dy;
    const double t_end = params.glm.t_end;
    const double cfl   = params.glm.cfl;
    const double gamma = params.gamma;

    // Task C: stage projection pointer (only for ELLIPTIC_PROJECTION runs with
    // project_each_stage enabled).
    const bool do_stage_projection =
        (type == CleaningType::ELLIPTIC_PROJECTION) &&
        params.glm.project_each_stage;

    // Initialize state.
    std::vector<State> U(nx * ny);
    if (params.problem == "orszag_tang") {
        initialize_orszag_tang_2d(U, params);
    } else if (params.problem == "brio_wu") {
        initialize_brio_wu_strip_2d(U, params);
    } else if (params.problem == "field_loop") {
        initialize_field_loop_2d(U, params);
    } else if (params.problem == "divergence_advection") {
        initialize_divergence_advection_2d(U, params);
    } else {
        throw std::invalid_argument(
            "run_mhd_2d_case: unknown problem '" + params.problem + "'"
        );
    }

    // Freeze ch at the initial max signal speed (Dedner 2002 convention).
    const double ch_init = max_signal_speed_2d(U, gamma, 0.0, false);
    params.glm.ch = ch_init;

    const bool glm_active =
        (type == CleaningType::HYPERBOLIC_GLM
      || type == CleaningType::MIXED_GLM
      || type == CleaningType::MIXED_EGLM
      || type == CleaningType::GI_MIXED_EGLM);

    const std::string name   = cleaning_name(type);
    const std::string prefix = params.glm.out_prefix;

    fs::create_directories("results/mhd_runner/divergence");
    fs::create_directories("results/mhd_runner/snapshots");
    fs::create_directories("results/mhd_runner/summaries");

    const MHDRunDiagnostics initial_diag =
        compute_mhd_run_diagnostics(U, gamma, dx * dy);
    const double energy_initial = initial_diag.total_energy;

    // Initial snapshot (optional).
    if (params.glm.write_snapshot && params.glm.write_initial_snapshot) {
        const std::string snap =
            "results/mhd_runner/snapshots/" + prefix + "_" + name + "_initial.csv";
        write_mhd_2d_snapshot(U, params, snap);
        std::cout << "  Wrote " << snap << "\n";
    }

    // Diagnostics CSV.
    const std::string diag_name =
        "results/mhd_runner/divergence/" + prefix + "_" + name + ".csv";
    std::ofstream diag(diag_name);
    if (!diag) {
        throw std::runtime_error("Failed to open diagnostic file: " + diag_name);
    }
    diag << "step,time,dt,"
         << "L1_fv,L2_fv,Linf_fv,"
         << "L1_norm_fv,L2_norm_fv,Linf_norm_fv,"
         << "total_mass,total_momentum_x,total_momentum_y,total_momentum_z,"
         << "total_energy,min_density,min_pressure,"
         << "has_nonfinite,has_negative_density,has_negative_pressure,"
         << "cleaning_subcycles_step,"
         << "projection_iterations_step,"
         << "projection_solver_update_residual,"
         << "projection_final_residual,"
         << "projection_true_residual,"
         << "projection_converged\n";

    double t    = 0.0;
    int    step = 0;
    bool stopped_for_failure = false;
    std::string failure_reason;
    double failure_time = std::numeric_limits<double>::quiet_NaN();
    long long cleaning_subcycles_total = 0;
    long long projection_iterations_total = 0;
    bool printed_parabolic_subcycling = false;
    int last_cleaning_subcycles_step = 0;
    int last_projection_iterations_step = 0;
    double last_projection_final_residual =
        std::numeric_limits<double>::quiet_NaN();
    double last_projection_solver_update_residual =
        std::numeric_limits<double>::quiet_NaN();
    double last_projection_true_residual =
        std::numeric_limits<double>::quiet_NaN();
    bool last_projection_converged = true;

    // Summary-level stage diagnostics: track minimums over the whole run
    // (for the failure step specifically, and as global minimums).
    StagePressureMins run_stage_mins;
    double run_projection_theta = 1.0;
    int    run_total_retries   = 0;
    double run_min_dt_used     = std::numeric_limits<double>::infinity();

    while (true) {
        const LocalDivBNorms norms =
            compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
        const MHDRunDiagnostics run_diag =
            compute_mhd_run_diagnostics(U, gamma, dx * dy);

        write_mhd_diagnostic_row(
            diag,
            step,
            t,
            params.glm.dt,
            norms,
            run_diag,
            last_cleaning_subcycles_step,
            last_projection_iterations_step,
            last_projection_solver_update_residual,
            last_projection_final_residual,
            last_projection_true_residual,
            last_projection_converged
        );

        if (has_mhd_run_failure(run_diag)) {
            stopped_for_failure = true;
            failure_time = t;
            if (run_diag.has_nonfinite) {
                failure_reason = "nonfinite_state";
            } else if (run_diag.has_negative_density) {
                failure_reason = "negative_density";
            } else if (run_diag.has_negative_pressure) {
                failure_reason = "negative_pressure";
            }
            print_mhd_run_failure_warning(
                params,
                name,
                step,
                t,
                params.glm.dt,
                run_diag
            );
            break;
        }

        if (t >= t_end - 1e-12) break;

        const double smax = max_signal_speed_2d(U, gamma, ch_init, glm_active);
        double dt = cfl * std::min(dx, dy) / smax;
        dt = std::min(dt, t_end - t);
        params.glm.dt = dt;

        // =====================================================================
        // Task A: record min physical BEFORE the hydro step.
        // =====================================================================
        StagePressureMins step_stage_mins;
        {
            const MinPhysical mp = compute_min_physical(U, gamma);
            step_stage_mins.min_pressure_before_hydro = mp.min_pressure;
            step_stage_mins.min_density_before_hydro  = mp.min_density;
        }

        // =====================================================================
        // Tasks D + A: full-step retry loop.
        //
        // If the trial step (hydro + cleaning) produces non-positive raw pressure
        // or density, reject it, halve dt, and retry up to max_step_retries
        // (only for PARABOLIC and ELLIPTIC_PROJECTION which benefit from smaller
        // dt for the cleaning step).
        //
        // Do NOT hide the failure by applying pressure floors.
        // =====================================================================

        const std::vector<State> U_begin = U;
        const int max_retries =
            benefits_from_hydro_retry(type) ? params.glm.max_step_retries : 0;
        int total_retries = 0;
        double min_dt_used = dt;

        BadStateRecord after_hydro_bad;
        CleaningAdvanceStats cleaning_stats;

        while (true) {
            U = U_begin;
            params.glm.dt = dt;

            // ------------------------------------------------------------------
            // Step 1: HLLD finite-volume update.
            // Task C: pass stage GLM params when project_each_stage is enabled.
            // ------------------------------------------------------------------
            if (do_stage_projection) {
                rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt, &params.glm);
            } else {
                rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt, nullptr);
            }

            after_hydro_bad =
                scan_physical_state(
                    U,
                    nx,
                    ny,
                    dx,
                    dy,
                    gamma,
                    "after_hydro_step"
                );

            if (after_hydro_bad.found) {
                if (total_retries < max_retries) {
                    dt *= 0.5;
                    ++total_retries;
                    continue;
                }
                // Retries exhausted.
                break;
            }

            // ------------------------------------------------------------------
            // Step 2: divergence-cleaning update (any CleaningType).
            // ------------------------------------------------------------------
            cleaning_stats =
                apply_cleaning_with_subcycles(
                    U,
                    type,
                    params.glm,
                    params,
                    name,
                    step,
                    t,
                    dt,
                    gamma,
                    type == CleaningType::PARABOLIC &&
                        !printed_parabolic_subcycling
                );

            if (cleaning_stats.bad_state.found && total_retries < max_retries) {
                // Cleaning failed: halve dt and retry the full step.
                dt *= 0.5;
                ++total_retries;
                continue;
            }

            // Full step accepted (or retries exhausted).
            min_dt_used = dt;
            break;
        }

        // Record per-step stage minimums from the cleaning substep.
        if (after_hydro_bad.found) {
            step_stage_mins.min_pressure_after_hydro =
                std::min(step_stage_mins.min_pressure_before_hydro,
                         after_hydro_bad.pressure);
        } else {
            const MinPhysical mp_hydro = compute_min_physical(U, gamma);
            step_stage_mins.min_pressure_after_hydro = mp_hydro.min_pressure;
            step_stage_mins.min_density_after_hydro  = mp_hydro.min_density;
        }

        step_stage_mins.min_pressure_after_cleaning_B =
            cleaning_stats.stage_mins.min_pressure_after_cleaning_B;
        step_stage_mins.min_density_after_cleaning_B =
            cleaning_stats.stage_mins.min_density_after_cleaning_B;
        step_stage_mins.min_pressure_after_energy_repair =
            cleaning_stats.stage_mins.min_pressure_after_energy_repair;
        step_stage_mins.min_density_after_energy_repair =
            cleaning_stats.stage_mins.min_density_after_energy_repair;

        // Update global run minimums (for summary).
        run_stage_mins.min_pressure_before_hydro = std::min(
            run_stage_mins.min_pressure_before_hydro,
            step_stage_mins.min_pressure_before_hydro);
        run_stage_mins.min_pressure_after_hydro = std::min(
            run_stage_mins.min_pressure_after_hydro,
            step_stage_mins.min_pressure_after_hydro);
        run_stage_mins.min_pressure_after_cleaning_B = std::min(
            run_stage_mins.min_pressure_after_cleaning_B,
            step_stage_mins.min_pressure_after_cleaning_B);
        run_stage_mins.min_pressure_after_energy_repair = std::min(
            run_stage_mins.min_pressure_after_energy_repair,
            step_stage_mins.min_pressure_after_energy_repair);
        run_total_retries = std::max(run_total_retries, total_retries);
        run_min_dt_used   = std::min(run_min_dt_used, min_dt_used);
        if (cleaning_stats.projection_theta < run_projection_theta) {
            run_projection_theta = cleaning_stats.projection_theta;
        }

        const double next_t = t + dt;
        const int next_step = step + 1;

        if (total_retries > 0 && !after_hydro_bad.found &&
            !cleaning_stats.bad_state.found) {
            std::cout << "  [" << name << "] step retry accepted"
                      << "  retries=" << total_retries
                      << "  step=" << next_step
                      << "  dt=" << dt << "\n";
        }

        // ------------------------------------------------------------------
        // Handle hydro failure (after retries exhausted).
        // ------------------------------------------------------------------
        if (after_hydro_bad.found) {
            stopped_for_failure = true;
            failure_time = next_t;
            failure_reason =
                "hydro_positivity_failure:" + after_hydro_bad.reason
                + ":retries=" + std::to_string(total_retries);

            step_stage_mins.failure_stage = "after_hydro_step";
            run_stage_mins.failure_stage  = "after_hydro_step";

            write_cleaning_failure_csv(
                params,
                name,
                next_step,
                failure_time,
                after_hydro_bad,
                step_stage_mins,
                cleaning_stats.projection_theta,
                total_retries,
                min_dt_used
            );

            const LocalDivBNorms failed_norms =
                compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
            const MHDRunDiagnostics failed_diag =
                compute_mhd_run_diagnostics(U, gamma, dx * dy);

            write_mhd_diagnostic_row(
                diag,
                next_step,
                failure_time,
                params.glm.dt,
                failed_norms,
                failed_diag,
                0,
                0,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                true
            );

            std::cerr << "WARNING: stopping MHD run after hydro update produced bad state"
                      << "  problem=" << params.problem
                      << "  cleaning=" << name
                      << "  step=" << next_step
                      << "  time=" << failure_time
                      << "  reason=" << after_hydro_bad.reason
                      << "  cell=(" << after_hydro_bad.i
                      << "," << after_hydro_bad.j << ")"
                      << "  rho=" << after_hydro_bad.rho
                      << "  p=" << after_hydro_bad.pressure
                      << "  retries=" << total_retries
                      << "\n";

            t = failure_time;
            step = next_step;
            break;
        }

        // ------------------------------------------------------------------
        // Safety re-scan before cleaning (was "before_cleaning" in old code).
        // With the retry loop, this should always pass if after_hydro passed,
        // but we keep it as an explicit contract check.
        // ------------------------------------------------------------------
        const BadStateRecord before_cleaning_bad =
            scan_physical_state(
                U,
                nx,
                ny,
                dx,
                dy,
                gamma,
                "before_cleaning"
            );

        if (before_cleaning_bad.found) {
            stopped_for_failure = true;
            failure_time = next_t;
            failure_reason =
                "pre_cleaning_positivity_failure:" + before_cleaning_bad.reason;

            step_stage_mins.failure_stage = "after_hydro_step";
            run_stage_mins.failure_stage  = "after_hydro_step";

            write_cleaning_failure_csv(
                params,
                name,
                next_step,
                failure_time,
                before_cleaning_bad,
                step_stage_mins,
                1.0,
                total_retries,
                min_dt_used
            );

            const LocalDivBNorms failed_norms =
                compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
            const MHDRunDiagnostics failed_diag =
                compute_mhd_run_diagnostics(U, gamma, dx * dy);

            write_mhd_diagnostic_row(
                diag,
                next_step,
                failure_time,
                params.glm.dt,
                failed_norms,
                failed_diag,
                0,
                0,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                true
            );

            std::cerr << "WARNING: stopping MHD run before cleaning due to bad state"
                      << "  problem=" << params.problem
                      << "  cleaning=" << name
                      << "  step=" << next_step
                      << "  time=" << failure_time
                      << "  reason=" << before_cleaning_bad.reason
                      << "  cell=(" << before_cleaning_bad.i
                      << "," << before_cleaning_bad.j << ")"
                      << "  rho=" << before_cleaning_bad.rho
                      << "  p=" << before_cleaning_bad.pressure
                      << "\n";

            t = failure_time;
            step = next_step;
            break;
        }

        if (type == CleaningType::PARABOLIC && cleaning_stats.subcycles > 1) {
            printed_parabolic_subcycling = true;
        }

        cleaning_subcycles_total += cleaning_stats.subcycles;
        projection_iterations_total += cleaning_stats.projection_iterations;
        last_cleaning_subcycles_step = cleaning_stats.subcycles;
        last_projection_iterations_step = cleaning_stats.projection_iterations;
        last_projection_solver_update_residual =
            cleaning_stats.projection_solver_update_residual;
        last_projection_final_residual = cleaning_stats.projection_final_residual;
        last_projection_true_residual =
            cleaning_stats.projection_true_residual;
        last_projection_converged = cleaning_stats.projection_converged;

        t += dt;
        ++step;
        params.glm.dt = dt;

        // Task A: record min physical after the full step.
        {
            const MinPhysical mp = compute_min_physical(U, gamma);
            step_stage_mins.min_pressure_after_full_step = mp.min_pressure;
            step_stage_mins.min_density_after_full_step  = mp.min_density;
            run_stage_mins.min_pressure_after_full_step = std::min(
                run_stage_mins.min_pressure_after_full_step,
                mp.min_pressure);
        }

        if (cleaning_stats.bad_state.found) {
            stopped_for_failure = true;
            failure_time = cleaning_stats.failure_time;
            failure_reason =
                "cleaning_induced_failure:" + cleaning_stats.bad_state.reason
                + ":retries=" + std::to_string(total_retries);

            // Determine failure stage from per-stage minimums.
            step_stage_mins.failure_stage =
                determine_failure_stage(step_stage_mins);
            run_stage_mins.failure_stage = step_stage_mins.failure_stage;

            write_cleaning_failure_csv(
                params,
                name,
                step,
                failure_time,
                cleaning_stats.bad_state,
                step_stage_mins,
                cleaning_stats.projection_theta,
                total_retries,
                min_dt_used
            );

            t = failure_time;

            const LocalDivBNorms failed_norms =
                compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
            const MHDRunDiagnostics failed_diag =
                compute_mhd_run_diagnostics(U, gamma, dx * dy);

            write_mhd_diagnostic_row(
                diag,
                step,
                t,
                params.glm.dt,
                failed_norms,
                failed_diag,
                cleaning_stats.subcycles,
                cleaning_stats.projection_iterations,
                cleaning_stats.projection_solver_update_residual,
                cleaning_stats.projection_final_residual,
                cleaning_stats.projection_true_residual,
                cleaning_stats.projection_converged
            );

            std::cerr << "WARNING: stopping MHD run after cleaning produced bad state"
                      << "  problem=" << params.problem
                      << "  cleaning=" << name
                      << "  step=" << step
                      << "  time=" << t
                      << "  reason=" << failure_reason
                      << "  cell=(" << cleaning_stats.bad_state.i
                      << "," << cleaning_stats.bad_state.j << ")"
                      << "  rho=" << cleaning_stats.bad_state.rho
                      << "  p=" << cleaning_stats.bad_state.pressure
                      << "  failure_stage=" << step_stage_mins.failure_stage
                      << "  theta=" << cleaning_stats.projection_theta
                      << "  retries=" << total_retries
                      << "\n";
            break;
        }

        if (step % 200 == 0) {
            std::cout << "  [" << name << "] step=" << step
                      << "  t=" << t << "  dt=" << dt << "\n";
        }
    }

    if (stopped_for_failure) {
        std::cout << "  [" << name << "] stopped after failure: " << step
                  << " steps, t=" << t << "\n";
    } else {
        std::cout << "  [" << name << "] finished: " << step
                  << " steps, t=" << t << "\n";
    }

    // Final snapshot.
    if (params.glm.write_snapshot) {
        const std::string snap =
            "results/mhd_runner/snapshots/" + prefix + "_" + name + "_final.csv";
        write_mhd_2d_snapshot(U, params, snap);
        std::cout << "  Wrote " << snap << "\n";
    }

    const LocalDivBNorms final_norms =
        compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
    const MHDRunDiagnostics final_diag =
        compute_mhd_run_diagnostics(U, gamma, dx * dy);
    const bool final_time_reached =
        !stopped_for_failure && t >= t_end - 1.0e-12;

    write_mhd_run_summary(
        params,
        name,
        final_time_reached,
        failure_time,
        failure_reason,
        final_norms,
        energy_initial,
        final_diag,
        cleaning_subcycles_total,
        projection_iterations_total,
        params.glm.energy_policy,
        last_projection_true_residual,
        run_stage_mins,
        run_projection_theta,
        run_total_retries,
        run_min_dt_used
    );

    std::cout << "  Wrote " << diag_name << "\n";
}
