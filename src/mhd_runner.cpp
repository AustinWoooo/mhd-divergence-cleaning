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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "HLLD_mhd_solver.hpp"
#include "mhd_reconstruction.hpp"
#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_common.hpp"
#include "mpi_domain.hpp"
#include "projection2d.hpp"

namespace fs = std::filesystem;
using namespace MHD;

using Clock = std::chrono::steady_clock;

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
    double min_raw_pressure = std::numeric_limits<double>::infinity();
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
    double actual_ch = std::numeric_limits<double>::quiet_NaN();
    double actual_cp = std::numeric_limits<double>::quiet_NaN();
    double effective_cd = std::numeric_limits<double>::quiet_NaN();
    double effective_cr = std::numeric_limits<double>::quiet_NaN();
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

struct ProjectionCorrectionSummary {
    double total_correction = 0.0;
    double outside_causal_correction = 0.0;
    double outside_fraction = 0.0;
    std::string snapshot_file;
};

struct MHDRunTiming {
    double total_wall_time_sec = 0.0;
    double initialization_time_sec = 0.0;
    double hydro_time_sec = 0.0;
    double cleaning_time_sec = 0.0;
    double diagnostics_compute_time_sec = 0.0;
    double diagnostics_write_time_sec = 0.0;
    double snapshot_write_time_sec = 0.0;
    double summary_write_time_sec = 0.0;
    double output_time_sec = 0.0;
    long long steps = 0;
    long long total_cell_updates = 0;
    double seconds_per_step = 0.0;
    double cell_updates_per_second = 0.0;
};

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void finalize_timing_fields(
    MHDRunTiming& timing,
    int nx,
    int ny
) {
    timing.total_cell_updates =
        static_cast<long long>(nx) * static_cast<long long>(ny) * timing.steps;
    timing.output_time_sec =
        timing.diagnostics_write_time_sec
      + timing.snapshot_write_time_sec
      + timing.summary_write_time_sec;
    timing.seconds_per_step =
        (timing.steps > 0)
        ? timing.total_wall_time_sec / static_cast<double>(timing.steps)
        : 0.0;
    timing.cell_updates_per_second =
        (timing.total_wall_time_sec > 0.0)
        ? static_cast<double>(timing.total_cell_updates)
            / timing.total_wall_time_sec
        : 0.0;
}

fs::path runner_output_path(
    const MHDRunParams& params,
    const std::string& subdir,
    const std::string& filename
) {
    return fs::path(params.output_root) / subdir / filename;
}

bool should_write_diagnostic_row(
    int step,
    double time,
    double t_end,
    bool failure,
    int stride
) {
    if (step == 0 || failure || time >= t_end - 1.0e-12) {
        return true;
    }
    const int s = std::max(1, stride);
    return step % s == 0;
}

std::string reconstruction_name(MHD::Reconstruction reconstruction) {
    switch (reconstruction) {
        case MHD::Reconstruction::PCM:
            return "pcm";
        case MHD::Reconstruction::PLM:
            return "plm";
    }
    return "unknown";
}

std::string limiter_name(MHD::SlopeLimiter limiter) {
    switch (limiter) {
        case MHD::SlopeLimiter::MINMOD:
            return "minmod";
        case MHD::SlopeLimiter::VANLEER:
            return "vanleer";
        case MHD::SlopeLimiter::MC:
            return "mc";
    }
    return "unknown";
}

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

struct CellThermo {
    double rho = std::numeric_limits<double>::quiet_NaN();
    double vx = std::numeric_limits<double>::quiet_NaN();
    double vy = std::numeric_limits<double>::quiet_NaN();
    double vz = std::numeric_limits<double>::quiet_NaN();
    double kinetic = std::numeric_limits<double>::quiet_NaN();
    double magnetic = std::numeric_limits<double>::quiet_NaN();
    double thermal = std::numeric_limits<double>::quiet_NaN();
    double pressure = std::numeric_limits<double>::quiet_NaN();
};

CellThermo decompose_cell(const State& cell, double gamma) {
    CellThermo out;
    out.rho = cell[RHO];
    out.magnetic = magnetic_energy_density(cell);

    if (std::isfinite(out.rho) && out.rho > 0.0) {
        out.vx = cell[MX] / out.rho;
        out.vy = cell[MY] / out.rho;
        out.vz = cell[MZ] / out.rho;
        out.kinetic = kinetic_energy_density(cell);
        out.thermal = cell[E] - out.kinetic - out.magnetic;
        out.pressure = (gamma - 1.0) * out.thermal;
    }

    return out;
}

// Task A helper: compute min raw pressure and min density from a state vector.
// Both fields are reduced globally when a decomposition domain is supplied so
// every rank agrees; min is idempotent, so running over a ghost-padded local
// array (halo cells repeat their owner's values) is harmless.  domain == nullptr
// (serial) makes global_min an identity.
MinPhysical compute_min_physical(
    const std::vector<State>& U,
    double gamma,
    const MPIDomain* domain = nullptr
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
    out.min_pressure = global_min(out.min_pressure, domain);
    out.min_density = global_min(out.min_density, domain);
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

BadStateRecord scan_raw_primitive_state(
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
            const PrimState W = state_to_prim(cell, gamma);

            std::string reason;
            if (!std::isfinite(W.rho) || !std::isfinite(W.p)) {
                reason = "nonfinite_state";
            } else if (W.rho <= 0.0) {
                reason = "non_positive_density";
            } else if (W.p <= 0.0) {
                reason = "non_positive_pressure";
            }

            if (!reason.empty()) {
                const double ke = kinetic_energy_density(cell);
                const double me = magnetic_energy_density(cell);
                out.found = true;
                out.stage = stage;
                out.reason = reason;
                out.i = i;
                out.j = j;
                out.rho = W.rho;
                out.pressure = W.p;
                out.total_energy = cell[E];
                out.kinetic_energy = ke;
                out.magnetic_energy = me;
                out.internal_energy = W.p / (gamma - 1.0);
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
        runner_output_path(
            params,
            "failures",
            params.glm.out_prefix
          + "_"
          + method
          + "_"
          + cleaning_energy_policy_name(params.glm.energy_policy)
          + "_failure.csv"
        ).string();
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

void write_powell_first_bad_cell_diagnostic(
    const MHDRunParams& params,
    const std::string& method_name,
    int step,
    double time,
    double dt,
    int i,
    int j,
    const State& before,
    const State& after,
    double divB
) {
    const std::string filename =
        runner_output_path(
            params,
            "failures",
            params.glm.out_prefix
          + "_"
          + method_name
          + "_first_bad_cell_diagnostic.csv"
        ).string();

    ensure_parent_directory(filename);

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error(
            "Failed to open Powell bad-cell diagnostic file: " + filename
        );
    }

    const CellThermo b = decompose_cell(before, params.gamma);
    const CellThermo a = decompose_cell(after, params.gamma);
    const double s = -dt * divB;
    const double u_dot_B_before =
        b.vx * before[BX] + b.vy * before[BY] + b.vz * before[BZ];

    out << "step,time,dt,i,j,"
        << "rho_before,rho_after,"
        << "vx_before,vy_before,vz_before,"
        << "vx_after,vy_after,vz_after,"
        << "Bx_before,By_before,Bz_before,"
        << "Bx_after,By_after,Bz_after,"
        << "E_before,E_after,"
        << "kinetic_before,kinetic_after,"
        << "magnetic_before,magnetic_after,"
        << "thermal_before,thermal_after,"
        << "p_before,p_after,"
        << "divB,"
        << "s_minus_dt_divB,"
        << "delta_mx,delta_my,delta_mz,"
        << "delta_Bx,delta_By,delta_Bz,"
        << "delta_E,"
        << "u_dot_B_before\n";

    out << step << "," << time << "," << dt << ","
        << i << "," << j << ","
        << before[RHO] << "," << after[RHO] << ","
        << b.vx << "," << b.vy << "," << b.vz << ","
        << a.vx << "," << a.vy << "," << a.vz << ","
        << before[BX] << "," << before[BY] << "," << before[BZ] << ","
        << after[BX] << "," << after[BY] << "," << after[BZ] << ","
        << before[E] << "," << after[E] << ","
        << b.kinetic << "," << a.kinetic << ","
        << b.magnetic << "," << a.magnetic << ","
        << b.thermal << "," << a.thermal << ","
        << b.pressure << "," << a.pressure << ","
        << divB << ","
        << s << ","
        << after[MX] - before[MX] << ","
        << after[MY] - before[MY] << ","
        << after[MZ] - before[MZ] << ","
        << after[BX] - before[BX] << ","
        << after[BY] - before[BY] << ","
        << after[BZ] - before[BZ] << ","
        << after[E] - before[E] << ","
        << u_dot_B_before << "\n";
}

MHDRunDiagnostics compute_mhd_run_diagnostics(
    const std::vector<State>& U,
    double gamma,
    double cell_area,
    const MPIDomain* domain = nullptr
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

        PrimitiveRecoveryStatus recovery;
        const PrimState W_checked =
            PrimState::from_conserved_checked(cell, gamma, &recovery);

        if (!std::isfinite(recovery.raw_pressure)) {
            out.has_nonfinite = true;
        } else {
            out.min_raw_pressure =
                std::min(out.min_raw_pressure, recovery.raw_pressure);
            if (recovery.raw_pressure < 0.0) {
                out.has_negative_pressure = true;
            }
        }
        if (!std::isfinite(W_checked.p)) {
            out.has_nonfinite = true;
        } else {
            out.min_pressure = std::min(out.min_pressure, W_checked.p);
        }
    }

    // Reduce only the evolution-gating quantities (failure flags + minima) so
    // every rank reaches the same has_mhd_run_failure() verdict each step; these
    // are idempotent (OR / min) and safe to evaluate over a ghost-padded local
    // array.  The conservation SUMS (mass/momentum/energy) are deliberately left
    // rank-local here: under domain decomposition they are recomputed on the root
    // rank from the gathered global field (see run_mhd_2d_case), because summing
    // the padded local array would double-count halo cells.  domain == nullptr
    // (serial) makes every global_* an identity, so this is a no-op.
    out.has_nonfinite = global_lor(out.has_nonfinite ? 1 : 0, domain) != 0;
    out.has_negative_density =
        global_lor(out.has_negative_density ? 1 : 0, domain) != 0;
    out.has_negative_pressure =
        global_lor(out.has_negative_pressure ? 1 : 0, domain) != 0;
    out.min_density = global_min(out.min_density, domain);
    out.min_raw_pressure = global_min(out.min_raw_pressure, domain);
    out.min_pressure = global_min(out.min_pressure, domain);
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

bool projection_correction_diagnostics_active(
    const MHDRunParams& params,
    CleaningType type
) {
    return params.write_projection_diagnostics
        && params.problem == "blast_wave"
        && type == CleaningType::ELLIPTIC_PROJECTION;
}

double blast_wave_radius() {
    return 0.1;
}

fs::path projection_diagnostics_dir(const MHDRunParams& params) {
    return fs::path(params.output_root) / "projection_diagnostics";
}

fs::path projection_diagnostics_summary_path(
    const MHDRunParams& params,
    const std::string& method_name
) {
    return projection_diagnostics_dir(params)
        / (params.glm.out_prefix + "_" + method_name
           + "_projection_causal_summary.csv");
}

void initialize_projection_diagnostics_summary_csv(
    const MHDRunParams& params,
    const std::string& method_name
) {
    fs::create_directories(projection_diagnostics_dir(params));
    const fs::path path =
        projection_diagnostics_summary_path(params, method_name);

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Failed to open projection diagnostics summary: " + path.string()
        );
    }

    out << "problem,method,step,substep,time,r0,xc,yc,"
        << "max_cfast_seen,r_causal,"
        << "total_projection_correction,"
        << "outside_causal_projection_correction,"
        << "outside_fraction,snapshot_file\n";
}

std::string projection_diagnostics_snapshot_name(
    const MHDRunParams& params,
    const std::string& method_name,
    int step,
    int substep
) {
    std::ostringstream name;
    name << params.glm.out_prefix
         << "_" << method_name
         << "_projection_step"
         << std::setw(6) << std::setfill('0') << step
         << "_sub"
         << std::setw(2) << std::setfill('0') << substep
         << ".csv";
    return name.str();
}

ProjectionCorrectionSummary write_projection_correction_diagnostic(
    const std::vector<State>& before_projection,
    const std::vector<State>& after_projection,
    const MHDRunParams& params,
    const std::string& method_name,
    int step,
    int substep,
    double time,
    double max_cfast_seen,
    bool write_grid
) {
    ProjectionCorrectionSummary summary;

    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;
    const double xc = 0.5 * params.glm.xlen;
    const double yc = 0.5 * params.glm.ylen;
    const double r0 = blast_wave_radius();
    const double r_causal = r0 + time * max_cfast_seen;

    fs::path snapshot_path;
    std::ofstream grid;
    if (write_grid) {
        fs::create_directories(projection_diagnostics_dir(params));
        const std::string snapshot_name =
            projection_diagnostics_snapshot_name(
                params,
                method_name,
                step,
                substep
            );
        snapshot_path = projection_diagnostics_dir(params) / snapshot_name;
        grid.open(snapshot_path);
        if (!grid) {
            throw std::runtime_error(
                "Failed to open projection diagnostic snapshot: "
                + snapshot_path.string()
            );
        }
        summary.snapshot_file = snapshot_name;
        grid << "i,j,x,y,r,outside_causal,"
             << "Bx_before_projection,By_before_projection,"
             << "Bx_after_projection,By_after_projection,"
             << "dBx_projection,dBy_projection,abs_dB_projection,"
             << "divB_before_projection,divB_after_projection,"
             << "pressure_after_cleaning_B\n";
    }

    for (int j = 0; j < ny; ++j) {
        const double y = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);
            const double x = (i + 0.5) * dx;
            const double rx = x - xc;
            const double ry = y - yc;
            const double r = std::sqrt(rx * rx + ry * ry);

            const double bx_before = before_projection[id][BX];
            const double by_before = before_projection[id][BY];
            const double bx_after = after_projection[id][BX];
            const double by_after = after_projection[id][BY];
            const double dbx = bx_after - bx_before;
            const double dby = by_after - by_before;
            const double abs_db = std::sqrt(dbx * dbx + dby * dby);
            const double corr2 = abs_db * abs_db;
            const bool outside = r > r_causal;

            summary.total_correction += corr2;
            if (outside) {
                summary.outside_causal_correction += corr2;
            }

            if (write_grid) {
                const double div_before =
                    compute_fv_divB_cell_2d(
                        before_projection,
                        nx,
                        ny,
                        i,
                        j,
                        dx,
                        dy
                    );
                const double div_after =
                    compute_fv_divB_cell_2d(
                        after_projection,
                        nx,
                        ny,
                        i,
                        j,
                        dx,
                        dy
                    );
                const PrimState W_after =
                    state_to_prim(after_projection[id], gamma);

                grid << i << "," << j << ","
                     << x << "," << y << ","
                     << r << "," << (outside ? 1 : 0) << ","
                     << bx_before << "," << by_before << ","
                     << bx_after << "," << by_after << ","
                     << dbx << "," << dby << "," << abs_db << ","
                     << div_before << "," << div_after << ","
                     << W_after.p << "\n";
            }
        }
    }

    if (summary.total_correction > 0.0) {
        summary.outside_fraction =
            summary.outside_causal_correction / summary.total_correction;
    }

    const fs::path summary_path =
        projection_diagnostics_summary_path(params, method_name);
    std::ofstream out(summary_path, std::ios::app);
    if (!out) {
        throw std::runtime_error(
            "Failed to append projection diagnostics summary: "
            + summary_path.string()
        );
    }

    out << params.problem << ","
        << method_name << ","
        << step << ","
        << substep << ","
        << time << ","
        << r0 << ","
        << xc << ","
        << yc << ","
        << max_cfast_seen << ","
        << r_causal << ","
        << summary.total_correction << ","
        << summary.outside_causal_correction << ","
        << summary.outside_fraction << ","
        << summary.snapshot_file << "\n";

    return summary;
}

bool is_glm_cleaning(CleaningType type) {
    return type == CleaningType::HYPERBOLIC_GLM
        || type == CleaningType::MIXED_GLM
        || type == CleaningType::MIXED_EGLM
        || type == CleaningType::GI_MIXED_EGLM;
}

bool uses_glm_damping(CleaningType type) {
    return type == CleaningType::MIXED_GLM
        || type == CleaningType::MIXED_EGLM
        || type == CleaningType::GI_MIXED_EGLM;
}

void validate_glm_tuning_params(const GLM2DParams& params) {
    if (params.glm_ch_factor <= 0.0) {
        throw std::invalid_argument("glm_ch_factor must be positive");
    }
    if (std::isfinite(params.glm_cd) &&
        !(params.glm_cd > 0.0 && params.glm_cd < 1.0)) {
        throw std::invalid_argument("glm_cd must satisfy 0 < glm_cd < 1");
    }
    if (std::isfinite(params.glm_cr) && !(params.glm_cr > 0.0)) {
        throw std::invalid_argument("glm_cr must be positive");
    }
    if (std::isfinite(params.glm_cd) && std::isfinite(params.glm_cr)) {
        throw std::invalid_argument("set only one of glm_cd or glm_cr");
    }
    if (params.glm_subcycles < 1) {
        throw std::invalid_argument("glm_subcycles must be >= 1");
    }
}

double cp_from_cd(double cd, double ch, double dt) {
    return std::sqrt(-(dt * ch * ch) / std::log(cd));
}

double effective_glm_cd(double cp, double ch, double dt) {
    return std::exp(-dt * ch * ch / (cp * cp));
}

void apply_glm_damping_parameterization(
    CleaningType type,
    GLM2DParams& params,
    double dt_clean,
    CleaningAdvanceStats& stats
) {
    if (!is_glm_cleaning(type)) {
        return;
    }

    stats.actual_ch = params.ch;

    if (!uses_glm_damping(type)) {
        return;
    }

    if (std::isfinite(params.glm_cd)) {
        params.cp = cp_from_cd(params.glm_cd, params.ch, dt_clean);
    } else if (std::isfinite(params.glm_cr)) {
        params.cp = std::sqrt(params.glm_cr * params.ch);
    }

    stats.actual_cp = params.cp;
    stats.effective_cd = effective_glm_cd(params.cp, params.ch, dt_clean);
    stats.effective_cr = params.cp * params.cp / params.ch;
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
    bool print_parabolic_subcycling,
    double max_cfast_seen
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
    if (is_glm_cleaning(type)) {
        nsub = std::max(nsub, params.glm_subcycles);
    }

    stats.subcycles = nsub;
    if (is_glm_cleaning(type)) {
        stats.actual_ch = params.ch;
    }

    if (type == CleaningType::PARABOLIC &&
        nsub > 1 &&
        print_parabolic_subcycling) {
        std::cout << "  [" << method_name << "] cleaning subcycles="
                  << nsub << " at step=" << step << "\n";
    }

    const double dt_sub = dt_mhd / static_cast<double>(nsub);

    for (int sub = 0; sub < nsub; ++sub) {
        params.dt = dt_sub;
        apply_glm_damping_parameterization(type, params, dt_sub, stats);

        std::vector<State> Uold;
        std::vector<State> Ubefore_powell;
        std::vector<State> Ubefore_projection;
        const bool repair_energy =
            params.energy_policy == CleaningEnergyPolicy::PreserveThermalPressure
         && pressure_preserving_policy_applies(type);
        const bool write_projection_diagnostics =
            projection_correction_diagnostics_active(run_params, type);

        if (repair_energy) {
            Uold = U;
        }
        if (type == CleaningType::POWELL_SOURCE) {
            Ubefore_powell = U;
        }
        if (write_projection_diagnostics) {
            Ubefore_projection = U;
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

        if (write_projection_diagnostics) {
            const double diagnostic_time =
                time + dt_sub * static_cast<double>(sub + 1);
            const int diagnostic_step = step + 1;
            const bool write_grid =
                should_write_diagnostic_row(
                    diagnostic_step,
                    diagnostic_time,
                    params.t_end,
                    false,
                    std::max(1, run_params.diagnostic_stride)
                );
            write_projection_correction_diagnostic(
                Ubefore_projection,
                U,
                run_params,
                method_name,
                diagnostic_step,
                sub + 1,
                diagnostic_time,
                max_cfast_seen,
                write_grid
            );
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
            if (type == CleaningType::POWELL_SOURCE &&
                bad.reason == "non_positive_pressure" &&
                !Ubefore_powell.empty()) {
                const int id = idx2d(bad.i, bad.j, params.nx);
                const double divB_before =
                    compute_fv_divB_cell_2d(
                        Ubefore_powell,
                        params.nx,
                        params.ny,
                        bad.i,
                        bad.j,
                        params.dx,
                        params.dy
                    );
                write_powell_first_bad_cell_diagnostic(
                    run_params,
                    method_name,
                    step + 1,
                    time + dt_sub * static_cast<double>(sub + 1),
                    dt_sub,
                    bad.i,
                    bad.j,
                    Ubefore_powell[id],
                    U[id],
                    divB_before
                );
            }
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
    bool include_ch,
    const MPIDomain* domain = nullptr
) {
    double smax = 0.0;
    const std::ptrdiff_t ncell = static_cast<std::ptrdiff_t>(U.size());
#pragma omp parallel for schedule(static) reduction(max : smax)
    for (std::ptrdiff_t id = 0; id < ncell; ++id) {
        const State& s = U[id];
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
    // CFL must be collective: every rank advances with the same dt.  global_max
    // is exact and order-independent, and harmless over ghost-padded arrays
    // (halo cells repeat values already counted on their owning rank).
    return global_max(smax, domain);
}

// -----------------------------------------------------------------------------
//  Finite-volume RHS = -div(F) on a periodic grid.
//  Reconstructs primitive interface states (PCM or MUSCL/PLM via
//  mhd_reconstruction.hpp), then calls MHD::compute_flux (HLLD) at every
//  interface, falling back to first-order states with the diffusive
//  MHD::compute_llf_flux on faces flagged in the LLF masks. Does not
//  re-implement either Riemann solver.
// -----------------------------------------------------------------------------
//  Face-flux storage convention:
//    flux_x[idx2d(i, j, nx)] = numerical flux between cells (i-1, j) and (i, j)
//    flux_y[idx2d(i, j, nx)] = numerical flux between cells (i, j-1) and (i, j)
//
//  llf_face_x[face] / llf_face_y[face] select the diffusive LLF flux on that
//  face. A face flux is shared by both adjacent cells, so swapping HLLD <-> LLF
//  on a face keeps the conservative update exactly conservative.
//
//  When mass_flux_x_out / mass_flux_y_out are non-null they receive the per-face
//  mass flux F[RHO] actually used (HLLD or LLF). The dual-energy entropy is
//  advected with these same mass fluxes so it stays consistent with density.

std::vector<State> compute_rhs_blended_2d(
    const std::vector<State>& U,
    const std::vector<char>& llf_face_x,
    const std::vector<char>& llf_face_y,
    int nx, int ny,
    double dx, double dy,
    double gamma,
    Reconstruction recon,
    SlopeLimiter limiter,
    std::vector<double>* mass_flux_x_out = nullptr,
    std::vector<double>* mass_flux_y_out = nullptr
) {
    const int ncell = nx * ny;

    // Cell-centred primitives, computed once per cell and reused on every face
    // (and as the base for the MUSCL slopes below).
    std::vector<PrimState> W(ncell);
#pragma omp parallel for schedule(static)
    for (int id = 0; id < ncell; ++id) {
        W[id] = state_to_prim(U[id], gamma);
    }

    // Limited primitive slopes for piecewise-linear (MUSCL) reconstruction.
    // For PCM they stay zero, so reconstruct_face() returns the cell averages
    // and the scheme reduces to first-order Godunov.
    std::vector<PrimState> slope_x(ncell), slope_y(ncell);
    if (recon == Reconstruction::PLM) {
#pragma omp parallel for schedule(static)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = idx2d(i, j, nx);
                const int iL = periodic_index(i - 1, nx);
                const int iR = periodic_index(i + 1, nx);
                slope_x[id] = limited_slope_prim(
                    W[idx2d(iL, j, nx)], W[id], W[idx2d(iR, j, nx)], limiter);
                const int jL = periodic_index(j - 1, ny);
                const int jR = periodic_index(j + 1, ny);
                slope_y[id] = limited_slope_prim(
                    W[idx2d(i, jL, nx)], W[id], W[idx2d(i, jR, nx)], limiter);
            }
        }
    }

    std::vector<State> flux_x(ncell);
    std::vector<State> flux_y(ncell);

    // X-direction Riemann problems. Each face writes a unique flux_x[face]
    // and compute_flux/compute_llf_flux are pure, so the iterations are
    // independent and safe to run concurrently.
    //
    // On faces flagged for the positivity fallback we deliberately drop to the
    // first-order (cell-average) states with the diffusive LLF flux; that is the
    // most robust combination and keeps the fallback genuinely monotone.
#pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int iL = periodic_index(i - 1, nx);
            const int face = idx2d(i, j, nx);
            const int idL = idx2d(iL, j, nx);
            const int idR = idx2d(i,  j, nx);
            if (llf_face_x[face]) {
                flux_x[face] = compute_llf_flux(W[idL], W[idR],
                                                /*direction=*/0, gamma);
            } else {
                const PrimState WL = reconstruct_face(W[idL], slope_x[idL], +1.0);
                const PrimState WR = reconstruct_face(W[idR], slope_x[idR], -1.0);
                flux_x[face] = compute_flux(WL, WR, /*direction=*/0, gamma);
            }
            if (mass_flux_x_out) {
                (*mass_flux_x_out)[face] = flux_x[face][RHO];
            }
        }
    }

    // Y-direction Riemann problems (same independence as the X sweep).
#pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        const int jL = periodic_index(j - 1, ny);
        for (int i = 0; i < nx; ++i) {
            const int face = idx2d(i, j, nx);
            const int idL = idx2d(i, jL, nx);
            const int idR = idx2d(i, j,  nx);
            if (llf_face_y[face]) {
                flux_y[face] = compute_llf_flux(W[idL], W[idR],
                                                /*direction=*/1, gamma);
            } else {
                const PrimState WL = reconstruct_face(W[idL], slope_y[idL], +1.0);
                const PrimState WR = reconstruct_face(W[idR], slope_y[idR], -1.0);
                flux_y[face] = compute_flux(WL, WR, /*direction=*/1, gamma);
            }
            if (mass_flux_y_out) {
                (*mass_flux_y_out)[face] = flux_y[face][RHO];
            }
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
//  Dual-energy formalism (Option C).
//
//  In the low-beta / cold core of Orszag-Tang the thermal pressure is a tiny
//  difference of large numbers, p = (gamma-1)(E - ke - me), so the conservative
//  total-energy update loses all significance there and can drive p < 0 even
//  with the LLF positivity fallback. We therefore carry a second thermodynamic
//  variable -- the conserved entropy density
//
//      Sc = p * rho^(1-gamma)        (so p = Sc * rho^(gamma-1))
//
//  which is simply advected by the flow in smooth regions (no source term) and
//  stays positive under upwind advection. After the hydro step, cells where the
//  total-energy internal energy is a negligible fraction of the total energy
//  recover p from the advected entropy instead, and rewrite E to match. Shocks
//  (large internal-energy fraction) keep using the total energy, which captures
//  the correct entropy jump. Reference: Bryan et al. 1995 (ENZO dual energy).
// -----------------------------------------------------------------------------

// Switch threshold: trust the total energy when e_int / E_total exceeds this,
// otherwise fall back to the advected entropy. Bryan et al. use ~1e-3.
constexpr double DUAL_ENERGY_ETA = 1.0e-3;

inline double conserved_entropy(double rho, double p, double gamma) {
    return p * std::pow(rho, 1.0 - gamma);
}

inline double pressure_from_entropy(double Sc, double rho, double gamma) {
    return Sc * std::pow(rho, gamma - 1.0);
}

// Build the entropy field from the (positive) start-of-step state. Because the
// previous step's recovery left every cell with a correct positive pressure,
// reconstructing Sc here is equivalent to having persisted and advected it.
std::vector<double> conserved_entropy_field(
    const std::vector<State>& U,
    double gamma
) {
    const int ncell = static_cast<int>(U.size());
    std::vector<double> Sc(ncell, 0.0);
    for (int id = 0; id < ncell; ++id) {
        const double rho = U[id][RHO];
        if (!(rho > 0.0) || !std::isfinite(rho)) {
            continue;
        }
        const double ke =
            0.5 * (U[id][MX]*U[id][MX] + U[id][MY]*U[id][MY]
                 + U[id][MZ]*U[id][MZ]) / rho;
        const double me =
            0.5 * (U[id][BX]*U[id][BX] + U[id][BY]*U[id][BY]
                 + U[id][BZ]*U[id][BZ]);
        const double p = (gamma - 1.0) * (U[id][E] - ke - me);
        Sc[id] = conserved_entropy(rho, std::max(p, 0.0), gamma);
    }
    return Sc;
}

// Entropy-density RHS = -div(F_S) with the consistent upwind passive-scalar
// flux  F_S(face) = mass_flux(face) * K_upwind,  K = Sc/rho. Using the same mass
// flux that advanced the density gives Sc a discrete maximum principle, so Sc
// stays positive whenever rho does.
std::vector<double> entropy_rhs_from_mass_flux(
    const std::vector<State>& U,
    const std::vector<double>& Sc,
    const std::vector<double>& mass_flux_x,
    const std::vector<double>& mass_flux_y,
    int nx, int ny,
    double dx, double dy
) {
    const int ncell = nx * ny;

    auto K_of = [&](int id) {
        const double rho = U[id][RHO];
        return (rho > 0.0 && std::isfinite(rho)) ? Sc[id] / rho : 0.0;
    };

    std::vector<double> fS_x(ncell, 0.0);
    std::vector<double> fS_y(ncell, 0.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int iL = periodic_index(i - 1, nx);
            const int face = idx2d(i, j, nx);
            const double fm = mass_flux_x[face];
            const double K_up = (fm >= 0.0) ? K_of(idx2d(iL, j, nx))
                                            : K_of(idx2d(i, j, nx));
            fS_x[face] = fm * K_up;
        }
    }
    for (int j = 0; j < ny; ++j) {
        const int jL = periodic_index(j - 1, ny);
        for (int i = 0; i < nx; ++i) {
            const int face = idx2d(i, j, nx);
            const double fm = mass_flux_y[face];
            const double K_up = (fm >= 0.0) ? K_of(idx2d(i, jL, nx))
                                            : K_of(idx2d(i, j, nx));
            fS_y[face] = fm * K_up;
        }
    }

    std::vector<double> rhs(ncell, 0.0);
    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);
        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);
            const int id = idx2d(i, j, nx);
            rhs[id] = -(fS_x[idx2d(ip, j, nx)] - fS_x[id]) / dx
                      -(fS_y[idx2d(i, jp, nx)] - fS_y[id]) / dy;
        }
    }
    return rhs;
}

// Dual-energy recovery: in cells where the total-energy internal energy is a
// negligible fraction of the total energy, recover p from the advected entropy
// and rewrite E so the conserved state is consistent and pressure-positive.
// Elsewhere the total energy is left untouched (it is the conserved primary).
void apply_dual_energy_recovery(
    std::vector<State>& U,
    const std::vector<double>& Sc,
    double gamma
) {
    const int ncell = static_cast<int>(U.size());
    for (int id = 0; id < ncell; ++id) {
        const double rho = U[id][RHO];
        if (!(rho > 0.0) || !std::isfinite(rho)) {
            continue;  // density failures are handled by the runner's scan
        }
        const double ke =
            0.5 * (U[id][MX]*U[id][MX] + U[id][MY]*U[id][MY]
                 + U[id][MZ]*U[id][MZ]) / rho;
        const double me =
            0.5 * (U[id][BX]*U[id][BX] + U[id][BY]*U[id][BY]
                 + U[id][BZ]*U[id][BZ]);
        const double e_int = U[id][E] - ke - me;

        // Trust the total energy in shocks / thermally-dominated cells.
        if (std::isfinite(e_int) && e_int > DUAL_ENERGY_ETA * U[id][E]) {
            continue;
        }

        // Low-beta / cold cell: recover pressure from the advected entropy.
        const double p = pressure_from_entropy(Sc[id], rho, gamma);
        if (std::isfinite(p) && p > 0.0) {
            U[id][E] = p / (gamma - 1.0) + ke + me;
        }
    }
}

// Raw positivity test on a conserved cell: finite, positive density, and
// positive thermal pressure (no flooring).
bool cell_is_physical(const State& s, double gamma) {
    const double rho = s[RHO];
    if (!std::isfinite(rho) || rho <= 0.0) {
        return false;
    }
    const double ke =
        0.5 * (s[MX]*s[MX] + s[MY]*s[MY] + s[MZ]*s[MZ]) / rho;
    const double me =
        0.5 * (s[BX]*s[BX] + s[BY]*s[BY] + s[BZ]*s[BZ]);
    const double p = (gamma - 1.0) * (s[E] - ke - me);
    return std::isfinite(p) && p > 0.0;
}

// -----------------------------------------------------------------------------
//  Positivity-preserving stage update (Option A).
//
//  Forms  new[id] = base[id] + coeff * dt * RHS(flux_state)  using HLLD fluxes.
//  Any cell whose updated state has non-positive density or pressure has the
//  four faces bounding it switched to the diffusive LLF flux, and the update is
//  recomputed. Because a face flux is shared, this stays conservative. The LLF
//  masks only ever grow, so the sweep loop is guaranteed to terminate; MAX_SWEEPS
//  bounds the cost. Any cell still non-positive after the limiter is left for the
//  runner's positivity scan to flag (we never floor silently).
//
//  For runs that never hit a positivity failure (e.g. None / GLM on Orszag-Tang)
//  no faces are ever flagged and the result is bit-identical to plain HLLD.
// -----------------------------------------------------------------------------

std::vector<State> positivity_limited_stage(
    const std::vector<State>& base,
    const std::vector<State>& flux_state,
    double coeff,
    double dt,
    int nx, int ny,
    double dx, double dy,
    double gamma,
    Reconstruction recon,
    SlopeLimiter limiter,
    std::vector<double>* mass_flux_x_out = nullptr,
    std::vector<double>* mass_flux_y_out = nullptr
) {
    const int ncell = nx * ny;
    constexpr int MAX_SWEEPS = 8;

    std::vector<char> llf_x(ncell, 0);
    std::vector<char> llf_y(ncell, 0);
    std::vector<State> out(ncell);

    for (int sweep = 0; ; ++sweep) {
        const std::vector<State> RHS =
            compute_rhs_blended_2d(
                flux_state, llf_x, llf_y, nx, ny, dx, dy, gamma,
                recon, limiter,
                mass_flux_x_out, mass_flux_y_out);

        for (int id = 0; id < ncell; ++id) {
            for (int k = 0; k < NVAR; ++k) {
                out[id][k] = base[id][k] + coeff * dt * RHS[id][k];
            }
        }

        if (sweep >= MAX_SWEEPS) {
            break;  // best effort; runner's positivity scan handles any remainder
        }

        int newly_marked = 0;
        for (int j = 0; j < ny; ++j) {
            const int jp = periodic_index(j + 1, ny);
            for (int i = 0; i < nx; ++i) {
                if (cell_is_physical(out[idx2d(i, j, nx)], gamma)) {
                    continue;
                }
                const int ip = periodic_index(i + 1, nx);
                char* faces[4] = {
                    &llf_x[idx2d(i,  j,  nx)],   // left
                    &llf_x[idx2d(ip, j,  nx)],   // right
                    &llf_y[idx2d(i,  j,  nx)],   // bottom
                    &llf_y[idx2d(i,  jp, nx)],   // top
                };
                for (char* f : faces) {
                    if (*f == 0) { *f = 1; ++newly_marked; }
                }
            }
        }

        if (newly_marked == 0) {
            break;  // offending cells already fully LLF; nothing more to do
        }
    }

    return out;
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
    Reconstruction recon,
    SlopeLimiter limiter,
    const GLM2DParams* stage_glm_params   // non-null enables stage projection
) {
    const int ncell = nx * ny;

    // Dual-energy (Option C): reconstruct the entropy from the positive
    // start-of-step state, then advect it alongside the conserved update.
    const std::vector<double> Sc = conserved_entropy_field(U, gamma);

    // Predictor: Us = U + dt * RHS(U), positivity-limited (Option A). Capture the
    // per-face mass fluxes so the entropy advects consistently with density.
    std::vector<double> mfx(ncell, 0.0), mfy(ncell, 0.0);
    std::vector<State> Us =
        positivity_limited_stage(U, U, /*coeff=*/1.0, dt,
                                 nx, ny, dx, dy, gamma, recon, limiter,
                                 &mfx, &mfy);

    const std::vector<double> S_rhs1 =
        entropy_rhs_from_mass_flux(U, Sc, mfx, mfy, nx, ny, dx, dy);
    std::vector<double> Sc_s(ncell);
    for (int id = 0; id < ncell; ++id) {
        Sc_s[id] = Sc[id] + dt * S_rhs1[id];
    }
    apply_dual_energy_recovery(Us, Sc_s, gamma);

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

    // Corrector: U = 0.5*(U + Us) + 0.5*dt*RHS(Us), positivity-limited (Option A).
    std::vector<State> base(ncell);
    for (int id = 0; id < ncell; ++id) {
        for (int k = 0; k < NVAR; ++k) {
            base[id][k] = 0.5 * (U[id][k] + Us[id][k]);
        }
    }
    std::vector<double> mfx2(ncell, 0.0), mfy2(ncell, 0.0);
    U = positivity_limited_stage(base, Us, /*coeff=*/0.5, dt,
                                 nx, ny, dx, dy, gamma, recon, limiter,
                                 &mfx2, &mfy2);

    // Dual-energy corrector: SSP-RK2 combination of the entropy, advected with
    // the corrector mass fluxes, then recover pressure in low-beta cells.
    const std::vector<double> S_rhs2 =
        entropy_rhs_from_mass_flux(Us, Sc_s, mfx2, mfy2, nx, ny, dx, dy);
    std::vector<double> Sc_new(ncell);
    for (int id = 0; id < ncell; ++id) {
        Sc_new[id] = 0.5 * (Sc[id] + Sc_s[id] + dt * S_rhs2[id]);
    }
    apply_dual_energy_recovery(U, Sc_new, gamma);
}

// Overload without stage projection (backward-compatible default).
void rk2_step_hlld_2d(
    std::vector<State>& U,
    int nx, int ny,
    double dx, double dy,
    double gamma, double dt,
    Reconstruction recon,
    SlopeLimiter limiter
) {
    rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt, recon, limiter, nullptr);
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
    double min_dt_used,
    double actual_ch,
    double actual_cp,
    double effective_cd,
    double effective_cr,
    const MHDRunTiming& timing
) {
    const std::string filename =
        runner_output_path(
            params,
            "summaries",
            params.glm.out_prefix + "_" + method + "_summary.csv"
        ).string();

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
        << "projection_theta,retry_count,min_dt_used,"
        << "glm_ch_factor,glm_cd,glm_cr,glm_subcycles,"
        << "glm_ch,glm_cp,glm_effective_cd,glm_effective_cr,"
        << "total_wall_time_sec,"
        << "initialization_time_sec,"
        << "hydro_time_sec,"
        << "cleaning_time_sec,"
        << "diagnostics_compute_time_sec,"
        << "diagnostics_write_time_sec,"
        << "snapshot_write_time_sec,"
        << "summary_write_time_sec,"
        << "output_time_sec,"
        << "steps,"
        << "total_cell_updates,"
        << "seconds_per_step,"
        << "cell_updates_per_second,"
        << "nx,ny,ncell,reconstruction,limiter,diagnostic_stride,"
        << "performance_mode\n";

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
         << final_diag.min_raw_pressure << ","
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
    const bool glm_damping_active = std::isfinite(actual_cp);

    out << ","
        << params.glm.glm_ch_factor << ",";
    write_optional_double(out, (glm_damping_active && std::isfinite(params.glm.glm_cd))
        ? params.glm.glm_cd
        : std::numeric_limits<double>::quiet_NaN());
    out << ",";
    write_optional_double(out, (glm_damping_active && std::isfinite(params.glm.glm_cr))
        ? params.glm.glm_cr
        : std::numeric_limits<double>::quiet_NaN());
    out << ","
        << params.glm.glm_subcycles << ",";
    write_optional_double(out, actual_ch);
    out << ",";
    write_optional_double(out, actual_cp);
    out << ",";
    write_optional_double(out, effective_cd);
    out << ",";
    write_optional_double(out, effective_cr);
    out << ","
        << timing.total_wall_time_sec << ","
        << timing.initialization_time_sec << ","
        << timing.hydro_time_sec << ","
        << timing.cleaning_time_sec << ","
        << timing.diagnostics_compute_time_sec << ","
        << timing.diagnostics_write_time_sec << ","
        << timing.snapshot_write_time_sec << ","
        << timing.summary_write_time_sec << ","
        << timing.output_time_sec << ","
        << timing.steps << ","
        << timing.total_cell_updates << ","
        << timing.seconds_per_step << ","
        << timing.cell_updates_per_second << ","
        << params.glm.nx << ","
        << params.glm.ny << ","
        << static_cast<long long>(params.glm.nx) * params.glm.ny << ","
        << reconstruction_name(params.reconstruction) << ","
        << limiter_name(params.limiter) << ","
        << std::max(1, params.diagnostic_stride) << ","
        << (params.performance_mode ? 1 : 0);
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

// -----------------------------------------------------------------------------
//  2D MHD blast wave (Balsara & Spicer 1999).
//
//  A circular over-pressured core (P_blast) sits in a uniform, strongly
//  magnetized ambient medium (P_ambient, B0 at 45 deg). The resulting fast
//  shock expands anisotropically -- fast along B, retarded across it -- so the
//  magnetic tension visibly deforms the blast. The strong-field, low-beta
//  ambient state and the steep pressure jump make this a demanding divB-cleaning
//  test (the GLM psi field must shed the divB generated at the shock).
//
//  Boundaries: the runner is periodic (compute_rhs_blended_2d). Choose t_end so
//  the shock does not reach the domain edge; t_end ~ 0.2 on the unit square with
//  the defaults below keeps it interior.
//
//  All setup constants are gathered here so the case is easy to retune.
// -----------------------------------------------------------------------------
void initialize_blast_wave_2d(
    std::vector<State>& U,
    const MHDRunParams& params
) {
    const int nx = params.glm.nx;
    const int ny = params.glm.ny;
    const double dx = params.glm.dx;
    const double dy = params.glm.dy;
    const double gamma = params.gamma;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error("initialize_blast_wave_2d: U size mismatch.");
    }

    // -- Tunable setup constants ---------------------------------------------
    // Blast centre, placed at the geometric centre of the domain so the test
    // works unchanged for [0, xlen] x [0, ylen] of any size.
    const double xc = 0.5 * params.glm.xlen;
    const double yc = 0.5 * params.glm.ylen;

    const double R_blast = 0.1;  // radius of the over-pressured core

    // Ambient (background) medium.
    const double rho0 = 1.0;
    const double u0   = 0.0;
    const double v0   = 0.0;
    const double w0   = 0.0;
    const double p_ambient = 0.1;

    // Strong uniform magnetic field at 45 deg in the x-y plane (|B0| = 1).
    const double B0 = 1.0;
    const double Bx0 = B0 / std::sqrt(2.0);
    const double By0 = B0 / std::sqrt(2.0);
    const double Bz0 = 0.0;

    // Over-pressured core (same rho, v, B as the ambient -- pressure only).
    const double p_blast = 10.0;

    const double psi0 = 0.0;  // GLM scalar starts divergence-free
    // ------------------------------------------------------------------------

    const double R2 = R_blast * R_blast;

    for (int j = 0; j < ny; ++j) {
        const double y = (j + 0.5) * dy;
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double rx = x - xc;
            const double ry = y - yc;
            const double r2 = rx * rx + ry * ry;

            const double p = (r2 < R2) ? p_blast : p_ambient;

            const PrimState W(
                rho0,
                u0,
                v0,
                w0,
                p,
                Bx0,
                By0,
                Bz0,
                psi0
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
    const auto run_start = Clock::now();
    const auto init_start = Clock::now();

    params.diagnostic_stride = std::max(1, params.diagnostic_stride);
    validate_glm_tuning_params(params.glm);

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
    } else if (params.problem == "field_loop") {
        initialize_field_loop_2d(U, params);
    } else if (params.problem == "divergence_advection") {
        initialize_divergence_advection_2d(U, params);
    } else if (params.problem == "blast_wave") {
        initialize_blast_wave_2d(U, params);
    } else {
        throw std::invalid_argument(
            "run_mhd_2d_case: unknown problem '" + params.problem + "'"
        );
    }

    // Freeze ch at the initial max signal speed (Dedner 2002 convention),
    // with optional paper-consistent scaling.
    const double ch_init = max_signal_speed_2d(U, gamma, 0.0, false);
    params.glm.ch = params.glm.glm_ch_factor * ch_init;

    const bool glm_active = is_glm_cleaning(type);

    const std::string name   = cleaning_name(type);
    const std::string prefix = params.glm.out_prefix;

    MHDRunTiming timing;
    timing.initialization_time_sec += seconds_since(init_start);

    fs::create_directories(fs::path(params.output_root) / "divergence");
    fs::create_directories(fs::path(params.output_root) / "snapshots");
    fs::create_directories(fs::path(params.output_root) / "summaries");
    if (projection_correction_diagnostics_active(params, type)) {
        initialize_projection_diagnostics_summary_csv(params, name);
    }

    const auto initial_diag_start = Clock::now();
    const MHDRunDiagnostics initial_diag =
        compute_mhd_run_diagnostics(U, gamma, dx * dy);
    timing.diagnostics_compute_time_sec += seconds_since(initial_diag_start);
    const double energy_initial = initial_diag.total_energy;

    // Initial snapshot (optional).
    if (params.glm.write_snapshot && params.glm.write_initial_snapshot) {
        const std::string snap =
            runner_output_path(
                params,
                "snapshots",
                prefix + "_" + name + "_initial.csv"
            ).string();
        const auto snapshot_start = Clock::now();
        write_mhd_2d_snapshot(U, params, snap);
        timing.snapshot_write_time_sec += seconds_since(snapshot_start);
        std::cout << "  Wrote " << snap << "\n";
    }

    // Diagnostics CSV.
    const std::string diag_name =
        runner_output_path(
            params,
            "divergence",
            prefix + "_" + name + ".csv"
        ).string();
    std::ofstream diag(diag_name);
    if (!diag) {
        throw std::runtime_error("Failed to open diagnostic file: " + diag_name);
    }
    {
        const auto diag_write_start = Clock::now();
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
        timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
    }

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
    double run_actual_ch = glm_active
        ? params.glm.ch
        : std::numeric_limits<double>::quiet_NaN();
    double run_actual_cp = std::numeric_limits<double>::quiet_NaN();
    double run_effective_cd = std::numeric_limits<double>::quiet_NaN();
    double run_effective_cr = std::numeric_limits<double>::quiet_NaN();
    double max_cfast_seen = ch_init;

    while (true) {
        const auto diag_compute_start = Clock::now();
        const LocalDivBNorms norms =
            compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
        const MHDRunDiagnostics run_diag =
            compute_mhd_run_diagnostics(U, gamma, dx * dy);
        timing.diagnostics_compute_time_sec += seconds_since(diag_compute_start);

        if (should_write_diagnostic_row(
                step,
                t,
                t_end,
                has_mhd_run_failure(run_diag),
                params.diagnostic_stride
            )) {
            const auto diag_write_start = Clock::now();
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
            timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
        }

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
            const MinPhysical mp = compute_min_physical(U, gamma);
            run_stage_mins.min_pressure_after_full_step = std::min(
                run_stage_mins.min_pressure_after_full_step,
                mp.min_pressure
            );
            run_stage_mins.min_density_after_full_step = std::min(
                run_stage_mins.min_density_after_full_step,
                mp.min_density
            );
            run_stage_mins.failure_stage = "step_start_diagnostic";
            BadStateRecord top_bad =
                scan_physical_state(
                    U,
                    nx,
                    ny,
                    dx,
                    dy,
                    gamma,
                    "step_start_diagnostic"
                );
            if (!top_bad.found) {
                top_bad = scan_raw_primitive_state(
                    U,
                    nx,
                    ny,
                    dx,
                    dy,
                    gamma,
                    "step_start_diagnostic"
                );
            }
            if (top_bad.found) {
                const auto diag_write_start = Clock::now();
                write_cleaning_failure_csv(
                    params,
                    name,
                    step,
                    failure_time,
                    top_bad,
                    run_stage_mins,
                    run_projection_theta,
                    run_total_retries,
                    run_min_dt_used
                );
                timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
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
        max_cfast_seen = std::max(max_cfast_seen, smax);
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
        MinPhysical accepted_after_hydro_min;
        bool have_accepted_after_hydro_min = false;

        while (true) {
            U = U_begin;
            params.glm.dt = dt;

            // ------------------------------------------------------------------
            // Step 1: HLLD finite-volume update.
            // Task C: pass stage GLM params when project_each_stage is enabled.
            // ------------------------------------------------------------------
            const auto hydro_start = Clock::now();
            if (do_stage_projection) {
                rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt,
                                 params.reconstruction, params.limiter,
                                 &params.glm);
            } else {
                rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt,
                                 params.reconstruction, params.limiter,
                                 nullptr);
            }
            timing.hydro_time_sec += seconds_since(hydro_start);

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

            accepted_after_hydro_min = compute_min_physical(U, gamma);
            have_accepted_after_hydro_min = true;

            // ------------------------------------------------------------------
            // Step 2: divergence-cleaning update (any CleaningType).
            // ------------------------------------------------------------------
            const auto cleaning_start = Clock::now();
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
                        !printed_parabolic_subcycling,
                    max_cfast_seen
                );
            timing.cleaning_time_sec += seconds_since(cleaning_start);

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
            if (have_accepted_after_hydro_min) {
                step_stage_mins.min_pressure_after_hydro =
                    accepted_after_hydro_min.min_pressure;
                step_stage_mins.min_density_after_hydro =
                    accepted_after_hydro_min.min_density;
            }
        }

        step_stage_mins.min_pressure_after_cleaning_B =
            cleaning_stats.stage_mins.min_pressure_after_cleaning_B;
        step_stage_mins.min_density_after_cleaning_B =
            cleaning_stats.stage_mins.min_density_after_cleaning_B;
        step_stage_mins.min_pressure_after_energy_repair =
            cleaning_stats.stage_mins.min_pressure_after_energy_repair;
        step_stage_mins.min_density_after_energy_repair =
            cleaning_stats.stage_mins.min_density_after_energy_repair;
        if (!cleaning_stats.stage_mins.failure_stage.empty()) {
            step_stage_mins.failure_stage =
                cleaning_stats.stage_mins.failure_stage;
        }

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

            {
                const auto diag_write_start = Clock::now();
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
                timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
            }

            const auto failed_diag_compute_start = Clock::now();
            const LocalDivBNorms failed_norms =
                compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
            const MHDRunDiagnostics failed_diag =
                compute_mhd_run_diagnostics(U, gamma, dx * dy);
            timing.diagnostics_compute_time_sec +=
                seconds_since(failed_diag_compute_start);

            const auto failed_diag_write_start = Clock::now();
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
            timing.diagnostics_write_time_sec +=
                seconds_since(failed_diag_write_start);

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
        if (std::isfinite(cleaning_stats.actual_ch)) {
            run_actual_ch = cleaning_stats.actual_ch;
        }
        if (std::isfinite(cleaning_stats.actual_cp)) {
            run_actual_cp = cleaning_stats.actual_cp;
        }
        if (std::isfinite(cleaning_stats.effective_cd)) {
            run_effective_cd = cleaning_stats.effective_cd;
        }
        if (std::isfinite(cleaning_stats.effective_cr)) {
            run_effective_cr = cleaning_stats.effective_cr;
        }

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

            if (step_stage_mins.failure_stage.empty()) {
                step_stage_mins.failure_stage =
                    determine_failure_stage(step_stage_mins);
            }
            run_stage_mins.failure_stage = step_stage_mins.failure_stage;

            {
                const auto diag_write_start = Clock::now();
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
                timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
            }

            t = failure_time;

            const auto failed_diag_compute_start = Clock::now();
            const LocalDivBNorms failed_norms =
                compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
            const MHDRunDiagnostics failed_diag =
                compute_mhd_run_diagnostics(U, gamma, dx * dy);
            timing.diagnostics_compute_time_sec +=
                seconds_since(failed_diag_compute_start);

            const auto failed_diag_write_start = Clock::now();
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
            timing.diagnostics_write_time_sec +=
                seconds_since(failed_diag_write_start);

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

    {
        const auto diag_write_start = Clock::now();
        diag.flush();
        timing.diagnostics_write_time_sec += seconds_since(diag_write_start);
    }

    // Final snapshot.
    if (params.glm.write_snapshot) {
        const std::string snap =
            runner_output_path(
                params,
                "snapshots",
                prefix + "_" + name + "_final.csv"
            ).string();
        const auto snapshot_start = Clock::now();
        write_mhd_2d_snapshot(U, params, snap);
        timing.snapshot_write_time_sec += seconds_since(snapshot_start);
        std::cout << "  Wrote " << snap << "\n";
    }

    const auto final_diag_compute_start = Clock::now();
    const LocalDivBNorms final_norms =
        compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
    const MHDRunDiagnostics final_diag =
        compute_mhd_run_diagnostics(U, gamma, dx * dy);
    timing.diagnostics_compute_time_sec += seconds_since(final_diag_compute_start);
    const bool final_time_reached =
        !stopped_for_failure && t >= t_end - 1.0e-12;

    timing.steps = step;
    timing.total_wall_time_sec = seconds_since(run_start);
    finalize_timing_fields(timing, nx, ny);

    auto write_summary_with_timing = [&](const MHDRunTiming& timing_for_file) {
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
            run_min_dt_used,
            run_actual_ch,
            run_actual_cp,
            run_effective_cd,
            run_effective_cr,
            timing_for_file
        );
    };

    // The summary row itself contains summary_write_time_sec.  Write once to
    // measure this small I/O cost, then overwrite with a self-consistent row that
    // includes one measured summary write in total/output time.  The second write
    // is intentionally not included, keeping the value stable and avoiding a
    // recursive timing dependency.
    MHDRunTiming timing_before_summary = timing;
    const auto summary_start = Clock::now();
    write_summary_with_timing(timing_before_summary);
    timing.summary_write_time_sec = seconds_since(summary_start);
    timing.total_wall_time_sec = seconds_since(run_start);
    finalize_timing_fields(timing, nx, ny);
    write_summary_with_timing(timing);

    std::cout << "  Wrote " << diag_name << "\n";
}
