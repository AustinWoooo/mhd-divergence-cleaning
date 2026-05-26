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
//  Adapters between State (std::array<double,9>) and std::vector<double>
// -----------------------------------------------------------------------------
//  HLLD_mhd_solver.hpp uses std::vector<double> for the conservative vector;
//  the GLM pipeline uses State. The two layouts are identical (indexed by Var),
//  so we just copy.

inline PrimState state_to_prim(const State& s, double gamma) {
    std::vector<double> v(s.begin(), s.end());
    return PrimState::from_conserved(v, gamma);
}

inline State prim_to_state(const PrimState& W, double gamma) {
    std::vector<double> v = W.to_conserved(gamma);
    State s;
    for (int k = 0; k < NVAR; ++k) s[k] = v[k];
    return s;
}

inline State vec_to_state(const std::vector<double>& v) {
    State s;
    for (int k = 0; k < NVAR; ++k) s[k] = v[k];
    return s;
}

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
    double projection_final_residual =
        std::numeric_limits<double>::quiet_NaN();
    bool projection_converged = true;
    bool projection_used = false;
    BadStateRecord bad_state;
    double failure_time = std::numeric_limits<double>::quiet_NaN();
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

void write_cleaning_failure_csv(
    const MHDRunParams& params,
    const std::string& method,
    int step,
    double time,
    const BadStateRecord& bad
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

    fout << "problem,method,stage,step,time,reason,i,j,"
         << "rho,pressure,total_energy,kinetic_energy,magnetic_energy,"
         << "internal_energy,Bx,By,Bz,psi,divB,energy_policy\n";

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
         << cleaning_energy_policy_name(params.glm.energy_policy) << "\n";
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

ProjectionResult apply_cleaning_update_for_runner(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
) {
    if (type == CleaningType::ELLIPTIC_PROJECTION) {
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
            apply_cleaning_update_for_runner(U, type, params);

        if (projection.iterations > 0 || type == CleaningType::ELLIPTIC_PROJECTION) {
            stats.projection_used = true;
            stats.projection_iterations += projection.iterations;
            stats.projection_final_residual = projection.final_residual;
            stats.projection_converged =
                stats.projection_converged && projection.converged;
        }

        if (repair_energy) {
            preserve_thermal_pressure_after_cleaning(U, Uold);
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
            write_cleaning_failure_csv(
                run_params,
                method_name,
                step + 1,
                stats.failure_time,
                bad
            );
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
            const std::vector<double> F = compute_flux(WL, WR, /*direction=*/0, gamma);
            flux_x[idx2d(i, j, nx)] = vec_to_state(F);
        }
    }

    // Y-direction Riemann problems.
    for (int j = 0; j < ny; ++j) {
        const int jL = periodic_index(j - 1, ny);
        for (int i = 0; i < nx; ++i) {
            const PrimState WL = state_to_prim(U[idx2d(i, jL, nx)], gamma);
            const PrimState WR = state_to_prim(U[idx2d(i, j,  nx)], gamma);
            const std::vector<double> F = compute_flux(WL, WR, /*direction=*/1, gamma);
            flux_y[idx2d(i, j, nx)] = vec_to_state(F);
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
// -----------------------------------------------------------------------------

void rk2_step_hlld_2d(
    std::vector<State>& U,
    int nx, int ny,
    double dx, double dy,
    double gamma, double dt
) {
    const int ncell = nx * ny;

    const std::vector<State> R1 = compute_rhs_hlld_2d(U, nx, ny, dx, dy, gamma);

    std::vector<State> Us(ncell);
    for (int id = 0; id < ncell; ++id) {
        for (int k = 0; k < NVAR; ++k) {
            Us[id][k] = U[id][k] + dt * R1[id][k];
        }
    }

    const std::vector<State> R2 = compute_rhs_hlld_2d(Us, nx, ny, dx, dy, gamma);

    for (int id = 0; id < ncell; ++id) {
        for (int k = 0; k < NVAR; ++k) {
            U[id][k] = 0.5 * (U[id][k] + Us[id][k] + dt * R2[id][k]);
        }
    }
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
    double projection_final_residual,
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
         << projection_iterations_step << ","
         << projection_final_residual << ","
         << (projection_converged ? 1 : 0)
         << "\n";
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
    long long projection_iterations_total
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

    out << "problem,method,final_time_reached,failure_time,failure_reason,"
        << "final_L1_fv,final_L2_fv,final_Linf_fv,"
        << "energy_initial,energy_final,energy_drift,"
        << "min_pressure,min_density,"
        << "cleaning_subcycles_total,projection_iterations_total\n";

    out << params.problem << ","
        << method << ","
        << (final_time_reached ? 1 : 0) << ","
        << failure_time << ","
        << failure_reason << ","
        << final_norms.L1 << ","
        << final_norms.L2 << ","
        << final_norms.Linf << ","
        << energy_initial << ","
        << energy_final << ","
        << energy_drift << ","
        << final_diag.min_pressure << ","
        << final_diag.min_density << ","
        << cleaning_subcycles_total << ","
        << projection_iterations_total << "\n";
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
         << "projection_final_residual,"
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
    bool last_projection_converged = true;

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
            last_projection_final_residual,
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

        // Step 1: HLLD finite-volume update.
        rk2_step_hlld_2d(U, nx, ny, dx, dy, gamma, dt);

        const double next_t = t + dt;
        const int next_step = step + 1;

        const BadStateRecord after_hydro_bad =
            scan_physical_state(
                U,
                nx,
                ny,
                dx,
                dy,
                gamma,
                "after_hydro"
            );

        if (after_hydro_bad.found) {
            stopped_for_failure = true;
            failure_time = next_t;
            failure_reason = "hydro_positivity_failure:" + after_hydro_bad.reason;
            write_cleaning_failure_csv(
                params,
                name,
                next_step,
                failure_time,
                after_hydro_bad
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
                      << "\n";

            t = failure_time;
            step = next_step;
            break;
        }

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
            write_cleaning_failure_csv(
                params,
                name,
                next_step,
                failure_time,
                before_cleaning_bad
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

        // Step 2: divergence-cleaning update (any CleaningType).
        const CleaningAdvanceStats cleaning_stats =
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

        if (type == CleaningType::PARABOLIC && cleaning_stats.subcycles > 1) {
            printed_parabolic_subcycling = true;
        }

        cleaning_subcycles_total += cleaning_stats.subcycles;
        projection_iterations_total += cleaning_stats.projection_iterations;
        last_cleaning_subcycles_step = cleaning_stats.subcycles;
        last_projection_iterations_step = cleaning_stats.projection_iterations;
        last_projection_final_residual = cleaning_stats.projection_final_residual;
        last_projection_converged = cleaning_stats.projection_converged;

        t += dt;
        ++step;
        params.glm.dt = dt;

        if (cleaning_stats.bad_state.found) {
            stopped_for_failure = true;
            failure_time = cleaning_stats.failure_time;
            failure_reason =
                "cleaning_induced_failure:" + cleaning_stats.bad_state.reason;
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
                cleaning_stats.projection_final_residual,
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
        projection_iterations_total
    );

    std::cout << "  Wrote " << diag_name << "\n";
}
