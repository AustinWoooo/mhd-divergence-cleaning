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
         << "has_nonfinite,has_negative_density,has_negative_pressure\n";

    double t    = 0.0;
    int    step = 0;
    bool stopped_for_failure = false;

    while (true) {
        const LocalDivBNorms norms =
            compute_fv_divB_norms_2d(U, nx, ny, dx, dy);
        const MHDRunDiagnostics run_diag =
            compute_mhd_run_diagnostics(U, gamma, dx * dy);

        diag << step << "," << t << "," << params.glm.dt << ","
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
             << (run_diag.has_negative_pressure ? 1 : 0)
             << "\n";

        if (has_mhd_run_failure(run_diag)) {
            stopped_for_failure = true;
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

        // Step 2: divergence-cleaning update (any CleaningType).
        advance_glm_2d_one_step(U, type, params.glm);

        t += dt;
        ++step;

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

    std::cout << "  Wrote " << diag_name << "\n";
}
