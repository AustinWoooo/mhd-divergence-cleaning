#include "glm2d.hpp"
#include "diagnostics.hpp"
#include "glm.hpp"
#include "glm2d_common.hpp"
#include "hyperbolic_glm2d.hpp"
#include "mixed_glm2d.hpp"
#include "parabolic2d.hpp"
#include "powell2d.hpp"
#include "projection2d.hpp"
#include "eglm2d.hpp"
#include "galilean_glm2d.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void validate_glm_2d_params(const GLM2DParams& params) {
    if (params.nx < 4 || params.ny < 4) {
        throw std::invalid_argument("GLM2DParams: nx and ny must both be >= 4.");
    }

    if (params.xlen <= 0.0 || params.ylen <= 0.0) {
        throw std::invalid_argument("GLM2DParams: xlen and ylen must be positive.");
    }

    if (params.t_end < 0.0) {
        throw std::invalid_argument("GLM2DParams: t_end must be non-negative.");
    }

    if (params.cfl <= 0.0) {
        throw std::invalid_argument("GLM2DParams: cfl must be positive.");
    }

    if (params.ch <= 0.0) {
        throw std::invalid_argument("GLM2DParams: ch must be positive.");
    }

    if (params.cp <= 0.0) {
        throw std::invalid_argument("GLM2DParams: cp must be positive.");
    }

    if (params.glm_ch_factor <= 0.0) {
        throw std::invalid_argument("GLM2DParams: glm_ch_factor must be positive.");
    }

    if (std::isfinite(params.glm_cd) &&
        !(params.glm_cd > 0.0 && params.glm_cd < 1.0)) {
        throw std::invalid_argument("GLM2DParams: glm_cd must satisfy 0 < glm_cd < 1.");
    }

    if (std::isfinite(params.glm_cr) && !(params.glm_cr > 0.0)) {
        throw std::invalid_argument("GLM2DParams: glm_cr must be positive.");
    }

    if (std::isfinite(params.glm_cd) && std::isfinite(params.glm_cr)) {
        throw std::invalid_argument("GLM2DParams: set only one of glm_cd or glm_cr.");
    }

    if (params.glm_subcycles < 1) {
        throw std::invalid_argument("GLM2DParams: glm_subcycles must be >= 1.");
    }

    if (params.poisson_max_iter <= 0) {
        throw std::invalid_argument("GLM2DParams: poisson_max_iter must be positive.");
    }

    if (params.poisson_tol <= 0.0) {
        throw std::invalid_argument("GLM2DParams: poisson_tol must be positive.");
    }

    if (!(params.poisson_omega > 0.0 && params.poisson_omega < 2.0)) {
        throw std::invalid_argument("GLM2DParams: poisson_omega must satisfy 0 < omega < 2.");
    }
}

double compute_glm_2d_timestep(
    CleaningType type,
    const GLM2DParams& params
) {
    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        // Projection is an instantaneous correction in this standalone test.
        return params.t_end > 0.0 ? params.t_end : 1.0;
    }

    const double dt_clean =
        max_cleaning_dt(type, params.dx, params.dy, params);

    if (std::isfinite(dt_clean)) {
        return dt_clean;
    }

    const double inv_dx = 1.0 / params.dx;
    const double inv_dy = 1.0 / params.dy;
    return params.cfl / (params.ch * (inv_dx + inv_dy));
}

bool all_state_values_finite(const std::vector<State>& U) {
    for (const State& cell : U) {
        for (double value : cell) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

CleaningType parse_cleaning_type_2d(const std::string& name) {
    if (name == "none") return CleaningType::NONE;
    if (name == "parabolic") return CleaningType::PARABOLIC;
    if (name == "hyperbolic_glm") return CleaningType::HYPERBOLIC_GLM;
    if (name == "mixed_glm") return CleaningType::MIXED_GLM;
    if (name == "elliptic_projection") return CleaningType::ELLIPTIC_PROJECTION;
    if (name == "powell_source") return CleaningType::POWELL_SOURCE;
    if (name == "eglm") return CleaningType::MIXED_EGLM;
    if (name == "mixed_eglm") return CleaningType::MIXED_EGLM;
    if (name == "gi_mixed_eglm") return CleaningType::GI_MIXED_EGLM;

    throw std::invalid_argument("Unknown cleaning type: " + name);
}

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
) {
    if (case_name == "all") {
        return {
            CleaningType::NONE,
            CleaningType::HYPERBOLIC_GLM,
            CleaningType::MIXED_GLM,
            CleaningType::PARABOLIC,
            CleaningType::ELLIPTIC_PROJECTION,
            CleaningType::POWELL_SOURCE,
            CleaningType::MIXED_EGLM,
            CleaningType::GI_MIXED_EGLM
        };
    }

    return {parse_cleaning_type_2d(case_name)};
}

double max_cleaning_dt(
    CleaningType method,
    double dx,
    double dy,
    const GLM2DParams& params
) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    if (method == CleaningType::PARABOLIC) {
        constexpr double Cdiff = 0.20;
        const double h = std::min(dx, dy);
        return Cdiff * h * h / (params.cp * params.cp);
    }

    if (method == CleaningType::HYPERBOLIC_GLM ||
        method == CleaningType::MIXED_GLM ||
        method == CleaningType::MIXED_EGLM ||
        method == CleaningType::GI_MIXED_EGLM) {
        return params.cfl / (params.ch * (inv_dx + inv_dy));
    }

    return std::numeric_limits<double>::infinity();
}

void initialize_divergence_pulse_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;

    if (static_cast<int>(U.size()) != nx * ny) {
        throw std::runtime_error(
            "initialize_divergence_pulse_2d: U has incorrect size."
        );
    }

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double y = (j + 0.5) * dy;

            State cell{};
            cell.fill(0.0);

            cell[RHO] = 1.0;

            // This sandbox only evolves B and psi. E is initialized to a
            // harmless positive value so that snapshots remain well-defined
            // if other diagnostics inspect the state vector.
            cell[E] = 1.0;

            const double dx0 = x - 0.5;
            const double dy0 = y - 0.5;
            const double r2 = dx0 * dx0 + dy0 * dy0;
            const double bump = std::exp(-100.0 * r2);

            // Intentionally non-solenoidal magnetic field:
            //
            //   Bx = 1 + 0.10 exp(-100 r^2)
            //   By =     0.05 exp(-100 r^2)
            //
            // Analytic initial divergence:
            //
            //   divB = [-20(x-0.5) - 10(y-0.5)] exp(-100 r^2).
            cell[BX] = 1.0 + 0.10 * bump;
            cell[BY] = 0.05 * bump;
            cell[BZ] = 0.0;

            cell[PSI] = 0.0;

            U[idx2d(i, j, nx)] = cell;
        }
    }
}

void advance_glm_2d_one_step(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
) {
    if (type == CleaningType::NONE) {
        return;
    }

    if (type == CleaningType::HYPERBOLIC_GLM) {
        update_hyperbolic_glm_2d(U, params);
        return;
    }

    if (type == CleaningType::MIXED_GLM) {
        update_mixed_glm_2d(U, params);
        return;
    }

    if (type == CleaningType::PARABOLIC) {
        apply_parabolic_cleaning_2d(U, params);
        return;
    }

    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        apply_elliptic_projection_2d(U, params);
        return;
    }

    if (type == CleaningType::POWELL_SOURCE) {
        apply_powell_source_2d(U, params);
        return;
    }

    if (type == CleaningType::MIXED_EGLM) {
        update_mixed_eglm_2d(U, params);
        return;
    }

    if (type == CleaningType::GI_MIXED_EGLM) {
        update_gi_mixed_eglm_2d(U, params);
        return;
    }

    throw std::runtime_error("Unsupported cleaning type in 2D GLM advance.");
}

void write_glm_2d_snapshot(
    const std::vector<State>& U,
    const GLM2DParams& params,
    const std::string& filename
) {
    ensure_parent_directory(filename);

    std::ofstream fout(filename);

    if (!fout) {
        throw std::runtime_error("Failed to open snapshot file: " + filename);
    }

    fout << "i,j,x,y,"
         << "Bx,By,Bz,psi,"
         << "divB_centered,normDivB_centered,"
         << "divB_fv,normDivB_fv,"
         << "divB_analytic_initial,"
         << "Bmag\n";

    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const double x = (i + 0.5) * params.dx;
            const double y = (j + 0.5) * params.dy;

            const State& cell = U[idx2d(i, j, params.nx)];

            const double divB_centered =
                compute_divB_cell_2d(
                    U,
                    params.nx,
                    params.ny,
                    i,
                    j,
                    params.dx,
                    params.dy
                );

            const double normDivB_centered =
                compute_normalized_divB_cell_2d(
                    U,
                    params.nx,
                    params.ny,
                    i,
                    j,
                    params.dx,
                    params.dy
                );

            const double divB_fv =
                compute_fv_divB_cell_2d(
                    U,
                    params.nx,
                    params.ny,
                    i,
                    j,
                    params.dx,
                    params.dy
                );

            const double normDivB_fv =
                compute_fv_normalized_divB_cell_2d(
                    U,
                    params.nx,
                    params.ny,
                    i,
                    j,
                    params.dx,
                    params.dy
                );

            const double divB_analytic =
                analytic_divergence_pulse_2d(x, y);

            const double Bmag = std::sqrt(
                cell[BX] * cell[BX]
              + cell[BY] * cell[BY]
              + cell[BZ] * cell[BZ]
            );

            fout << i << ","
                 << j << ","
                 << x << ","
                 << y << ","
                 << cell[BX] << ","
                 << cell[BY] << ","
                 << cell[BZ] << ","
                 << cell[PSI] << ","
                 << divB_centered << ","
                 << normDivB_centered << ","
                 << divB_fv << ","
                 << normDivB_fv << ","
                 << divB_analytic << ","
                 << Bmag << "\n";
        }
    }
}

void run_glm_2d_case(
    CleaningType type,
    GLM2DParams params
) {
    validate_glm_2d_params(params);

    params.dx = params.xlen / static_cast<double>(params.nx);
    params.dy = params.ylen / static_cast<double>(params.ny);

    params.dt = compute_glm_2d_timestep(type, params);

    int nstep = 1;

    if (params.t_end > 0.0) {
        nstep = std::max(
            1,
            static_cast<int>(std::ceil(params.t_end / params.dt))
        );
    }

    // Elliptic projection is instantaneous in this standalone sandbox.
    // We keep two diagnostic samples: before and after projection.
    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        nstep = 1;
    }

    params.dt = params.t_end / static_cast<double>(nstep);

    std::vector<State> U(params.nx * params.ny);
    initialize_divergence_pulse_2d(U, params);

    const std::string name = cleaning_name(type);

    fs::create_directories("results/glm_2d/divergence");
    fs::create_directories("results/glm_2d/snapshots");

    if (params.write_snapshot && params.write_initial_snapshot) {
        const std::string initial_snap_name =
            "results/glm_2d/snapshots/"
          + params.out_prefix
          + "_"
          + name
          + "_initial.csv";

        write_glm_2d_snapshot(U, params, initial_snap_name);

        std::cout << "Wrote " << initial_snap_name << "\n";
    }

    const std::string diag_name =
        "results/glm_2d/divergence/"
      + params.out_prefix
      + "_"
      + name
      + ".csv";

    std::ofstream diag(diag_name);

    if (!diag) {
        throw std::runtime_error("Failed to open diagnostic file: " + diag_name);
    }

    diag << "step,time,dt,"
         << "L1_centered,L2_centered,Linf_centered,"
         << "L1_norm_centered,L2_norm_centered,Linf_norm_centered,"
         << "L1_fv,L2_fv,Linf_fv,"
         << "L1_norm_fv,L2_norm_fv,Linf_norm_fv\n";

    for (int step = 0; step <= nstep; ++step) {
        const double time = step * params.dt;

        const DivBNorms centered =
            compute_divB_norms_2d(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy
            );

        const LocalDivBNorms fv =
            compute_fv_divB_norms_2d(
                U,
                params.nx,
                params.ny,
                params.dx,
                params.dy
            );

        diag << step << ","
             << time << ","
             << params.dt << ","
             << centered.L1 << ","
             << centered.L2 << ","
             << centered.Linf << ","
             << centered.L1_norm << ","
             << centered.L2_norm << ","
             << centered.Linf_norm << ","
             << fv.L1 << ","
             << fv.L2 << ","
             << fv.Linf << ","
             << fv.L1_norm << ","
             << fv.L2_norm << ","
             << fv.Linf_norm << "\n";

        if (step == nstep) {
            break;
        }

        advance_glm_2d_one_step(U, type, params);

        if (!all_state_values_finite(U)) {
            throw std::runtime_error(
                "run_glm_2d_case: nonfinite state after "
              + name
              + " update at step "
              + std::to_string(step + 1)
            );
        }
    }

    if (params.write_snapshot) {
        const std::string final_snap_name =
            "results/glm_2d/snapshots/"
          + params.out_prefix
          + "_"
          + name
          + "_final.csv";

        write_glm_2d_snapshot(U, params, final_snap_name);

        std::cout << "Wrote " << final_snap_name << "\n";
    }

    std::cout << "Wrote " << diag_name << "\n";
}
