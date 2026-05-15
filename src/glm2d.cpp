#include "glm2d.hpp"
#include "glm2d_common.hpp"
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



// Finite-volume-compatible divergence.
//
// This is the divergence operator used by the elliptic projection and parabolic
// cleaning in this file. It is intentionally different from a centered derivative:
//
//   div_fv(B)_ij =
//       (Bx_ij - Bx_{i-1,j}) / dx
//     + (By_ij - By_{i,j-1}) / dy
//
// With the forward gradient
//
//   grad_x(phi)_ij = (phi_{i+1,j} - phi_ij) / dx,
//   grad_y(phi)_ij = (phi_{i,j+1} - phi_ij) / dy,
//
// we get exactly
//
//   div_fv(grad_fv phi) = Laplacian_5pt(phi).








void subtract_mean(std::vector<double>& a) {
    if (a.empty()) return;

    double mean = 0.0;
    for (double x : a) {
        mean += x;
    }

    mean /= static_cast<double>(a.size());

    for (double& x : a) {
        x -= mean;
    }
}

double compute_glm_2d_timestep(
    CleaningType type,
    const GLM2DParams& params
) {
    const double inv_dx = 1.0 / params.dx;
    const double inv_dy = 1.0 / params.dy;

    if (type == CleaningType::PARABOLIC) {
        // Explicit diffusion-like stability condition in 2D:
        //
        //   dt <= C / [ cp^2 (1/dx^2 + 1/dy^2) ].
        //
        // C <= 0.5 is the formal FTCS limit in 2D. We use min(cfl, 0.45)
        // to prevent accidental unstable settings.
        const double safe_cfl = std::min(params.cfl, 0.45);
        return safe_cfl / (
            params.cp * params.cp * (inv_dx * inv_dx + inv_dy * inv_dy)
        );
    }

    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        // Projection is an instantaneous correction in this standalone test.
        return params.t_end > 0.0 ? params.t_end : 1.0;
    }

    // Unsplit 2D hyperbolic CFL estimate:
    //
    //   dt <= C / [ ch (1/dx + 1/dy) ].
    //
    // This is safer than dt = C * min(dx,dy) / ch for a 2D unsplit update.
    return params.cfl / (params.ch * (inv_dx + inv_dy));
}

std::vector<double> solve_periodic_poisson_sor_5pt(
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

    // Periodic Poisson equation requires zero-mean RHS.
    subtract_mean(rhs);

    double max_update = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < params.poisson_max_iter; ++iter) {
        max_update = 0.0;

        for (int j = 0; j < ny; ++j) {
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            for (int i = 0; i < nx; ++i) {
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

                max_update = std::max(
                    max_update,
                    std::abs(new_value - old_value)
                );
            }
        }

        // Remove the arbitrary constant mode from time to time.
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

CleaningType parse_cleaning_type_2d(const std::string& name) {
    if (name == "none") return CleaningType::NONE;
    if (name == "parabolic") return CleaningType::PARABOLIC;
    if (name == "hyperbolic_glm") return CleaningType::HYPERBOLIC_GLM;
    if (name == "mixed_glm") return CleaningType::MIXED_GLM;
    if (name == "elliptic_projection") return CleaningType::ELLIPTIC_PROJECTION;
    if (name == "powell_source") return CleaningType::POWELL_SOURCE;
    if (name == "mixed_eglm") return CleaningType::MIXED_EGLM;

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
            CleaningType::MIXED_EGLM
        };
    }

    return {parse_cleaning_type_2d(case_name)};
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

void update_hyperbolic_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;
    const double ch = params.ch;

    std::vector<GLMFlux> flux_x((nx + 1) * ny);
    std::vector<GLMFlux> flux_y(nx * (ny + 1));

    auto fx_id = [nx](int iface, int j) {
        return j * (nx + 1) + iface;
    };

    auto fy_id = [nx](int i, int jface) {
        return jface * nx + i;
    };

    // x-direction GLM subsystem:
    //
    //   d/dt [Bx, psi]^T + d/dx [psi, ch^2 Bx]^T = 0.
    //
    // The characteristic speeds are +/- ch. compute_hyperbolic_glm_flux()
    // should return the Rusanov flux for this 2-variable subsystem.
    for (int j = 0; j < ny; ++j) {
        for (int iface = 0; iface <= nx; ++iface) {
            const int iL = periodic_index(iface - 1, nx);
            const int iR = periodic_index(iface,     nx);

            const State& UL = U[idx2d(iL, j, nx)];
            const State& UR = U[idx2d(iR, j, nx)];

            flux_x[fx_id(iface, j)] =
                compute_hyperbolic_glm_flux(
                    UL[BX], UL[PSI],
                    UR[BX], UR[PSI],
                    ch
                );
        }
    }

    // y-direction GLM subsystem:
    //
    //   d/dt [By, psi]^T + d/dy [psi, ch^2 By]^T = 0.
    for (int jface = 0; jface <= ny; ++jface) {
        const int jL = periodic_index(jface - 1, ny);
        const int jR = periodic_index(jface,     ny);

        for (int i = 0; i < nx; ++i) {
            const State& UL = U[idx2d(i, jL, nx)];
            const State& UR = U[idx2d(i, jR, nx)];

            flux_y[fy_id(i, jface)] =
                compute_hyperbolic_glm_flux(
                    UL[BY], UL[PSI],
                    UR[BY], UR[PSI],
                    ch
                );
        }
    }

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const GLMFlux& fxL = flux_x[fx_id(i,     j)];
            const GLMFlux& fxR = flux_x[fx_id(i + 1, j)];

            const GLMFlux& fyL = flux_y[fy_id(i, j)];
            const GLMFlux& fyR = flux_y[fy_id(i, j + 1)];

            State cell = Uold[idx2d(i, j, nx)];

            cell[BX] =
                Uold[idx2d(i, j, nx)][BX]
              - dt / dx * (fxR.FBn - fxL.FBn);

            cell[BY] =
                Uold[idx2d(i, j, nx)][BY]
              - dt / dy * (fyR.FBn - fyL.FBn);

            cell[PSI] =
                Uold[idx2d(i, j, nx)][PSI]
              - dt / dx * (fxR.Fpsi - fxL.Fpsi)
              - dt / dy * (fyR.Fpsi - fyL.Fpsi);

            U[idx2d(i, j, nx)] = cell;
        }
    }
}

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const double damping_rate =
        params.ch * params.ch / (params.cp * params.cp);

    const double factor = std::exp(-params.dt * damping_rate);

    for (auto& cell : U) {
        cell[PSI] *= factor;
    }
}

void apply_parabolic_cleaning_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;
    const double cp2 = params.cp * params.cp;

    // Use finite-volume-compatible divB and forward gradient so that
    // divB evolves approximately by:
    //
    //   d(divB)/dt = cp^2 Laplacian(divB).
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);

            const int id = idx2d(i, j, nx);

            const double dDiv_dx =
                (divB[idx2d(ip, j,  nx)] - divB[id]) / dx;

            const double dDiv_dy =
                (divB[idx2d(i,  jp, nx)] - divB[id]) / dy;

            State cell = Uold[id];

            // Parabolic cleaning:
            //
            //   dB/dt = cp^2 grad(divB).
            //
            // Then divB satisfies a diffusion equation.
            cell[BX] += dt * cp2 * dDiv_dx;
            cell[BY] += dt * cp2 * dDiv_dy;

            U[id] = cell;
        }
    }
}

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;

    // Compatible RHS:
    //
    //   Laplacian_5pt(phi) = div_fv(B).
    std::vector<double> rhs =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    // Periodic Poisson equation is solvable only for zero-mean RHS.
    subtract_mean(rhs);

    const std::vector<double> phi =
        solve_periodic_poisson_sor_5pt(rhs, params);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = periodic_index(j + 1, ny);

        for (int i = 0; i < nx; ++i) {
            const int ip = periodic_index(i + 1, nx);

            const int id = idx2d(i, j, nx);

            const double dphi_dx =
                (phi[idx2d(ip, j, nx)] - phi[id]) / dx;

            const double dphi_dy =
                (phi[idx2d(i, jp, nx)] - phi[id]) / dy;

            State cell = Uold[id];

            // Compatible projection:
            //
            //   B <- B - grad_fv(phi).
            //
            // With the div_fv operator above:
            //
            //   div_fv(B_new) = div_fv(B_old) - Laplacian_5pt(phi).
            cell[BX] -= dphi_dx;
            cell[BY] -= dphi_dy;

            cell[PSI] = 0.0;

            U[id] = cell;
        }
    }
}

void apply_powell_source_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const int nx = params.nx;
    const int ny = params.ny;

    const double dx = params.dx;
    const double dy = params.dy;
    const double dt = params.dt;

    // Standalone induction-only Powell toy model:
    //
    //   dB/dt = -u divB.
    //
    // This only demonstrates advection of divergence errors. It is NOT the full
    // Powell 8-wave MHD system. The full system also modifies momentum and energy:
    //
    //   S_Powell = -(divB) [0, Bx, By, Bz, u.B, ux, uy, uz]^T
    //
    // for the full conservative MHD state.
    const std::vector<double> divB =
        compute_fv_divB_field_2d(U, nx, ny, dx, dy);

    const std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);

            State cell = Uold[id];

            cell[BX] -= dt * params.powell_vx * divB[id];
            cell[BY] -= dt * params.powell_vy * divB[id];

            U[id] = cell;
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
        update_hyperbolic_glm_2d(U, params);
        apply_mixed_glm_damping_2d(U, params);
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
        // In this standalone B-psi sandbox, EGLM-specific momentum and energy
        // source terms are not active. Therefore this case reduces to the same
        // B-psi subsystem as mixed GLM.
        update_hyperbolic_glm_2d(U, params);
        apply_mixed_glm_damping_2d(U, params);
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

    fs::create_directories("results/divergence");
    fs::create_directories("results/snapshots");

    if (params.write_snapshot && params.write_initial_snapshot) {
        const std::string initial_snap_name =
            "results/snapshots/"
          + params.out_prefix
          + "_"
          + name
          + "_initial.csv";

        write_glm_2d_snapshot(U, params, initial_snap_name);

        std::cout << "Wrote " << initial_snap_name << "\n";
    }

    const std::string diag_name =
        "results/divergence/"
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
    }

    if (params.write_snapshot) {
        const std::string final_snap_name =
            "results/snapshots/"
          + params.out_prefix
          + "_"
          + name
          + "_final.csv";

        write_glm_2d_snapshot(U, params, final_snap_name);

        std::cout << "Wrote " << final_snap_name << "\n";
    }

    std::cout << "Wrote " << diag_name << "\n";
}