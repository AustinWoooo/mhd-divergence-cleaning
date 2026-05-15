#include "glm2d.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

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

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double y = (j + 0.5) * dy;

            State cell{};
            cell.fill(0.0);

            cell[RHO] = 1.0;
            cell[E]   = 1.0;

            const double r2 =
                (x - 0.5) * (x - 0.5)
              + (y - 0.5) * (y - 0.5);

            const double bump = std::exp(-100.0 * r2);

            // Intentionally non-solenoidal magnetic field.
            // This creates divB != 0, so cleaning can be tested.
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

    // x-direction GLM flux.
    // Normal magnetic component is Bx.
    for (int j = 0; j < ny; ++j) {
        for (int iface = 0; iface <= nx; ++iface) {
            const int iL = (iface - 1 + nx) % nx;
            const int iR = iface % nx;

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

    // y-direction GLM flux.
    // Normal magnetic component is By.
    for (int jface = 0; jface <= ny; ++jface) {
        const int jL = (jface - 1 + ny) % ny;
        const int jR = jface % ny;

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

    std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const GLMFlux& fxL = flux_x[fx_id(i, j)];
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
    const double factor =
        std::exp(-params.dt * params.ch * params.ch / (params.cp * params.cp));

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
    const double cp = params.cp;

    std::vector<double> divB(nx * ny, 0.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            divB[idx2d(i, j, nx)] =
                compute_divB_cell_2d(U, nx, ny, i, j, dx, dy);
        }
    }

    std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;

        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;

            const double dDiv_dx =
                (divB[idx2d(ip, j, nx)] - divB[idx2d(im, j, nx)]) / (2.0 * dx);

            const double dDiv_dy =
                (divB[idx2d(i, jp, nx)] - divB[idx2d(i, jm, nx)]) / (2.0 * dy);

            State cell = Uold[idx2d(i, j, nx)];

            // Parabolic cleaning:
            // dB/dt = cp^2 grad(divB)
            cell[BX] += dt * cp * cp * dDiv_dx;
            cell[BY] += dt * cp * cp * dDiv_dy;

            U[idx2d(i, j, nx)] = cell;
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

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (inv_dx2 + inv_dy2);

    std::vector<double> rhs(nx * ny, 0.0);
    std::vector<double> phi(nx * ny, 0.0);
    std::vector<double> phi_new(nx * ny, 0.0);

    double mean_rhs = 0.0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double divB =
                compute_divB_cell_2d(U, nx, ny, i, j, dx, dy);

            rhs[idx2d(i, j, nx)] = divB;
            mean_rhs += divB;
        }
    }

    mean_rhs /= static_cast<double>(nx * ny);

    // Periodic Poisson equation requires zero-mean RHS.
    for (double& val : rhs) {
        val -= mean_rhs;
    }

    double residual = 1.0e99;

    for (int iter = 0; iter < params.poisson_max_iter; ++iter) {
        residual = 0.0;

        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny;
            const int jm = (j - 1 + ny) % ny;

            for (int i = 0; i < nx; ++i) {
                const int ip = (i + 1) % nx;
                const int im = (i - 1 + nx) % nx;

                const double candidate =
                    (
                        (phi[idx2d(ip, j, nx)] + phi[idx2d(im, j, nx)]) * inv_dx2
                      + (phi[idx2d(i, jp, nx)] + phi[idx2d(i, jm, nx)]) * inv_dy2
                      - rhs[idx2d(i, j, nx)]
                    ) / denom;

                phi_new[idx2d(i, j, nx)] = candidate;

                residual = std::max(
                    residual,
                    std::abs(candidate - phi[idx2d(i, j, nx)])
                );
            }
        }

        phi.swap(phi_new);

        if (residual < params.poisson_tol) {
            break;
        }
    }

    std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;

        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;

            const double dphi_dx =
                (phi[idx2d(ip, j, nx)] - phi[idx2d(im, j, nx)]) / (2.0 * dx);

            const double dphi_dy =
                (phi[idx2d(i, jp, nx)] - phi[idx2d(i, jm, nx)]) / (2.0 * dy);

            State cell = Uold[idx2d(i, j, nx)];

            // Projection:
            // B <- B - grad(phi), where laplacian(phi) = divB.
            cell[BX] -= dphi_dx;
            cell[BY] -= dphi_dy;
            cell[PSI] = 0.0;

            U[idx2d(i, j, nx)] = cell;
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

    std::vector<double> divB(nx * ny, 0.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            divB[idx2d(i, j, nx)] =
                compute_divB_cell_2d(U, nx, ny, i, j, dx, dy);
        }
    }

    std::vector<State> Uold = U;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            State cell = Uold[idx2d(i, j, nx)];

            // Toy Powell source for the induction equation:
            // dB/dt = -u divB.
            // In a full MHD solver, this must be coupled to the full Powell 8-wave system.
            cell[BX] -= dt * params.powell_vx * divB[idx2d(i, j, nx)];
            cell[BY] -= dt * params.powell_vy * divB[idx2d(i, j, nx)];

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
        // In this standalone B-psi sandbox, EGLM-specific momentum/energy
        // source terms are not active. Therefore we use the same B-psi
        // cleaning subsystem as mixed GLM.
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
    fs::create_directories(fs::path(filename).parent_path());

    std::ofstream fout(filename);

    fout << "i,j,x,y,"
         << "Bx,By,Bz,psi,"
         << "divB,normDivB,Bmag\n";

    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const double x = (i + 0.5) * params.dx;
            const double y = (j + 0.5) * params.dy;

            const State& cell = U[idx2d(i, j, params.nx)];

            const double divB =
                compute_divB_cell_2d(
                    U, params.nx, params.ny, i, j, params.dx, params.dy
                );

            const double normDivB =
                compute_normalized_divB_cell_2d(
                    U, params.nx, params.ny, i, j, params.dx, params.dy
                );

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
                 << divB << ","
                 << normDivB << ","
                 << Bmag << "\n";
        }
    }
}

void run_glm_2d_case(
    CleaningType type,
    GLM2DParams params
) {
    params.dx = params.xlen / static_cast<double>(params.nx);
    params.dy = params.ylen / static_cast<double>(params.ny);

    const double dx_min = std::min(params.dx, params.dy);

    if (type == CleaningType::PARABOLIC) {
        params.dt = 0.20 * dx_min * dx_min / (params.cp * params.cp);
    } else {
        params.dt = params.cfl * dx_min / params.ch;
    }

    int nstep = static_cast<int>(std::ceil(params.t_end / params.dt));

    // Elliptic projection is instantaneous in this standalone sandbox.
    // Keep only two samples: initial and projected.
    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        nstep = 1;
    }

    params.dt = params.t_end / static_cast<double>(nstep);

    std::vector<State> U(params.nx * params.ny);
    initialize_divergence_pulse_2d(U, params);

    const std::string name = cleaning_name(type);

    fs::create_directories("results/divergence");
    fs::create_directories("results/snapshots");

    const std::string diag_name =
        "results/divergence/" + params.out_prefix + "_" + name + ".csv";

    std::ofstream diag(diag_name);

    diag << "step,time,dt,"
         << "L1,L2,Linf,"
         << "L1_norm,L2_norm,Linf_norm\n";

    for (int step = 0; step <= nstep; ++step) {
        const double time = step * params.dt;

        DivBNorms norms =
            compute_divB_norms_2d(
                U, params.nx, params.ny, params.dx, params.dy
            );

        diag << step << ","
             << time << ","
             << params.dt << ","
             << norms.L1 << ","
             << norms.L2 << ","
             << norms.Linf << ","
             << norms.L1_norm << ","
             << norms.L2_norm << ","
             << norms.Linf_norm << "\n";

        if (step == nstep) {
            break;
        }

        advance_glm_2d_one_step(U, type, params);
    }

    if (params.write_snapshot) {
        const std::string snap_name =
            "results/snapshots/" + params.out_prefix + "_" + name + "_final.csv";

        write_glm_2d_snapshot(U, params, snap_name);

        std::cout << "Wrote " << snap_name << "\n";
    }

    std::cout << "Wrote " << diag_name << "\n";
}