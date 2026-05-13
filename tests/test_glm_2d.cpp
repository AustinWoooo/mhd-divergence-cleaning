#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include "state.hpp"
#include "glm.hpp"
#include "diagnostics.hpp"

static int id(int i, int j, int nx) {
    return j * nx + i;
}

void initialize_divergence_pulse_2d(
    std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
) {
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * dx;
            const double y = (j + 0.5) * dy;

            State cell{};
            for (int v = 0; v < NVAR; ++v) {
                cell[v] = 0.0;
            }

            cell[RHO] = 1.0;
            cell[E]   = 1.0;

            const double r2 =
                (x - 0.5) * (x - 0.5)
              + (y - 0.5) * (y - 0.5);

            // Intentional nonzero divergence source.
            cell[BX] = 1.0 + 0.1 * std::exp(-100.0 * r2);
            cell[BY] = 0.0;
            cell[BZ] = 0.0;
            cell[PSI] = 0.0;

            U[id(i, j, nx)] = cell;
        }
    }
}

void write_snapshot(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy,
    const std::string& filename
) {
    std::ofstream fout(filename);
    fout << "i,j,x,y,Bx,By,Bz,psi,divB\n";

    for (int j = 0; j < ny; ++j) {
        int jp = (j + 1) % ny;
        int jm = (j - 1 + ny) % ny;

        for (int i = 0; i < nx; ++i) {
            int ip = (i + 1) % nx;
            int im = (i - 1 + nx) % nx;

            const double x = (i + 0.5) * dx;
            const double y = (j + 0.5) * dy;

            const double dBx_dx =
                (U[id(ip, j, nx)][BX] - U[id(im, j, nx)][BX]) / (2.0 * dx);

            const double dBy_dy =
                (U[id(i, jp, nx)][BY] - U[id(i, jm, nx)][BY]) / (2.0 * dy);

            const double divB = dBx_dx + dBy_dy;

            const State& c = U[id(i, j, nx)];

            fout << i << ","
                 << j << ","
                 << x << ","
                 << y << ","
                 << c[BX] << ","
                 << c[BY] << ","
                 << c[BZ] << ","
                 << c[PSI] << ","
                 << divB << "\n";
        }
    }
}

void update_hyperbolic_glm_2d(
    std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy,
    const GLMParams& params
) {
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

    // x-direction faces
    for (int j = 0; j < ny; ++j) {
        for (int iface = 0; iface <= nx; ++iface) {
            int iL = (iface - 1 + nx) % nx;
            int iR = iface % nx;

            const State& UL = U[id(iL, j, nx)];
            const State& UR = U[id(iR, j, nx)];

            flux_x[fx_id(iface, j)] =
                compute_hyperbolic_glm_flux(
                    UL[BX], UL[PSI],
                    UR[BX], UR[PSI],
                    ch
                );
        }
    }

    // y-direction faces
    for (int jface = 0; jface <= ny; ++jface) {
        int jL = (jface - 1 + ny) % ny;
        int jR = jface % ny;

        for (int i = 0; i < nx; ++i) {
            const State& UL = U[id(i, jL, nx)];
            const State& UR = U[id(i, jR, nx)];

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

            State cell = Uold[id(i, j, nx)];

            cell[BX] =
                Uold[id(i, j, nx)][BX]
              - dt / dx * (fxR.FBn - fxL.FBn);

            cell[BY] =
                Uold[id(i, j, nx)][BY]
              - dt / dy * (fyR.FBn - fyL.FBn);

            cell[PSI] =
                Uold[id(i, j, nx)][PSI]
              - dt / dx * (fxR.Fpsi - fxL.Fpsi)
              - dt / dy * (fyR.Fpsi - fyL.Fpsi);

            U[id(i, j, nx)] = cell;
        }
    }
}

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLMParams& params
) {
    const double factor =
        std::exp(-params.dt * params.ch * params.ch / (params.cp * params.cp));

    for (auto& cell : U) {
        cell[PSI] *= factor;
    }
}

void run_case(CleaningType type) {
    const int nx = 128;
    const int ny = 128;

    const double xlen = 1.0;
    const double ylen = 1.0;

    const double dx = xlen / nx;
    const double dy = ylen / ny;

    GLMParams params;
    params.type = type;
    params.dx = dx;
    params.ch = 1.0;
    params.cp = 0.2;

    // Conservative 2D CFL for waves propagating in x and y.
    params.dt = 0.25 * std::min(dx, dy) / params.ch;

    const double t_end = 0.5;
    const int nstep = static_cast<int>(std::ceil(t_end / params.dt));
    params.dt = t_end / nstep;

    std::vector<State> U(nx * ny);
    initialize_divergence_pulse_2d(U, nx, ny, dx, dy);

    const std::string name = cleaning_name(type);
    const std::string diag_name =
        "results/divergence/glm_2d_" + name + ".csv";

    std::ofstream diag(diag_name);
    diag << "step,time,L1,L2,Linf\n";

    for (int step = 0; step <= nstep; ++step) {
        const double time = step * params.dt;

        DivBNorms norms =
            compute_divB_norms_2d(U, nx, ny, dx, dy);

        diag << step << ","
             << time << ","
             << norms.L1 << ","
             << norms.L2 << ","
             << norms.Linf << "\n";

        if (step == nstep) break;

        if (type == CleaningType::NONE) {
            // no update
        } else if (type == CleaningType::HYPERBOLIC_GLM) {
            update_hyperbolic_glm_2d(U, nx, ny, dx, dy, params);
        } else if (type == CleaningType::MIXED_GLM) {
            update_hyperbolic_glm_2d(U, nx, ny, dx, dy, params);
            apply_mixed_glm_damping_2d(U, params);
        }
    }

    const std::string snap_name =
        "results/snapshots/glm_2d_" + name + "_final.csv";

    write_snapshot(U, nx, ny, dx, dy, snap_name);

    std::cout << "Wrote " << diag_name << "\n";
    std::cout << "Wrote " << snap_name << "\n";
}

int main() {
    run_case(CleaningType::NONE);
    run_case(CleaningType::HYPERBOLIC_GLM);
    run_case(CleaningType::MIXED_GLM);

    return 0;
}