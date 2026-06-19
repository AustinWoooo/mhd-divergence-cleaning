#include "hyperbolic_glm2d.hpp"

#include <vector>

#include "glm.hpp"
#include "glm2d_common.hpp"

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

    if (params.energy_policy ==
        CleaningEnergyPolicy::PreserveThermalPressure) {
        preserve_thermal_pressure_after_magnetic_update(U, Uold);
    }
}
