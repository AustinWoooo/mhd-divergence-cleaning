#include <cassert>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "eglm2d.hpp"
#include "galilean_glm2d.hpp"
#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_common.hpp"
#include "mixed_glm2d.hpp"
#include "parabolic2d.hpp"
#include "powell2d.hpp"
#include "projection2d.hpp"

namespace {

bool approx_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

double raw_pressure(const State& cell, double gamma) {
    const double rho = cell[RHO];
    const double ux = cell[MX] / rho;
    const double uy = cell[MY] / rho;
    const double uz = cell[MZ] / rho;
    const double kinetic =
        0.5 * rho * (ux * ux + uy * uy + uz * uz);
    const double magnetic =
        0.5 * (
            cell[BX] * cell[BX]
          + cell[BY] * cell[BY]
          + cell[BZ] * cell[BZ]
        );
    return (gamma - 1.0) * (cell[E] - kinetic - magnetic);
}

void set_total_energy_from_pressure(State& cell, double pressure, double gamma) {
    const double rho = cell[RHO];
    const double ux = cell[MX] / rho;
    const double uy = cell[MY] / rho;
    const double uz = cell[MZ] / rho;
    const double kinetic =
        0.5 * rho * (ux * ux + uy * uy + uz * uz);
    const double magnetic =
        0.5 * (
            cell[BX] * cell[BX]
          + cell[BY] * cell[BY]
          + cell[BZ] * cell[BZ]
        );
    cell[E] = pressure / (gamma - 1.0) + kinetic + magnetic;
}

void fill_divergent_test_state(std::vector<State>& U, int nx, int ny) {
    const double pi = std::acos(-1.0);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) / static_cast<double>(nx);
            const double y = (j + 0.5) / static_cast<double>(ny);

            State cell{};
            cell.fill(0.0);

            cell[RHO] = 1.0 + 0.01 * i + 0.02 * j;
            cell[MX] = 0.2 + 0.03 * i;
            cell[MY] = -0.1 + 0.02 * j;
            cell[MZ] = 0.05 + 0.01 * (i + j);

            cell[BX] = 1.0 + 0.20 * std::sin(2.0 * pi * x)
                            + 0.05 * std::cos(2.0 * pi * y);
            cell[BY] = -0.4 + 0.15 * std::sin(2.0 * pi * y)
                             - 0.04 * std::cos(2.0 * pi * x);
            cell[BZ] = 0.25 + 0.03 * std::sin(2.0 * pi * (x + y));

            cell[E] = 2.5 + 0.01 * i + 0.02 * j;
            cell[PSI] = 0.3 * std::sin(2.0 * pi * x)
                      - 0.2 * std::cos(2.0 * pi * y);

            U[idx2d(i, j, nx)] = cell;
        }
    }
}

void assert_all_finite(const std::vector<State>& U) {
    for (const State& cell : U) {
        for (double value : cell) {
            assert(std::isfinite(value));
        }
    }
}

void assert_positive_raw_pressure(
    const std::vector<State>& U,
    double gamma
) {
    for (const State& cell : U) {
        assert(cell[RHO] > 0.0);
        assert(raw_pressure(cell, gamma) > 0.0);
    }
}

void test_eglm_source_one_step() {
    GLM2DParams params;
    params.nx = 4;
    params.ny = 4;
    params.dx = 1.0;
    params.dy = 1.0;
    params.dt = 0.1;

    const int nx = params.nx;
    const int ny = params.ny;

    std::vector<State> Uold(nx * ny);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            State cell{};
            cell.fill(0.0);

            cell[RHO] = 1.0 + 0.1 * i + 0.05 * j;
            cell[MX] = 0.3 + 0.02 * i;
            cell[MY] = -0.4 + 0.03 * j;
            cell[MZ] = 0.2 + 0.01 * (i + j);

            cell[BX] = 1.0 + 0.2 * i + 0.1 * j;
            cell[BY] = -0.5 + 0.3 * i - 0.05 * j;
            cell[BZ] = 0.25 + 0.07 * i + 0.11 * j;

            cell[E] = 2.0 + 0.2 * i + 0.15 * j;
            cell[PSI] = 0.4 * i * i - 0.2 * j + 0.1 * i * j;

            Uold[idx2d(i, j, nx)] = cell;
        }
    }

    std::vector<State> U = Uold;

    apply_eglm_source_2d(U, Uold, params);

    const std::vector<double> divB =
        compute_fv_divB_field_2d(Uold, nx, ny, params.dx, params.dy);

    const int i = 2;
    const int j = 1;
    const int id = idx2d(i, j, nx);

    const int ip = periodic_index(i + 1, nx);
    const int im = periodic_index(i - 1, nx);
    const int jp = periodic_index(j + 1, ny);
    const int jm = periodic_index(j - 1, ny);

    const double dpsidx =
        (Uold[idx2d(ip, j, nx)][PSI] - Uold[idx2d(im, j, nx)][PSI])
      / (2.0 * params.dx);

    const double dpsidy =
        (Uold[idx2d(i, jp, nx)][PSI] - Uold[idx2d(i, jm, nx)][PSI])
      / (2.0 * params.dy);

    const double Bx = Uold[id][BX];
    const double By = Uold[id][BY];
    const double Bz = Uold[id][BZ];

    assert(approx_equal(U[id][MX], Uold[id][MX] - params.dt * divB[id] * Bx));
    assert(approx_equal(U[id][MY], Uold[id][MY] - params.dt * divB[id] * By));
    assert(approx_equal(U[id][MZ], Uold[id][MZ] - params.dt * divB[id] * Bz));
    assert(approx_equal(
        U[id][E],
        Uold[id][E] - params.dt * (Bx * dpsidx + By * dpsidy)
    ));

    assert(approx_equal(U[id][BX], Uold[id][BX]));
    assert(approx_equal(U[id][BY], Uold[id][BY]));
    assert(approx_equal(U[id][BZ], Uold[id][BZ]));
    assert(approx_equal(U[id][PSI], Uold[id][PSI]));

    std::cout << "EGLM one-step source assertion passed.\n";
}

void test_powell_source_one_step() {
    GLM2DParams params;
    params.nx = 4;
    params.ny = 4;
    params.dx = 1.0;
    params.dy = 1.0;
    params.dt = 0.01;

    const int nx = params.nx;
    const int ny = params.ny;

    std::vector<State> U(nx * ny);
    fill_divergent_test_state(U, nx, ny);

    const std::vector<State> Uold = U;
    const std::vector<double> divB =
        compute_fv_divB_field_2d(Uold, nx, ny, params.dx, params.dy);

    apply_powell_source_2d(U, params);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);
            const State& old = Uold[id];
            const State& cell = U[id];

            const double rho = old[RHO];
            const double ux = old[MX] / rho;
            const double uy = old[MY] / rho;
            const double uz = old[MZ] / rho;
            const double u_dot_B =
                ux * old[BX] + uy * old[BY] + uz * old[BZ];

            assert(approx_equal(cell[RHO], old[RHO]));
            assert(approx_equal(cell[MX], old[MX] - params.dt * divB[id] * old[BX]));
            assert(approx_equal(cell[MY], old[MY] - params.dt * divB[id] * old[BY]));
            assert(approx_equal(cell[MZ], old[MZ] - params.dt * divB[id] * old[BZ]));
            assert(approx_equal(cell[BX], old[BX] - params.dt * divB[id] * ux));
            assert(approx_equal(cell[BY], old[BY] - params.dt * divB[id] * uy));
            assert(approx_equal(cell[BZ], old[BZ] - params.dt * divB[id] * uz));
            assert(approx_equal(cell[E], old[E] - params.dt * divB[id] * u_dot_B));
            assert(approx_equal(cell[PSI], old[PSI]));
        }
    }

    assert_all_finite(U);
    std::cout << "Powell one-step source assertion passed.\n";
}

void test_gi_eglm_source_one_step() {
    GLM2DParams params;
    params.nx = 4;
    params.ny = 4;
    params.dx = 1.0;
    params.dy = 1.0;
    params.dt = 0.01;

    const int nx = params.nx;
    const int ny = params.ny;

    std::vector<State> U(nx * ny);
    fill_divergent_test_state(U, nx, ny);

    const std::vector<State> Uold = U;
    const std::vector<double> divB =
        compute_fv_divB_field_2d(Uold, nx, ny, params.dx, params.dy);

    apply_gi_eglm_source_2d(U, Uold, params);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = idx2d(i, j, nx);

            const int ip = periodic_index(i + 1, nx);
            const int im = periodic_index(i - 1, nx);
            const int jp = periodic_index(j + 1, ny);
            const int jm = periodic_index(j - 1, ny);

            const State& old = Uold[id];
            const State& cell = U[id];

            const double rho = old[RHO];
            const double ux = old[MX] / rho;
            const double uy = old[MY] / rho;
            const double uz = old[MZ] / rho;

            const double dpsidx =
                (Uold[idx2d(ip, j, nx)][PSI] - Uold[idx2d(im, j, nx)][PSI])
              / (2.0 * params.dx);

            const double dpsidy =
                (Uold[idx2d(i, jp, nx)][PSI] - Uold[idx2d(i, jm, nx)][PSI])
              / (2.0 * params.dy);

            const double u_dot_B =
                ux * old[BX] + uy * old[BY] + uz * old[BZ];
            const double B_dot_grad_psi =
                old[BX] * dpsidx + old[BY] * dpsidy;
            const double u_dot_grad_psi = ux * dpsidx + uy * dpsidy;

            assert(approx_equal(cell[RHO], old[RHO]));
            assert(approx_equal(cell[MX], old[MX] - params.dt * divB[id] * old[BX]));
            assert(approx_equal(cell[MY], old[MY] - params.dt * divB[id] * old[BY]));
            assert(approx_equal(cell[MZ], old[MZ] - params.dt * divB[id] * old[BZ]));
            assert(approx_equal(cell[BX], old[BX] - params.dt * divB[id] * ux));
            assert(approx_equal(cell[BY], old[BY] - params.dt * divB[id] * uy));
            assert(approx_equal(cell[BZ], old[BZ] - params.dt * divB[id] * uz));
            assert(approx_equal(
                cell[E],
                old[E] - params.dt * (divB[id] * u_dot_B + B_dot_grad_psi)
            ));
            assert(approx_equal(
                cell[PSI],
                old[PSI] - params.dt * u_dot_grad_psi
            ));
        }
    }

    assert_all_finite(U);
    std::cout << "GI mixed EGLM one-step source assertion passed.\n";
}

void test_projection_reduces_fv_divB() {
    GLM2DParams params;
    params.nx = 16;
    params.ny = 16;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.poisson_max_iter = 50000;
    params.poisson_tol = 1.0e-12;
    params.poisson_omega = 1.7;

    std::vector<State> U(params.nx * params.ny);
    fill_divergent_test_state(U, params.nx, params.ny);

    const LocalDivBNorms initial =
        compute_fv_divB_norms_2d(U, params.nx, params.ny, params.dx, params.dy);

    const ProjectionResult projection =
        apply_elliptic_projection_2d(U, params);

    const LocalDivBNorms final =
        compute_fv_divB_norms_2d(U, params.nx, params.ny, params.dx, params.dy);

    assert_all_finite(U);
    assert(std::isfinite(projection.true_residual_Linf));
    assert(projection.true_residual_Linf < 1.0e-7);
    assert(initial.L2 > 0.0);
    assert(final.L2 < initial.L2);

    const double reduction = final.L2 / initial.L2;
    std::cout << "Projection FV L2 reduction factor: " << reduction << "\n";
    assert(reduction < 1.0e-3);
}

void test_projection_preserves_uniform_field() {
    GLM2DParams params;
    params.nx = 16;
    params.ny = 16;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);

    std::vector<State> U(params.nx * params.ny);
    for (State& cell : U) {
        cell.fill(0.0);
        cell[RHO] = 1.0;
        cell[BX] = 0.7;
        cell[BY] = -0.2;
        cell[BZ] = 0.1;
        cell[E] = 2.0;
    }

    const std::vector<State> Uold = U;
    const ProjectionResult projection =
        apply_elliptic_projection_2d(U, params);

    assert_all_finite(U);
    assert(projection.true_residual_Linf < 1.0e-12);
    for (std::size_t id = 0; id < U.size(); ++id) {
        assert(approx_equal(U[id][BX], Uold[id][BX]));
        assert(approx_equal(U[id][BY], Uold[id][BY]));
        assert(approx_equal(U[id][BZ], Uold[id][BZ]));
    }
    std::cout << "Projection uniform-field assertion passed.\n";
}

void test_projection_moderate_case_keeps_pressure_positive() {
    constexpr double gamma = 5.0 / 3.0;
    GLM2DParams params;
    params.nx = 32;
    params.ny = 32;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.poisson_max_iter = 50000;
    params.poisson_tol = 1.0e-11;

    std::vector<State> U(params.nx * params.ny);
    fill_divergent_test_state(U, params.nx, params.ny);
    for (State& cell : U) {
        cell[RHO] = 1.0;
        cell[MX] = 0.05;
        cell[MY] = -0.02;
        cell[MZ] = 0.0;
        set_total_energy_from_pressure(cell, 1.0, gamma);
    }

    apply_elliptic_projection_2d(U, params);
    assert_all_finite(U);
    assert_positive_raw_pressure(U, gamma);
    std::cout << "Projection pressure assertion passed.\n";
}

void test_parabolic_reduces_fv_divB() {
    GLM2DParams params;
    params.nx = 16;
    params.ny = 16;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.cp = 0.4;
    params.dt = 0.001;

    std::vector<State> U(params.nx * params.ny);
    fill_divergent_test_state(U, params.nx, params.ny);

    const LocalDivBNorms initial =
        compute_fv_divB_norms_2d(U, params.nx, params.ny, params.dx, params.dy);

    for (int step = 0; step < 20; ++step) {
        apply_parabolic_cleaning_2d(U, params);
        assert_all_finite(U);
    }

    const LocalDivBNorms final =
        compute_fv_divB_norms_2d(U, params.nx, params.ny, params.dx, params.dy);

    assert(initial.L2 > 0.0);
    assert(final.L2 < initial.L2);
    std::cout << "Parabolic FV L2 reduction factor: "
              << final.L2 / initial.L2 << "\n";
}

void test_parabolic_moderate_case_keeps_pressure_positive() {
    constexpr double gamma = 5.0 / 3.0;
    GLM2DParams params;
    params.nx = 32;
    params.ny = 32;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.cp = 0.2;
    params.dt = 0.2 * params.dx * params.dx / (params.cp * params.cp);

    std::vector<State> U(params.nx * params.ny);
    fill_divergent_test_state(U, params.nx, params.ny);
    for (State& cell : U) {
        cell[RHO] = 1.0;
        cell[MX] = 0.05;
        cell[MY] = -0.02;
        cell[MZ] = 0.0;
        set_total_energy_from_pressure(cell, 1.0, gamma);
    }

    for (int step = 0; step < 10; ++step) {
        const std::vector<State> Uold = U;
        apply_parabolic_cleaning_2d(U, params);

        for (std::size_t id = 0; id < U.size(); ++id) {
            const double old_me =
                0.5 * (
                    Uold[id][BX] * Uold[id][BX]
                  + Uold[id][BY] * Uold[id][BY]
                  + Uold[id][BZ] * Uold[id][BZ]
                );
            const double new_me =
                0.5 * (
                    U[id][BX] * U[id][BX]
                  + U[id][BY] * U[id][BY]
                  + U[id][BZ] * U[id][BZ]
                );
            U[id][E] = Uold[id][E] + (new_me - old_me);
        }
        assert_all_finite(U);
        assert_positive_raw_pressure(U, gamma);
    }

    std::cout << "Parabolic pressure assertion passed.\n";
}

double periodic_delta(double a, double b) {
    double d = a - b;
    while (d > 0.5) d -= 1.0;
    while (d < -0.5) d += 1.0;
    return d;
}

double positive_bump_centroid_x(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx
) {
    double sum_w = 0.0;
    double sum_x = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const State& cell = U[idx2d(i, j, nx)];
            const double w = std::max(cell[BX] - 1.0, 0.0);
            const double x = (i + 0.5) * dx;
            sum_w += w;
            sum_x += w * x;
        }
    }
    return sum_x / sum_w;
}

void test_powell_advects_bump_with_uniform_velocity() {
    GLM2DParams params;
    params.nx = 128;
    params.ny = 4;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.dt = 0.10 * params.dx;

    const double vx = 1.0;
    const double t_end = 0.05;
    const int nstep = static_cast<int>(std::round(t_end / params.dt));
    params.dt = t_end / static_cast<double>(nstep);

    std::vector<State> U(params.nx * params.ny);
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const double x = (i + 0.5) * params.dx;
            const double r = x - 0.25;
            const double bump = std::exp(-400.0 * r * r);

            State cell{};
            cell.fill(0.0);
            cell[RHO] = 1.0;
            cell[MX] = vx;
            cell[MY] = 0.0;
            cell[MZ] = 0.0;
            cell[BX] = 1.0 + 0.02 * bump;
            cell[BY] = 0.0;
            cell[BZ] = 0.0;
            cell[E] = 4.0;
            U[idx2d(i, j, params.nx)] = cell;
        }
    }

    const double x0 =
        positive_bump_centroid_x(U, params.nx, params.ny, params.dx);

    for (int step = 0; step < nstep; ++step) {
        apply_powell_source_2d(U, params);
        assert_all_finite(U);
    }

    const double x1 =
        positive_bump_centroid_x(U, params.nx, params.ny, params.dx);
    const double shift = periodic_delta(x1, x0);

    assert(std::abs(shift - vx * t_end) < 2.0 * params.dx);
    std::cout << "Powell advection assertion passed: shift="
              << shift << "\n";
}

void test_mixed_glm_damping_factor() {
    GLM2DParams params;
    params.nx = 4;
    params.ny = 4;
    params.ch = 1.5;
    params.cp = 0.5;
    params.dt = 0.02;

    std::vector<State> U(params.nx * params.ny);

    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            State cell{};
            cell.fill(0.0);
            cell[RHO] = 1.0;
            cell[BX] = 0.2 + 0.01 * i;
            cell[BY] = -0.3 + 0.02 * j;
            cell[BZ] = 0.1;
            cell[E] = 1.0;
            cell[PSI] = 0.5 + 0.1 * i - 0.05 * j;
            U[idx2d(i, j, params.nx)] = cell;
        }
    }

    const std::vector<State> Uold = U;
    const double factor =
        std::exp(-params.dt * params.ch * params.ch / (params.cp * params.cp));

    apply_mixed_glm_damping_2d(U, params);

    for (int id = 0; id < static_cast<int>(U.size()); ++id) {
        assert(approx_equal(U[id][PSI], Uold[id][PSI] * factor));
        assert(approx_equal(U[id][BX], Uold[id][BX]));
        assert(approx_equal(U[id][BY], Uold[id][BY]));
        assert(approx_equal(U[id][BZ], Uold[id][BZ]));
    }

    assert_all_finite(U);
    std::cout << "Mixed GLM damping assertion passed.\n";
}

void initialize_sinusoidal_divergence_state(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    constexpr double gamma = 5.0 / 3.0;
    constexpr double amplitude = 0.05;
    const double pi = std::acos(-1.0);

    for (int j = 0; j < params.ny; ++j) {
        const double y = (j + 0.5) * params.dy;
        for (int i = 0; i < params.nx; ++i) {
            const double x = (i + 0.5) * params.dx;

            State cell{};
            cell.fill(0.0);
            cell[RHO] = 1.0;
            cell[BX] =
                amplitude
              * std::sin(2.0 * pi * x)
              * std::sin(2.0 * pi * y);
            cell[BY] = 0.0;
            cell[BZ] = 0.0;
            cell[PSI] = 0.0;
            set_total_energy_from_pressure(cell, 1.0, gamma);

            U[idx2d(i, j, params.nx)] = cell;
        }
    }
}

void run_pure_cleaning_steps(
    std::vector<State>& U,
    CleaningType type,
    GLM2DParams params,
    double t_end
) {
    if (type == CleaningType::NONE) {
        return;
    }

    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        apply_elliptic_projection_2d(U, params);
        return;
    }

    const double stable_dt =
        max_cleaning_dt(type, params.dx, params.dy, params);
    const int nstep =
        std::max(1, static_cast<int>(std::ceil(t_end / stable_dt)));
    params.dt = t_end / static_cast<double>(nstep);

    for (int step = 0; step < nstep; ++step) {
        advance_glm_2d_one_step(U, type, params);
        assert_all_finite(U);
    }
}

void test_pure_glm_cleaning_reduces_sinusoidal_divergence() {
    GLM2DParams params;
    params.nx = 32;
    params.ny = 32;
    params.dx = 1.0 / static_cast<double>(params.nx);
    params.dy = 1.0 / static_cast<double>(params.ny);
    params.ch = 1.0;
    params.cp = 0.2;
    params.cfl = 0.25;
    params.poisson_max_iter = 50000;
    params.poisson_tol = 1.0e-12;
    params.poisson_omega = 1.7;

    const double t_end = 0.5;

    std::vector<State> initial(params.nx * params.ny);
    initialize_sinusoidal_divergence_state(initial, params);

    const LocalDivBNorms initial_norms =
        compute_fv_divB_norms_2d(
            initial,
            params.nx,
            params.ny,
            params.dx,
            params.dy
        );
    assert(initial_norms.L2 > 0.0);

    std::vector<State> none = initial;
    std::vector<State> hyperbolic = initial;
    std::vector<State> mixed = initial;
    std::vector<State> projection = initial;

    run_pure_cleaning_steps(none, CleaningType::NONE, params, t_end);
    run_pure_cleaning_steps(
        hyperbolic,
        CleaningType::HYPERBOLIC_GLM,
        params,
        t_end
    );
    run_pure_cleaning_steps(mixed, CleaningType::MIXED_GLM, params, t_end);
    run_pure_cleaning_steps(
        projection,
        CleaningType::ELLIPTIC_PROJECTION,
        params,
        t_end
    );

    const LocalDivBNorms none_norms =
        compute_fv_divB_norms_2d(none, params.nx, params.ny, params.dx, params.dy);
    const LocalDivBNorms hyperbolic_norms =
        compute_fv_divB_norms_2d(
            hyperbolic,
            params.nx,
            params.ny,
            params.dx,
            params.dy
        );
    const LocalDivBNorms mixed_norms =
        compute_fv_divB_norms_2d(mixed, params.nx, params.ny, params.dx, params.dy);
    const LocalDivBNorms projection_norms =
        compute_fv_divB_norms_2d(
            projection,
            params.nx,
            params.ny,
            params.dx,
            params.dy
        );

    assert(approx_equal(none_norms.L2, initial_norms.L2));
    assert(hyperbolic_norms.L2 < none_norms.L2);
    assert(mixed_norms.L2 < 0.5 * none_norms.L2);
    assert(projection_norms.L2 < 1.0e-6 * none_norms.L2);

    std::cout << "Pure GLM sinusoidal FV L2 ratios:"
              << " hyperbolic=" << hyperbolic_norms.L2 / none_norms.L2
              << " mixed=" << mixed_norms.L2 / none_norms.L2
              << " projection=" << projection_norms.L2 / none_norms.L2
              << "\n";
}

} // namespace

int main() {
    test_eglm_source_one_step();
    test_powell_source_one_step();
    test_gi_eglm_source_one_step();
    test_projection_reduces_fv_divB();
    test_projection_preserves_uniform_field();
    test_projection_moderate_case_keeps_pressure_positive();
    test_parabolic_reduces_fv_divB();
    test_parabolic_moderate_case_keeps_pressure_positive();
    test_powell_advects_bump_with_uniform_velocity();
    test_mixed_glm_damping_factor();
    test_pure_glm_cleaning_reduces_sinusoidal_divergence();

    GLM2DParams params;

    // ============================================================
    // Hard-coded 2D GLM test configuration
    // ============================================================

    params.nx = 128;
    params.ny = 128;

    params.xlen = 1.0;
    params.ylen = 1.0;

    params.t_end = 0.5;

    params.ch = 1.0;
    params.cp = 0.2;

    params.cfl = 0.25;

    params.powell_vx = 1.0;
    params.powell_vy = 0.5;

    params.poisson_max_iter = 10000;
    params.poisson_tol = 1.0e-10;

    params.write_snapshot = true;
    params.out_prefix = "glm_2d";

    // ============================================================
    // Select cleaning methods to run
    // ============================================================

    const std::vector<CleaningType> cases = {
        CleaningType::NONE,
        CleaningType::HYPERBOLIC_GLM,
        CleaningType::MIXED_GLM,
        CleaningType::PARABOLIC,
        CleaningType::ELLIPTIC_PROJECTION,
        CleaningType::POWELL_SOURCE,
        CleaningType::MIXED_EGLM,
        CleaningType::GI_MIXED_EGLM
    };
    // ============================================================
    // Run all selected cases
    // ============================================================

    for (CleaningType type : cases) {
        GLM2DParams case_params = params;
        if (type == CleaningType::POWELL_SOURCE) {
            // The standalone Powell source is a non-conservative transport
            // control, not a robust cleaning method. Keep this diagnostic
            // short so it cannot silently run into nonfinite output.
            case_params.t_end = 0.05;
        }

        std::cout << "========================================\n";
        std::cout << "Running 2D GLM case: "
                  << cleaning_name(type)
                  << "\n";
        std::cout << "========================================\n";

        run_glm_2d_case(type, case_params);
    }

    std::cout << "All 2D GLM cleaning cases finished.\n";

    return 0;
}
