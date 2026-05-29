#pragma once

#include <limits>
#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "state.hpp"

struct PowellControlBadState {
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

struct PowellControlCellDiagnostic {
    bool found = false;
    int i = -1;
    int j = -1;
    double time = 0.0;
    double dt = 0.0;
    double divB = 0.0;
    double theta = 1.0;
    State before{};
    State after{};
    State full_delta{};
};

struct PowellControlResult {
    int subcycles = 0;
    PowellControlBadState bad_state;
    std::string failure_stage;
    double failure_time = std::numeric_limits<double>::quiet_NaN();

    double min_pressure_before_source =
        std::numeric_limits<double>::infinity();
    double min_density_before_source =
        std::numeric_limits<double>::infinity();
    double min_pressure_after_source =
        std::numeric_limits<double>::infinity();
    double min_density_after_source =
        std::numeric_limits<double>::infinity();

    double max_abs_divB = 0.0;
    long long total_limited_cells = 0;
    double theta_min = 1.0;
    double theta_sum_limited = 0.0;
    long long limiter_activations = 0;

    PowellControlCellDiagnostic first_bad_cell;
    PowellControlCellDiagnostic first_limited_cell;
};

int compute_powell_subcycle_count(
    const std::vector<State>& U,
    const GLM2DParams& params,
    double dt,
    double source_cfl
);

void apply_powell_subcycled_2d(
    std::vector<State>& U,
    GLM2DParams params,
    double dt_mhd,
    double gamma,
    double source_cfl
);

PowellControlResult apply_powell_subcycled_2d(
    std::vector<State>& U,
    GLM2DParams params,
    double dt_mhd,
    double gamma,
    double source_cfl,
    double start_time
);

double find_pressure_safe_theta(
    const State& base,
    const State& dU,
    double gamma,
    double p_floor,
    double rho_floor
);

PowellControlResult apply_powell_limited_2d(
    std::vector<State>& U,
    const GLM2DParams& params,
    double dt_mhd,
    double gamma,
    double p_floor,
    double rho_floor,
    double start_time,
    bool capture_first_limited_cell
);
