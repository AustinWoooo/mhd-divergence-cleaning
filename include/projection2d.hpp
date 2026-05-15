#pragma once

#include <string>
#include <vector>

#include "state.hpp"
#include "diagnostics.hpp"

struct Projection2DParams {
    int nx = 128;
    int ny = 128;

    double xlen = 1.0;
    double ylen = 1.0;

    double dx = 1.0 / 128.0;
    double dy = 1.0 / 128.0;

    int poisson_max_iter = 50000;
    double poisson_tol = 1.0e-12;
    double poisson_omega = 1.7;

    bool write_snapshot = true;
    std::string out_prefix = "projection_2d";
};

void initialize_projection_test_field_2d(
    std::vector<State>& U,
    const Projection2DParams& params
);

void apply_elliptic_projection_2d(
    std::vector<State>& U,
    const Projection2DParams& params
);

void write_projection_2d_snapshot(
    const std::vector<State>& U,
    const Projection2DParams& params,
    const std::string& filename
);

void run_projection_2d_case(
    Projection2DParams params
);