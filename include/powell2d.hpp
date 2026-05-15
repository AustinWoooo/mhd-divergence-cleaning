#pragma once

#include <string>
#include <vector>

#include "state.hpp"
#include "diagnostics.hpp"

struct Powell2DParams {
    int nx = 128;
    int ny = 128;

    double xlen = 1.0;
    double ylen = 1.0;

    double dx = 1.0 / 128.0;
    double dy = 1.0 / 128.0;

    double dt = 1.0e-3;
    double t_end = 0.5;
    double cfl = 0.25;

    double vx = 1.0;
    double vy = 0.5;

    bool write_snapshot = true;
    std::string out_prefix = "powell_2d";
};

void initialize_powell_test_field_2d(
    std::vector<State>& U,
    const Powell2DParams& params
);

void apply_powell_source_2d(
    std::vector<State>& U,
    const Powell2DParams& params
);

void advance_powell_2d_one_step(
    std::vector<State>& U,
    const Powell2DParams& params
);

void write_powell_2d_snapshot(
    const std::vector<State>& U,
    const Powell2DParams& params,
    const std::string& filename
);

void run_powell_2d_case(
    Powell2DParams params
);