#pragma once

#include <string>
#include <vector>

#include "state.hpp"

constexpr double DIVB_NORM_EPS = 1.0e-12;

inline int idx2d(int i, int j, int nx) {
    return j * nx + i;
}

int periodic_index(int i, int n);

void ensure_parent_directory(const std::string& filename);

double analytic_divergence_pulse_2d(double x, double y);

double compute_fv_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
);

double compute_fv_normalized_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
);

struct LocalDivBNorms {
    double L1 = 0.0;
    double L2 = 0.0;
    double Linf = 0.0;

    double L1_norm = 0.0;
    double L2_norm = 0.0;
    double Linf_norm = 0.0;
};

LocalDivBNorms compute_fv_divB_norms_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
);

std::vector<double> compute_fv_divB_field_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
);
