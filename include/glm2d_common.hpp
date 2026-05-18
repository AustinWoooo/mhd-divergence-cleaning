#pragma once

#include "glm2d.hpp"

#include <string>
#include <vector>

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