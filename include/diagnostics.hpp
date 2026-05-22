#pragma once

#include <vector>
#include "state.hpp"

struct DivBNorms {
    double L1 = 0.0;
    double L2 = 0.0;
    double Linf = 0.0;

    // Normalized divergence:
    // dx_eff * |divB| / (|B| + eps)
    double L1_norm = 0.0;
    double L2_norm = 0.0;
    double Linf_norm = 0.0;
};

double compute_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
);

double compute_normalized_divB_cell_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    int i,
    int j,
    double dx,
    double dy
);

DivBNorms compute_divB_norms_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
);
