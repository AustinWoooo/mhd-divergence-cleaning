#pragma once

#include <vector>
#include "state.hpp"

struct DivBNorms {
    double L1 = 0.0;
    double L2 = 0.0;
    double Linf = 0.0;
};

inline int idx2d(int i, int j, int nx) {
    return j * nx + i;
}

DivBNorms compute_divB_norms_2d(
    const std::vector<State>& U,
    int nx,
    int ny,
    double dx,
    double dy
);