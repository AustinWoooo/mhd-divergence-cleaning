#include <iostream>
#include <vector>

#include "glm2d.hpp"

int main() {
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
        CleaningType::MIXED_EGLM
    };

    // ============================================================
    // Run all selected cases
    // ============================================================

    for (CleaningType type : cases) {
        std::cout << "========================================\n";
        std::cout << "Running 2D GLM case: "
                  << cleaning_name(type)
                  << "\n";
        std::cout << "========================================\n";

        run_glm_2d_case(type, params);
    }

    std::cout << "All 2D GLM cleaning cases finished.\n";

    return 0;
}