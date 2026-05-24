// =============================================================================
//  tests/test_mhd_runner.cpp
//
//  Drives the integrated HLLD + GLM runner on two MHD problems:
//    1. Orszag-Tang vortex   (smooth, periodic, full 2D)
//    2. Brio-Wu shock tube   (1D-in-y replicated 2D strip)
//
//  Each problem is run with several cleaning methods so divB behaviour can be
//  compared in results/divergence/*.csv and results/snapshots/*_final.csv.
// =============================================================================

#include <iostream>
#include <string>
#include <vector>

#include "glm2d_types.hpp"
#include "mhd_runner.hpp"

namespace {

void run_problem(
    const std::string& problem,
    const std::string& prefix,
    double gamma,
    double t_end,
    int N,
    const std::vector<CleaningType>& cases
) {
    MHDRunParams params;
    params.problem = problem;
    params.gamma   = gamma;

    params.glm.nx = N;
    params.glm.ny = N;
    params.glm.xlen = 1.0;
    params.glm.ylen = 1.0;
    params.glm.t_end = t_end;
    params.glm.cfl   = 0.4;
    params.glm.cp    = 0.2;
    params.glm.write_snapshot = true;
    params.glm.write_initial_snapshot = false;
    params.glm.out_prefix = prefix;

    for (CleaningType type : cases) {
        std::cout << "========================================\n";
        std::cout << " " << problem << " | cleaning = "
                  << static_cast<int>(type) << "\n";
        std::cout << "========================================\n";
        run_mhd_2d_case(type, params);
    }
}

} // namespace

int main() {
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

    std::cout << "\n#### Orszag-Tang vortex ####\n";
    run_problem("orszag_tang", "mhd_ot", 5.0 / 3.0, 0.5, 128, cases);

    std::cout << "\n#### Brio-Wu shock tube (2D strip) ####\n";
    run_problem("brio_wu", "mhd_bw", 2.0, 0.2, 128, cases);

    std::cout << "\nAll integrated MHD + cleaning cases finished.\n";
    return 0;
}
