// =============================================================================
//  tests/test_mhd_runner.cpp
//
//  Drives the integrated HLLD + GLM runner on two MHD problems:
//    1. Orszag-Tang vortex   (smooth, periodic, full 2D)
//    2. Brio-Wu shock tube   (1D-in-y replicated 2D strip)
//
//  Each problem is run with several cleaning methods so divB behaviour can be
//  compared in results/divergence/*.csv and results/snapshots/*_final.csv.
//
//  Usage:
//    ./test_mhd_runner                              # all problems, all cleanings
//    ./test_mhd_runner orszag_tang                  # one problem, all cleanings
//    ./test_mhd_runner brio_wu mixed_glm            # one problem, one cleaning
//    ./test_mhd_runner orszag_tang none mixed_glm   # one problem, multiple cleanings
//
//  Cleaning type names (lowercase):
//    none, parabolic, hyperbolic_glm, mixed_glm, elliptic_projection,
//    powell_source, mixed_eglm, gi_mixed_eglm
// =============================================================================

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "glm2d.hpp"
#include "glm2d_types.hpp"
#include "mhd_runner.hpp"

namespace {

const std::vector<CleaningType> ALL_CASES = {
    CleaningType::NONE,
    CleaningType::HYPERBOLIC_GLM,
    CleaningType::MIXED_GLM,
    CleaningType::PARABOLIC,
    CleaningType::ELLIPTIC_PROJECTION,
    CleaningType::POWELL_SOURCE,
    CleaningType::MIXED_EGLM,
    CleaningType::GI_MIXED_EGLM
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [problem] [cleaning...]\n"
        << "\n"
        << "  problem   : orszag_tang | brio_wu  (default: both)\n"
        << "  cleaning  : none | parabolic | hyperbolic_glm | mixed_glm\n"
        << "              elliptic_projection | powell_source\n"
        << "              mixed_eglm | gi_mixed_eglm  (default: all)\n"
        << "\n"
        << "Examples:\n"
        << "  " << prog << "\n"
        << "      Run all problems with all cleaning methods.\n"
        << "\n"
        << "  " << prog << " orszag_tang\n"
        << "      Run Orszag-Tang vortex with all cleaning methods.\n"
        << "\n"
        << "  " << prog << " brio_wu mixed_glm\n"
        << "      Run Brio-Wu shock tube with mixed GLM only.\n"
        << "\n"
        << "  " << prog << " orszag_tang none mixed_glm hyperbolic_glm\n"
        << "      Run Orszag-Tang with three selected cleaning methods.\n";
}

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

int main(int argc, char* argv[]) {

    // -------------------------------------------------------------------------
    // Help flag
    // -------------------------------------------------------------------------
    if (argc >= 2) {
        const std::string first = argv[1];
        if (first == "--help" || first == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // -------------------------------------------------------------------------
    // Problem selection  (argv[1], optional)
    // -------------------------------------------------------------------------
    using ProblemSpec = std::pair<std::string, std::string>;  // (problem, prefix)
    std::vector<ProblemSpec> problems;

    if (argc < 2) {
        // No arguments → run both problems.
        problems = {{"orszag_tang", "mhd_ot"}, {"brio_wu", "mhd_bw"}};
    } else {
        const std::string p = argv[1];
        if (p == "orszag_tang") {
            problems = {{"orszag_tang", "mhd_ot"}};
        } else if (p == "brio_wu") {
            problems = {{"brio_wu", "mhd_bw"}};
        } else {
            std::cerr << "Unknown problem: \"" << p << "\"\n\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // Cleaning method selection  (argv[2..], optional)
    // -------------------------------------------------------------------------
    std::vector<CleaningType> cases;

    if (argc < 3) {
        // No cleaning arguments → run all methods.
        cases = ALL_CASES;
    } else {
        for (int i = 2; i < argc; ++i) {
            try {
                cases.push_back(parse_cleaning_type_2d(argv[i]));
            } catch (const std::invalid_argument& e) {
                std::cerr << e.what() << "\n\n";
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Run
    // -------------------------------------------------------------------------
    for (const auto& [problem, prefix] : problems) {
        double gamma, t_end;
        if (problem == "orszag_tang") {
            std::cout << "\n#### Orszag-Tang vortex ####\n";
            gamma = 5.0 / 3.0;
            t_end = 0.5;
        } else {
            std::cout << "\n#### Brio-Wu shock tube (2D strip) ####\n";
            gamma = 2.0;
            t_end = 0.2;
        }
        run_problem(problem, prefix, gamma, t_end, 128, cases);
    }

    std::cout << "\nAll selected MHD + cleaning cases finished.\n";
    return 0;
}
