// =============================================================================
//  tests/test_mhd_runner.cpp
//
//  Drives the integrated HLLD + GLM runner on several MHD problems:
//    1. Orszag-Tang vortex   (smooth, periodic, full 2D)
//    2. Field-loop advection (weak localized magnetic loop)
//    3. Divergence advection (controlled non-solenoidal perturbation)
//
//  Each problem is run with several cleaning methods so divB behaviour can be
//  compared in results/divergence/*.csv and results/snapshots/*_final.csv.
//
//  Usage:
//    ./test_mhd_runner                              # all problems, all cleanings
//    ./test_mhd_runner orszag_tang                  # one problem, all cleanings
//    ./test_mhd_runner field_loop mixed_glm         # one problem, one cleaning
//    ./test_mhd_runner orszag_tang none mixed_glm   # one problem, multiple cleanings
//
//  Cleaning type names (lowercase):
//    none, parabolic, hyperbolic_glm, mixed_glm, elliptic_projection,
//    powell_source, powell_source_subcycled, powell_source_limited,
//    eglm (alias for mixed_eglm), mixed_eglm, gi_mixed_eglm
// =============================================================================

#include <algorithm>
#include <cstdlib>
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
    CleaningType::POWELL_SOURCE_SUBCYCLED,
    CleaningType::POWELL_SOURCE_LIMITED,
    CleaningType::MIXED_EGLM,
    CleaningType::GI_MIXED_EGLM
};

void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog
        << " [--smoke] [--preserve-thermal-pressure] [--project-each-stage]"
        << " [--reconstruction pcm|plm] [--limiter minmod|vanleer|mc]"
        << " [--nx N] [--ny N] [--tfinal T] [--output-prefix PREFIX]"
        << " [--first-order]"
        << " [problem] [cleaning...]\n"
        << "\n"
        << "  --smoke   : run a fast low-resolution verification case.\n"
        << "  --reconstruction pcm|plm\n"
        << "             spatial reconstruction order for the HLLD update.\n"
        << "             pcm = first-order Godunov, plm = second-order MUSCL\n"
        << "             high-resolution shock capturing (default: plm).\n"
        << "  --first-order\n"
        << "             alias for --reconstruction pcm.\n"
        << "  --limiter minmod|vanleer|mc\n"
        << "             TVD slope limiter for plm reconstruction (default: mc).\n"
        << "  --nx N, --ny N\n"
        << "             grid resolution for non-smoke runs (default: problem N).\n"
        << "  --tfinal T, --t-end T\n"
        << "             final time override for the selected problem.\n"
        << "  --output-prefix PREFIX\n"
        << "             output filename prefix for the selected run.\n"
        << "  --preserve-thermal-pressure\n"
        << "             use explicit cleaning energy repair where supported.\n"
        << "  --project-each-stage\n"
        << "             (Task C) apply elliptic projection after each RK predictor\n"
        << "             stage in addition to the end-of-step projection.\n"
        << "             Only affects elliptic_projection runs.\n"
        << "  problem   : orszag_tang | field_loop\n"
        << "              divergence_advection  (default: orszag_tang)\n"
        << "  cleaning  : none | parabolic | hyperbolic_glm | mixed_glm\n"
        << "              elliptic_projection | powell_source\n"
        << "              powell_source_subcycled | powell_source_limited\n"
        << "              eglm | mixed_eglm | gi_mixed_eglm  (default: all)\n"
        << "\n"
        << "Examples:\n"
        << "  " << prog << "\n"
        << "      Run all problems with all cleaning methods.\n"
        << "\n"
        << "  " << prog << " orszag_tang\n"
        << "      Run Orszag-Tang vortex with all cleaning methods.\n"
        << "\n"
        << "  " << prog << " field_loop mixed_glm\n"
        << "      Run field-loop advection with mixed GLM only.\n"
        << "\n"
        << "  " << prog << " divergence_advection none hyperbolic_glm mixed_glm\n"
        << "      Run divergence advection with three selected cleaning methods.\n"
        << "\n"
        << "  " << prog << " orszag_tang none mixed_glm hyperbolic_glm\n"
        << "      Run Orszag-Tang with three selected cleaning methods.\n"
        << "\n"
        << "  " << prog << " orszag_tang elliptic_projection --project-each-stage\n"
        << "      Run Orszag-Tang with stage-wise projection (Task C).\n";
}

void run_problem(
    const std::string& problem,
    const std::string& prefix,
    double gamma,
    double t_end,
    int N,
    const std::vector<CleaningType>& cases,
    bool smoke,
    CleaningEnergyPolicy energy_policy,
    bool energy_policy_explicit,
    bool project_each_stage,   // Task C
    MHD::Reconstruction reconstruction,
    MHD::SlopeLimiter limiter,
    int nx_override,
    int ny_override,
    double t_end_override,
    bool has_t_end_override,
    const std::string& output_prefix_override
) {
    MHDRunParams params;
    params.problem = problem;
    params.gamma   = gamma;
    params.reconstruction = reconstruction;
    params.limiter        = limiter;

    params.glm.nx = smoke ? 32 : (nx_override > 0 ? nx_override : N);
    params.glm.ny = smoke ? 32 : (ny_override > 0 ? ny_override : N);
    params.glm.xlen = 1.0;
    params.glm.ylen = 1.0;
    const double selected_t_end =
        has_t_end_override ? t_end_override : t_end;
    params.glm.t_end = smoke ? std::min(selected_t_end, 0.02) : selected_t_end;
    params.glm.cfl   = 0.4;
    params.glm.cp    = 0.2;
    params.glm.write_snapshot = !smoke;
    params.glm.write_initial_snapshot = false;
    const std::string base_prefix =
        output_prefix_override.empty() ? prefix : output_prefix_override;
    params.glm.out_prefix = smoke ? base_prefix + "_smoke" : base_prefix;
    params.glm.project_each_stage = project_each_stage;

    for (CleaningType type : cases) {
        MHDRunParams method_params = params;
        method_params.glm.energy_policy = energy_policy;

        if (!energy_policy_explicit &&
            (type == CleaningType::PARABOLIC ||
             type == CleaningType::ELLIPTIC_PROJECTION)) {
            method_params.glm.energy_policy =
                CleaningEnergyPolicy::PreserveThermalPressure;
        }

        std::cout << "========================================\n";
        std::cout << " " << problem << " | cleaning = "
                  << static_cast<int>(type);
        if (project_each_stage &&
            type == CleaningType::ELLIPTIC_PROJECTION) {
            std::cout << " (project-each-stage)";
        }
        std::cout << "\n";
        std::cout << "========================================\n";
        run_mhd_2d_case(type, method_params);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    bool smoke = false;
    bool energy_policy_explicit = false;
    bool project_each_stage = false;  // Task C
    CleaningEnergyPolicy energy_policy =
        CleaningEnergyPolicy::ConserveTotalEnergy;
    MHD::Reconstruction reconstruction = MHD::Reconstruction::PLM;
    MHD::SlopeLimiter   limiter        = MHD::SlopeLimiter::MC;
    int nx_override = -1;
    int ny_override = -1;
    double t_end_override = 0.0;
    bool has_t_end_override = false;
    std::string output_prefix_override;

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--smoke") {
            smoke = true;
            continue;
        }
        if (arg == "--first-order") {
            reconstruction = MHD::Reconstruction::PCM;
            continue;
        }
        if (arg == "--reconstruction") {
            if (i + 1 >= argc) {
                std::cerr << "--reconstruction requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            const std::string val = argv[++i];
            if (val == "pcm") {
                reconstruction = MHD::Reconstruction::PCM;
            } else if (val == "plm") {
                reconstruction = MHD::Reconstruction::PLM;
            } else {
                std::cerr << "Unknown reconstruction: \"" << val << "\"\n\n";
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (arg == "--limiter") {
            if (i + 1 >= argc) {
                std::cerr << "--limiter requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            const std::string val = argv[++i];
            if (val == "minmod") {
                limiter = MHD::SlopeLimiter::MINMOD;
            } else if (val == "vanleer") {
                limiter = MHD::SlopeLimiter::VANLEER;
            } else if (val == "mc") {
                limiter = MHD::SlopeLimiter::MC;
            } else {
                std::cerr << "Unknown limiter: \"" << val << "\"\n\n";
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (arg == "--nx") {
            if (i + 1 >= argc) {
                std::cerr << "--nx requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            nx_override = std::atoi(argv[++i]);
            if (nx_override <= 0) {
                std::cerr << "--nx must be positive\n\n";
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (arg == "--ny") {
            if (i + 1 >= argc) {
                std::cerr << "--ny requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            ny_override = std::atoi(argv[++i]);
            if (ny_override <= 0) {
                std::cerr << "--ny must be positive\n\n";
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (arg == "--tfinal" || arg == "--t-end") {
            if (i + 1 >= argc) {
                std::cerr << arg << " requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            t_end_override = std::atof(argv[++i]);
            if (t_end_override <= 0.0) {
                std::cerr << arg << " must be positive\n\n";
                print_usage(argv[0]);
                return 1;
            }
            has_t_end_override = true;
            continue;
        }
        if (arg == "--output-prefix") {
            if (i + 1 >= argc) {
                std::cerr << "--output-prefix requires an argument\n\n";
                print_usage(argv[0]);
                return 1;
            }
            output_prefix_override = argv[++i];
            if (output_prefix_override.empty()) {
                std::cerr << "--output-prefix must be non-empty\n\n";
                print_usage(argv[0]);
                return 1;
            }
            continue;
        }
        if (arg == "--preserve-thermal-pressure") {
            energy_policy = CleaningEnergyPolicy::PreserveThermalPressure;
            energy_policy_explicit = true;
            continue;
        }
        if (arg == "--conserve-total-energy") {
            energy_policy = CleaningEnergyPolicy::ConserveTotalEnergy;
            energy_policy_explicit = true;
            continue;
        }
        // Task C: apply projection after each RK predictor stage.
        if (arg == "--project-each-stage") {
            project_each_stage = true;
            continue;
        }
        positional.push_back(arg);
    }

    // -------------------------------------------------------------------------
    // Problem selection  (first positional argument, optional)
    // -------------------------------------------------------------------------
    using ProblemSpec = std::pair<std::string, std::string>;  // (problem, prefix)
    std::vector<ProblemSpec> problems;

    if (positional.empty()) {
        // No arguments → run the default problem.
        problems = {{"orszag_tang", "mhd_ot"}};
    } else {
        const std::string p = positional[0];
        if (p == "orszag_tang") {
            problems = {{"orszag_tang", "mhd_ot"}};
        } else if (p == "field_loop") {
            problems = {{"field_loop", "mhd_fl"}};
        } else if (p == "divergence_advection") {
            problems = {{"divergence_advection", "mhd_da"}};
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

    if (positional.size() < 2) {
        // No cleaning arguments → run all methods.
        cases = ALL_CASES;
    } else {
        for (std::size_t i = 1; i < positional.size(); ++i) {
            try {
                cases.push_back(parse_cleaning_type_2d(positional[i]));
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
        } else if (problem == "field_loop") {
            std::cout << "\n#### Field-loop advection ####\n";
            gamma = 5.0 / 3.0;
            t_end = 0.5;
        } else if (problem == "divergence_advection") {
            std::cout << "\n#### Divergence advection ####\n";
            gamma = 5.0 / 3.0;
            t_end = 0.5;
        } else {
            throw std::runtime_error("Unhandled problem configuration: " + problem);
        }
        run_problem(
            problem,
            prefix,
            gamma,
            t_end,
            128,
            cases,
            smoke,
            energy_policy,
            energy_policy_explicit,
            project_each_stage,
            reconstruction,
            limiter,
            nx_override,
            ny_override,
            t_end_override,
            has_t_end_override,
            output_prefix_override
        );
    }

    std::cout << "\nAll selected MHD + cleaning cases finished.\n";
    return 0;
}
