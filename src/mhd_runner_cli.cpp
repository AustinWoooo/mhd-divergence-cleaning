#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_types.hpp"
#include "mhd_runner.hpp"

namespace {

const std::vector<std::string> PROBLEMS = {
    "orszag_tang",
    "field_loop",
    "divergence_advection",
    "blast_wave"
};

const std::vector<std::string> METHODS = {
    "none",
    "hyperbolic_glm",
    "mixed_glm",
    "parabolic",
    "elliptic_projection",
    "powell_source",
    "powell_source_subcycled",
    "powell_source_limited",
    "eglm",
    "mixed_eglm",
    "gi_mixed_eglm"
};

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
        << " [--smoke] [--performance-mode] [--no-snapshots]"
        << " [--diagnostic-stride N] [--output-root PATH]"
        << " [--glm-ch-factor X] [--glm-cd X | --glm-cr X]"
        << " [--glm-subcycles N]"
        << " [--preserve-thermal-pressure] [--project-each-stage]"
        << " [--reconstruction pcm|plm] [--limiter minmod|vanleer|mc]"
        << " [--nx N] [--ny N] [--tfinal T] [--output-prefix PREFIX]"
        << " [problem] [cleaning...]\n\n"
        << "Problems: orszag_tang | field_loop | divergence_advection | blast_wave\n"
        << "Cleaning methods: none | hyperbolic_glm | mixed_glm | parabolic | "
        << "elliptic_projection | powell_source | powell_source_subcycled | "
        << "powell_source_limited | eglm | mixed_eglm | gi_mixed_eglm\n\n"
        << "Benchmark controls:\n"
        << "  --performance-mode      Disable snapshots and use sparse diagnostics unless overridden.\n"
        << "  --no-snapshots          Disable initial/final snapshot CSV output.\n"
        << "  --diagnostic-stride N   Write time-history diagnostics every N accepted steps.\n"
        << "  --output-root PATH      Root for runner CSV outputs (default: results/mhd_runner).\n"
        << "  --glm-ch-factor X       Scale ch from the initial max signal speed (default: 1).\n"
        << "  --glm-cd X              Retained psi fraction per mixed-GLM cleaning substep, 0<X<1.\n"
        << "  --glm-cr X              Dedner-style cr=cp^2/ch for mixed-GLM damping.\n"
        << "  --glm-subcycles N       Minimum GLM cleaning subcycles per MHD step.\n"
        << "  --list-problems         Print supported MHD problems.\n"
        << "  --list-methods          Print supported cleaning methods.\n";
}

void print_list(const std::vector<std::string>& values) {
    for (const std::string& value : values) {
        std::cout << value << "\n";
    }
}

bool is_problem_name(const std::string& value) {
    return std::find(PROBLEMS.begin(), PROBLEMS.end(), value) != PROBLEMS.end();
}

int parse_positive_int(const char* text, const std::string& option) {
    const int value = std::atoi(text);
    if (value <= 0) {
        throw std::invalid_argument(option + " must be positive");
    }
    return value;
}

double parse_positive_double(const char* text, const std::string& option) {
    const double value = std::atof(text);
    if (value <= 0.0) {
        throw std::invalid_argument(option + " must be positive");
    }
    return value;
}

std::string require_value(int argc, char* argv[], int& i, const std::string& option) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(option + " requires an argument");
    }
    return argv[++i];
}

std::string default_prefix_for_problem(const std::string& problem) {
    if (problem == "orszag_tang") return "mhd_ot";
    if (problem == "field_loop") return "mhd_fl";
    if (problem == "divergence_advection") return "mhd_da";
    if (problem == "blast_wave") return "mhd_blast";
    throw std::invalid_argument("unknown problem: " + problem);
}

double default_tfinal_for_problem(const std::string& problem) {
    // Stop the blast before its fast shock reaches the periodic boundary.
    if (problem == "blast_wave") return 0.2;
    return 0.5;
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
    bool project_each_stage,
    MHD::Reconstruction reconstruction,
    MHD::SlopeLimiter limiter,
    int nx_override,
    int ny_override,
    double t_end_override,
    bool has_t_end_override,
    const std::string& output_prefix_override,
    bool no_snapshots,
    bool performance_mode,
    int diagnostic_stride,
    bool diagnostic_stride_explicit,
    double glm_ch_factor,
    double glm_cd,
    double glm_cr,
    int glm_subcycles,
    const std::string& output_root
) {
    MHDRunParams params;
    params.problem = problem;
    params.gamma = gamma;
    params.reconstruction = reconstruction;
    params.limiter = limiter;
    params.output_root = output_root;
    params.performance_mode = performance_mode;

    params.glm.nx = smoke ? 32 : (nx_override > 0 ? nx_override : N);
    params.glm.ny = smoke ? 32 : (ny_override > 0 ? ny_override : N);
    params.glm.xlen = 1.0;
    params.glm.ylen = 1.0;
    const double selected_t_end =
        has_t_end_override ? t_end_override : t_end;
    params.glm.t_end = smoke ? std::min(selected_t_end, 0.02) : selected_t_end;
    params.glm.cfl = 0.4;
    params.glm.cp = 0.2;
    params.glm.glm_ch_factor = glm_ch_factor;
    params.glm.glm_cd = glm_cd;
    params.glm.glm_cr = glm_cr;
    params.glm.glm_subcycles = glm_subcycles;
    params.glm.write_snapshot = !smoke;
    params.glm.write_initial_snapshot = false;
    params.glm.project_each_stage = project_each_stage;

    if (performance_mode) {
        params.glm.write_snapshot = false;
        params.diagnostic_stride = diagnostic_stride_explicit
            ? diagnostic_stride
            : 100;
    } else {
        params.diagnostic_stride = diagnostic_stride;
    }
    if (no_snapshots) {
        params.glm.write_snapshot = false;
        params.glm.write_initial_snapshot = false;
    }

    const std::string base_prefix =
        output_prefix_override.empty() ? prefix : output_prefix_override;
    params.glm.out_prefix = smoke ? base_prefix + "_smoke" : base_prefix;

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
                  << cleaning_name(type);
        if (performance_mode) {
            std::cout << " | performance-mode";
        }
        std::cout << "\n";
        std::cout << "========================================\n";

        run_mhd_2d_case(type, method_params);
    }
}

} // namespace

int main(int argc, char* argv[]) {
    bool smoke = false;
    bool no_snapshots = false;
    bool performance_mode = false;
    bool diagnostic_stride_explicit = false;
    bool energy_policy_explicit = false;
    bool project_each_stage = false;
    CleaningEnergyPolicy energy_policy =
        CleaningEnergyPolicy::ConserveTotalEnergy;
    MHD::Reconstruction reconstruction = MHD::Reconstruction::PLM;
    MHD::SlopeLimiter limiter = MHD::SlopeLimiter::MC;
    int nx_override = -1;
    int ny_override = -1;
    int diagnostic_stride = 1;
    double glm_ch_factor = 4.0;
    double glm_cd = std::numeric_limits<double>::quiet_NaN();
    double glm_cr = 0.1;
    int glm_subcycles = 1;
    double t_end_override = 0.0;
    bool has_t_end_override = false;
    std::string output_prefix_override;
    std::string output_root = "results/mhd_runner";

    std::vector<std::string> positional;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if (arg == "--list-problems") {
                print_list(PROBLEMS);
                return 0;
            }
            if (arg == "--list-methods") {
                print_list(METHODS);
                return 0;
            }
            if (arg == "--smoke") {
                smoke = true;
                continue;
            }
            if (arg == "--performance-mode") {
                performance_mode = true;
                no_snapshots = true;
                continue;
            }
            if (arg == "--no-snapshots") {
                no_snapshots = true;
                continue;
            }
            if (arg == "--diagnostic-stride") {
                diagnostic_stride =
                    parse_positive_int(
                        require_value(argc, argv, i, arg).c_str(),
                        arg
                    );
                diagnostic_stride_explicit = true;
                continue;
            }
            if (arg == "--glm-ch-factor") {
                glm_ch_factor =
                    parse_positive_double(
                        require_value(argc, argv, i, arg).c_str(),
                        arg
                    );
                continue;
            }
            if (arg == "--glm-cd") {
                glm_cd = std::atof(require_value(argc, argv, i, arg).c_str());
                if (!(glm_cd > 0.0 && glm_cd < 1.0)) {
                    throw std::invalid_argument("--glm-cd must satisfy 0 < value < 1");
                }
                continue;
            }
            if (arg == "--glm-cr") {
                glm_cr =
                    parse_positive_double(
                        require_value(argc, argv, i, arg).c_str(),
                        arg
                    );
                continue;
            }
            if (arg == "--glm-subcycles") {
                glm_subcycles =
                    parse_positive_int(
                        require_value(argc, argv, i, arg).c_str(),
                        arg
                    );
                continue;
            }
            if (arg == "--output-root") {
                output_root = require_value(argc, argv, i, arg);
                if (output_root.empty()) {
                    throw std::invalid_argument("--output-root must be non-empty");
                }
                continue;
            }
            if (arg == "--first-order") {
                reconstruction = MHD::Reconstruction::PCM;
                continue;
            }
            if (arg == "--reconstruction") {
                const std::string val = require_value(argc, argv, i, arg);
                if (val == "pcm") {
                    reconstruction = MHD::Reconstruction::PCM;
                } else if (val == "plm") {
                    reconstruction = MHD::Reconstruction::PLM;
                } else {
                    throw std::invalid_argument("unknown reconstruction: " + val);
                }
                continue;
            }
            if (arg == "--limiter") {
                const std::string val = require_value(argc, argv, i, arg);
                if (val == "minmod") {
                    limiter = MHD::SlopeLimiter::MINMOD;
                } else if (val == "vanleer") {
                    limiter = MHD::SlopeLimiter::VANLEER;
                } else if (val == "mc") {
                    limiter = MHD::SlopeLimiter::MC;
                } else {
                    throw std::invalid_argument("unknown limiter: " + val);
                }
                continue;
            }
            if (arg == "--nx") {
                nx_override =
                    parse_positive_int(require_value(argc, argv, i, arg).c_str(), arg);
                continue;
            }
            if (arg == "--ny") {
                ny_override =
                    parse_positive_int(require_value(argc, argv, i, arg).c_str(), arg);
                continue;
            }
            if (arg == "--tfinal" || arg == "--t-end") {
                t_end_override =
                    parse_positive_double(
                        require_value(argc, argv, i, arg).c_str(),
                        arg
                    );
                has_t_end_override = true;
                continue;
            }
            if (arg == "--output-prefix") {
                output_prefix_override = require_value(argc, argv, i, arg);
                if (output_prefix_override.empty()) {
                    throw std::invalid_argument("--output-prefix must be non-empty");
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
            if (arg == "--project-each-stage") {
                project_each_stage = true;
                continue;
            }
            positional.push_back(arg);
        }

        std::string problem = "orszag_tang";
        std::size_t method_start = 0;
        if (!positional.empty() && is_problem_name(positional[0])) {
            problem = positional[0];
            method_start = 1;
        } else if (!positional.empty()) {
            throw std::invalid_argument("unknown problem: " + positional[0]);
        }

        std::vector<CleaningType> cases;
        if (method_start >= positional.size()) {
            cases = ALL_CASES;
        } else {
            for (std::size_t i = method_start; i < positional.size(); ++i) {
                cases.push_back(parse_cleaning_type_2d(positional[i]));
            }
        }

        if (std::isfinite(glm_cd) && std::isfinite(glm_cr)) {
            throw std::invalid_argument("set only one of --glm-cd or --glm-cr");
        }

        run_problem(
            problem,
            default_prefix_for_problem(problem),
            5.0 / 3.0,
            default_tfinal_for_problem(problem),
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
            output_prefix_override,
            no_snapshots,
            performance_mode,
            diagnostic_stride,
            diagnostic_stride_explicit,
            glm_ch_factor,
            glm_cd,
            glm_cr,
            glm_subcycles,
            output_root
        );
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "\nAll selected MHD + cleaning cases finished.\n";
    return 0;
}
