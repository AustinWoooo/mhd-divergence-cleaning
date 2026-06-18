// =============================================================================
//  src/mhd_runner_mpi.cpp
//
//  MPI entry point for the domain-decomposed 2D MHD runner.  This is a thin
//  wrapper: it initialises MPI and calls the same run_mhd_2d_case() used by the
//  serial CLI.  That function detects the multi-rank communicator and builds a
//  2D Cartesian decomposition itself (each rank evolves its own ghost-padded
//  sub-block, exchanging halos every substep); rank 0 gathers for all I/O.
//
//  This differs from src/mhd_sweep_mpi.cpp, which is coarse-grained MPI over
//  independent whole-grid cases.  Here a SINGLE simulation is split across ranks
//  (true domain decomposition).
//
//  Build:   cmake -S . -B build-mpi -DENABLE_MPI=ON && cmake --build build-mpi -j
//  Run:     mpirun -np 4 ./build-mpi/mhd_runner_mpi --nx 256 --ny 256 \
//                 orszag_tang mixed_glm
//
//  The global nx/ny must be divisible by the MPI process-grid dimensions
//  (MPI_Dims_create chooses them); run_mhd_2d_case throws a clear error if not.
//  elliptic_projection is rejected on more than one rank (its global Poisson
//  solve is not decomposed); use np=1 or a local cleaning method.
// =============================================================================

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm.hpp"
#include "glm2d.hpp"
#include "glm2d_types.hpp"
#include "mhd_runner.hpp"

namespace {

const std::vector<std::string> PROBLEMS = {
    "orszag_tang", "field_loop", "divergence_advection", "blast_wave"
};

bool is_problem_name(const std::string& v) {
    return std::find(PROBLEMS.begin(), PROBLEMS.end(), v) != PROBLEMS.end();
}

void print_usage(const char* prog) {
    std::cerr
        << "Usage: mpirun -np P " << prog
        << " [--smoke] [--nx N] [--ny N] [--tfinal T] [--cfl C]\n"
        << "       [--no-snapshots] [--performance-mode] [--diagnostic-stride N]\n"
        << "       [--output-prefix PREFIX] [--output-root PATH]\n"
        << "       [--glm-ch-factor X] [--glm-subcycles N]\n"
        << "       [problem] [cleaning...]\n\n"
        << "Domain-decomposed single-simulation runner.  Problems: orszag_tang |\n"
        << "field_loop | divergence_advection | blast_wave.  Cleaning: none |\n"
        << "hyperbolic_glm | mixed_glm | parabolic | powell_source | eglm |\n"
        << "gi_mixed_eglm.  (elliptic_projection requires np=1.)\n";
}

std::string require_value(int argc, char* argv[], int& i, const std::string& opt) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(opt + " requires an argument");
    }
    return argv[++i];
}

int parse_pos_int(const std::string& s, const std::string& opt) {
    const int v = std::atoi(s.c_str());
    if (v <= 0) throw std::invalid_argument(opt + " must be positive");
    return v;
}

double parse_pos_double(const std::string& s, const std::string& opt) {
    const double v = std::atof(s.c_str());
    if (v <= 0.0) throw std::invalid_argument(opt + " must be positive");
    return v;
}

std::string default_prefix(const std::string& problem) {
    if (problem == "orszag_tang") return "mhd_ot";
    if (problem == "field_loop") return "mhd_fl";
    if (problem == "divergence_advection") return "mhd_da";
    if (problem == "blast_wave") return "mhd_blast";
    return "mhd";
}

double default_tfinal(const std::string& problem) {
    return problem == "blast_wave" ? 0.2 : 0.5;
}

struct Options {
    std::string problem = "orszag_tang";
    std::vector<CleaningType> cases;
    bool smoke = false;
    bool no_snapshots = false;
    bool performance_mode = false;
    int nx = 256;
    int ny = 256;
    bool has_tfinal = false;
    double tfinal = 0.0;
    double cfl = 0.4;
    int diagnostic_stride = 1;
    bool diagnostic_stride_explicit = false;
    double glm_ch_factor = 4.0;
    int glm_subcycles = 1;
    std::string output_prefix;
    std::string output_root = "results/mhd_runner";
};

Options parse_options(int argc, char* argv[]) {
    Options o;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--smoke") {
            o.smoke = true;
        } else if (arg == "--no-snapshots") {
            o.no_snapshots = true;
        } else if (arg == "--performance-mode") {
            o.performance_mode = true;
            o.no_snapshots = true;
        } else if (arg == "--nx") {
            o.nx = parse_pos_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--ny") {
            o.ny = parse_pos_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--tfinal" || arg == "--t-end") {
            o.tfinal = parse_pos_double(require_value(argc, argv, i, arg), arg);
            o.has_tfinal = true;
        } else if (arg == "--cfl") {
            o.cfl = parse_pos_double(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--diagnostic-stride") {
            o.diagnostic_stride =
                parse_pos_int(require_value(argc, argv, i, arg), arg);
            o.diagnostic_stride_explicit = true;
        } else if (arg == "--glm-ch-factor") {
            o.glm_ch_factor =
                parse_pos_double(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--glm-subcycles") {
            o.glm_subcycles =
                parse_pos_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--output-prefix") {
            o.output_prefix = require_value(argc, argv, i, arg);
        } else if (arg == "--output-root") {
            o.output_root = require_value(argc, argv, i, arg);
        } else if (arg == "--help" || arg == "-h") {
            throw std::invalid_argument("help");
        } else {
            positional.push_back(arg);
        }
    }

    std::size_t method_start = 0;
    if (!positional.empty() && is_problem_name(positional[0])) {
        o.problem = positional[0];
        method_start = 1;
    } else if (!positional.empty()) {
        throw std::invalid_argument("unknown problem: " + positional[0]);
    }

    if (method_start >= positional.size()) {
        o.cases = {CleaningType::NONE, CleaningType::MIXED_GLM};
    } else {
        for (std::size_t i = method_start; i < positional.size(); ++i) {
            o.cases.push_back(parse_cleaning_type_2d(positional[i]));
        }
    }

    if (o.smoke) {
        o.nx = 32;
        o.ny = 32;
        if (!o.has_tfinal) {
            o.tfinal = 0.02;
            o.has_tfinal = true;
        }
        o.no_snapshots = true;
    }
    return o;
}

MHDRunParams make_params(const Options& o, CleaningType type) {
    MHDRunParams params;
    params.problem = o.problem;
    params.gamma = 5.0 / 3.0;
    params.output_root = o.output_root;
    params.performance_mode = o.performance_mode;

    params.glm.nx = o.nx;
    params.glm.ny = o.ny;
    params.glm.xlen = 1.0;
    params.glm.ylen = 1.0;
    params.glm.t_end = o.has_tfinal ? o.tfinal : default_tfinal(o.problem);
    params.glm.cfl = o.cfl;
    params.glm.cp = 0.2;
    params.glm.glm_ch_factor = o.glm_ch_factor;
    params.glm.glm_subcycles = o.glm_subcycles;
    params.glm.write_snapshot = !o.no_snapshots && !o.smoke;
    params.glm.write_initial_snapshot = false;

    const std::string base =
        o.output_prefix.empty() ? default_prefix(o.problem) : o.output_prefix;
    params.glm.out_prefix = o.smoke ? base + "_smoke" : base;

    if (o.performance_mode) {
        params.diagnostic_stride =
            o.diagnostic_stride_explicit ? o.diagnostic_stride : 100;
    } else {
        params.diagnostic_stride = o.diagnostic_stride;
    }

    // Match the serial CLI's energy-policy default for the diffusive methods.
    if (type == CleaningType::PARABOLIC ||
        type == CleaningType::ELLIPTIC_PROJECTION) {
        params.glm.energy_policy = CleaningEnergyPolicy::PreserveThermalPressure;
    }
    return params;
}

}  // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;
    try {
        const Options opts = parse_options(argc, argv);

        if (rank == 0) {
            std::cout << "Domain-decomposed MHD runner: " << opts.problem
                      << "  nx=" << opts.nx << " ny=" << opts.ny
                      << "  ranks=" << size << "\n";
        }

        for (CleaningType type : opts.cases) {
            if (rank == 0) {
                std::cout << "---- cleaning = " << cleaning_name(type)
                          << " ----\n";
            }
            run_mhd_2d_case(type, make_params(opts, type));
        }
    } catch (const std::exception& exc) {
        if (rank == 0) {
            std::cerr << "mhd_runner_mpi error: " << exc.what() << "\n";
            print_usage(argv[0]);
        }
        exit_code = 1;
    }

    int global_exit = 0;
    MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit;
}
