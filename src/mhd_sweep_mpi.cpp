#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm.hpp"
#include "glm2d.hpp"
#include "mhd_runner.hpp"

namespace fs = std::filesystem;

namespace {

struct SweepOptions {
    bool smoke = false;
    std::string problem = "orszag_tang";
    int nx = 32;
    int ny = 32;
    double final_time = 0.02;
    double cfl = 0.4;
    std::string prefix = "mpi_sweep";
    std::vector<std::string> methods = {
        "none",
        "hyperbolic_glm",
        "mixed_glm",
        "parabolic",
        "elliptic_projection",
        "powell_source",
        "eglm",
        "gi_mixed_eglm"
    };
};

struct SweepJob {
    std::string method;
    CleaningType type = CleaningType::NONE;
    int nx = 0;
    int ny = 0;
    double final_time = 0.0;
    double cfl = 0.0;
    std::string output_prefix;
};

struct CaseSummary {
    std::string method;
    int nx = 0;
    int ny = 0;
    int rank = 0;
    double runtime_sec = 0.0;
    std::string status = "unknown";
    double final_divB_L2 = std::numeric_limits<double>::quiet_NaN();
    double final_divB_Linf = std::numeric_limits<double>::quiet_NaN();
    double min_pressure = std::numeric_limits<double>::quiet_NaN();
    double energy_drift = std::numeric_limits<double>::quiet_NaN();
};

std::vector<std::string> split(const std::string& text, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

void print_usage(const char* prog, int rank) {
    if (rank != 0) {
        return;
    }
    std::cerr
        << "Usage: " << prog << " [--smoke] [--problem orszag_tang|field_loop|divergence_advection]\n"
        << "       [--nx N] [--ny N] [--t-end T] [--cfl CFL]\n"
        << "       [--prefix PREFIX] [--methods comma,separated,list]\n\n"
        << "This is coarse-grained MPI over independent mhd_runner cases, not\n"
        << "domain-decomposed MPI inside the solver.\n";
}

int parse_int_arg(int argc, char* argv[], int& i, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
    }
    return std::stoi(argv[++i]);
}

double parse_double_arg(int argc, char* argv[], int& i, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
    }
    return std::stod(argv[++i]);
}

std::string parse_string_arg(int argc, char* argv[], int& i, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
    }
    return argv[++i];
}

SweepOptions parse_options(int argc, char* argv[], int rank) {
    SweepOptions opts;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0], rank);
            MPI_Finalize();
            std::exit(0);
        }
        if (arg == "--smoke") {
            opts.smoke = true;
            opts.nx = 16;
            opts.ny = 16;
            opts.final_time = 0.01;
            opts.cfl = 0.4;
            opts.prefix = "mpi_smoke";
            continue;
        }
        if (arg == "--problem") {
            opts.problem = parse_string_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--nx") {
            opts.nx = parse_int_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--ny") {
            opts.ny = parse_int_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--t-end") {
            opts.final_time = parse_double_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--cfl") {
            opts.cfl = parse_double_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--prefix") {
            opts.prefix = parse_string_arg(argc, argv, i, arg);
            continue;
        }
        if (arg == "--methods") {
            opts.methods = split(parse_string_arg(argc, argv, i, arg), ',');
            continue;
        }
        throw std::invalid_argument("unknown option: " + arg);
    }

    if (opts.nx < 4 || opts.ny < 4) {
        throw std::invalid_argument("nx and ny must both be >= 4");
    }
    if (opts.final_time <= 0.0) {
        throw std::invalid_argument("final time must be positive");
    }
    if (opts.cfl <= 0.0) {
        throw std::invalid_argument("CFL must be positive");
    }
    if (opts.methods.empty()) {
        throw std::invalid_argument("method list must not be empty");
    }

    return opts;
}

std::string sanitize_method_for_prefix(const std::string& method) {
    if (method == "eglm") {
        return "eglm";
    }
    return method;
}

std::vector<SweepJob> build_jobs(const SweepOptions& opts) {
    std::vector<SweepJob> jobs;
    for (const std::string& method : opts.methods) {
        SweepJob job;
        job.method = method;
        job.type = parse_cleaning_type_2d(method);
        job.nx = opts.nx;
        job.ny = opts.ny;
        job.final_time = opts.final_time;
        job.cfl = opts.cfl;
        job.output_prefix =
            opts.prefix + "_" + sanitize_method_for_prefix(method)
          + "_" + std::to_string(opts.nx) + "x" + std::to_string(opts.ny);
        jobs.push_back(job);
    }
    return jobs;
}

std::vector<std::string> csv_split_line(const std::string& line) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(line);
    while (std::getline(ss, item, ',')) {
        out.push_back(item);
    }
    if (!line.empty() && line.back() == ',') {
        out.emplace_back();
    }
    return out;
}

double parse_optional_double(const std::map<std::string, std::string>& row,
                             const std::string& key) {
    const auto it = row.find(key);
    if (it == row.end() || it->second.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::stod(it->second);
}

std::map<std::string, std::string> read_last_summary_row(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("missing runner summary: " + path.string());
    }

    std::string header_line;
    if (!std::getline(in, header_line)) {
        throw std::runtime_error("empty runner summary: " + path.string());
    }
    const std::vector<std::string> headers = csv_split_line(header_line);

    std::string line;
    std::string last;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            last = line;
        }
    }
    if (last.empty()) {
        throw std::runtime_error("runner summary has no data row: " + path.string());
    }

    const std::vector<std::string> values = csv_split_line(last);
    std::map<std::string, std::string> row;
    for (std::size_t i = 0; i < headers.size(); ++i) {
        row[headers[i]] = (i < values.size()) ? values[i] : "";
    }
    return row;
}

CaseSummary run_job(const SweepJob& job, const SweepOptions& opts, int rank) {
    CaseSummary summary;
    summary.method = job.method;
    summary.nx = job.nx;
    summary.ny = job.ny;
    summary.rank = rank;

    const auto start = std::chrono::steady_clock::now();

    try {
        MHDRunParams params;
        params.problem = opts.problem;
        params.gamma = 5.0 / 3.0;
        params.glm.nx = job.nx;
        params.glm.ny = job.ny;
        params.glm.xlen = 1.0;
        params.glm.ylen = 1.0;
        params.glm.t_end = job.final_time;
        params.glm.cfl = job.cfl;
        params.glm.cp = 0.2;
        params.glm.write_snapshot = false;
        params.glm.write_initial_snapshot = false;
        params.glm.out_prefix = job.output_prefix;

        if (job.type == CleaningType::PARABOLIC ||
            job.type == CleaningType::ELLIPTIC_PROJECTION) {
            params.glm.energy_policy =
                CleaningEnergyPolicy::PreserveThermalPressure;
        }

        run_mhd_2d_case(job.type, params);

        const fs::path runner_summary =
            fs::path("results/mhd_runner/summaries")
          / (job.output_prefix + "_" + cleaning_name(job.type) + "_summary.csv");
        const std::map<std::string, std::string> row =
            read_last_summary_row(runner_summary);

        const auto status_it = row.find("status");
        summary.status =
            (status_it != row.end() && !status_it->second.empty())
            ? status_it->second
            : "finished";
        summary.final_divB_L2 = parse_optional_double(row, "final_L2_fv");
        summary.final_divB_Linf = parse_optional_double(row, "final_Linf_fv");
        summary.min_pressure = parse_optional_double(row, "min_pressure");
        summary.energy_drift = parse_optional_double(row, "energy_drift");
    } catch (const std::exception& exc) {
        summary.status = std::string("exception:") + exc.what();
    }

    const auto stop = std::chrono::steady_clock::now();
    summary.runtime_sec =
        std::chrono::duration<double>(stop - start).count();
    return summary;
}

fs::path rank_summary_path(int rank) {
    std::ostringstream name;
    name << "summary_rank_" << std::setw(4) << std::setfill('0') << rank << ".csv";
    return fs::path("results/mhd_sweep_mpi/summaries") / name.str();
}

void write_rank_summary(const std::vector<CaseSummary>& rows, int rank) {
    fs::create_directories("results/mhd_sweep_mpi/summaries");
    const fs::path path = rank_summary_path(rank);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open MPI sweep summary: " + path.string());
    }

    out << "method,nx,ny,rank,runtime_sec,status,"
        << "final_divB_L2,final_divB_Linf,min_pressure,energy_drift\n";

    for (const CaseSummary& row : rows) {
        out << row.method << ","
            << row.nx << ","
            << row.ny << ","
            << row.rank << ","
            << row.runtime_sec << ","
            << row.status << ","
            << row.final_divB_L2 << ","
            << row.final_divB_Linf << ","
            << row.min_pressure << ","
            << row.energy_drift << "\n";
    }
}

} // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;

    try {
        const SweepOptions opts = parse_options(argc, argv, rank);
        const std::vector<SweepJob> jobs = build_jobs(opts);

        if (rank == 0) {
            fs::create_directories("results/mhd_sweep_mpi/summaries");
            std::cout << "MPI sweep: " << jobs.size()
                      << " independent jobs across " << size
                      << " ranks\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<CaseSummary> local_rows;
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            if (static_cast<int>(j % static_cast<std::size_t>(size)) != rank) {
                continue;
            }
            std::cout << "[rank " << rank << "] job " << j
                      << " method=" << jobs[j].method
                      << " nx=" << jobs[j].nx
                      << " ny=" << jobs[j].ny << "\n";
            local_rows.push_back(run_job(jobs[j], opts, rank));
        }

        write_rank_summary(local_rows, rank);
    } catch (const std::exception& exc) {
        if (rank == 0) {
            std::cerr << "mhd_sweep_mpi error: " << exc.what() << "\n";
            print_usage(argv[0], rank);
        }
        exit_code = 1;
    }

    int global_exit = 0;
    MPI_Allreduce(&exit_code, &global_exit, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_exit;
}
