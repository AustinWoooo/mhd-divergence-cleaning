#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "glm.hpp"
#include "state.hpp"

void initialize(std::vector<State>& U, double dx) {
    const int N = static_cast<int>(U.size());

    for (int i = 0; i < N; ++i) {
        const double x = (i + 0.5) * dx;

        for (int v = 0; v < NVAR; ++v) {
            U[i][v] = 0.0;
        }

        U[i][RHO] = 1.0;
        U[i][E]   = 1.0;

        // A localized nonzero Bx bump.
        // In 1D this creates nonzero divB = dBx/dx.
        U[i][BX] = 1.0 + 0.1 * std::exp(-200.0 * (x - 0.5) * (x - 0.5));
        U[i][PSI] = 0.0;
    }
}

void run_case(CleaningType type) {
    const int N = 400;
    const double xlen = 1.0;
    const double dx = xlen / N;

    GLMParams params;
    params.type = type;
    params.dx = dx;
    params.ch = 1.0;
    params.cp = 0.2;

    const double dt_hyp  = 0.4 * dx / params.ch;
    const double dt_para = 0.4 * dx * dx / (2.0 * params.cp * params.cp);

    if (type == CleaningType::PARABOLIC) {
        params.dt = dt_para;
    } else {
        params.dt = dt_hyp;
    }

    const double t_end = 0.5;
    const int nstep = static_cast<int>(std::ceil(t_end / params.dt));

    // Adjust dt so every method ends exactly at t_end.
    // Since we used ceil, this makes dt slightly smaller, so stability is preserved.
    params.dt = t_end / nstep;

    std::vector<State> U(N);
    initialize(U, dx);

    const std::string name = cleaning_name(type);
    const std::string outname = "results/glm_1d/glm_1d_" + name + ".csv";

    std::ofstream fout(outname);
    fout << "step,time,L1,L2,Linf\n";

    for (int step = 0; step <= nstep; ++step) {
        const double time = step * params.dt;

        double L1, L2, Linf;
        compute_divB_norms_1d(U, dx, L1, L2, Linf);

        fout << step << "," << time << "," << L1 << "," << L2 << "," << Linf << "\n";

        if (step == nstep) break;

        if (type == CleaningType::NONE) {
            // no update
        }
        else if (type == CleaningType::HYPERBOLIC_GLM) {
            update_hyperbolic_glm_1d(U, params);
        }
        else if (type == CleaningType::MIXED_GLM) {
            update_hyperbolic_glm_1d(U, params);
            apply_mixed_glm_damping(U, params);
        }
        else if (type == CleaningType::PARABOLIC) {
            apply_parabolic_cleaning_1d(U, params);
        }
    }

    std::cout << "Wrote " << outname << "\n";
}

int main() {
    run_case(CleaningType::NONE);
    run_case(CleaningType::HYPERBOLIC_GLM);
    run_case(CleaningType::MIXED_GLM);
    run_case(CleaningType::PARABOLIC);

    return 0;
}