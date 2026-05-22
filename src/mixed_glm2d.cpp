#include "mixed_glm2d.hpp"

#include <cmath>
#include <vector>

#include "hyperbolic_glm2d.hpp"

void apply_mixed_glm_damping_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    const double damping_rate =
        params.ch * params.ch / (params.cp * params.cp);

    const double factor = std::exp(-params.dt * damping_rate);

    for (auto& cell : U) {
        cell[PSI] *= factor;
    }
}

void update_mixed_glm_2d(
    std::vector<State>& U,
    const GLM2DParams& params
) {
    update_hyperbolic_glm_2d(U, params);
    apply_mixed_glm_damping_2d(U, params);
}
