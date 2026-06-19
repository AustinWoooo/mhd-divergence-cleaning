#include "glm2d.hpp"

#include "hyperbolic_glm2d.hpp"
#include "mixed_glm2d.hpp"
#include "parabolic2d.hpp"
#include "powell2d.hpp"
#include "projection2d.hpp"
#include "eglm2d.hpp"
#include "galilean_glm2d.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 2D divergence-cleaning dispatch.  The standalone divergence-pulse sandbox
// (run_glm_2d_case / write_glm_2d_snapshot / initialize_divergence_pulse_2d and
// their helpers) was removed together with tests/test_glm_2d.cpp, its only
// caller; the cleaning kernels themselves live in their own translation units
// and are driven by the integrated runner via advance_glm_2d_one_step().

CleaningType parse_cleaning_type_2d(const std::string& name) {
    if (name == "none") return CleaningType::NONE;
    if (name == "parabolic") return CleaningType::PARABOLIC;
    if (name == "hyperbolic_glm") return CleaningType::HYPERBOLIC_GLM;
    if (name == "mixed_glm") return CleaningType::MIXED_GLM;
    if (name == "elliptic_projection") return CleaningType::ELLIPTIC_PROJECTION;
    if (name == "powell_source") return CleaningType::POWELL_SOURCE;
    if (name == "eglm") return CleaningType::MIXED_EGLM;
    if (name == "mixed_eglm") return CleaningType::MIXED_EGLM;
    if (name == "gi_mixed_eglm") return CleaningType::GI_MIXED_EGLM;

    throw std::invalid_argument("Unknown cleaning type: " + name);
}

std::vector<CleaningType> selected_cleaning_cases_2d(
    const std::string& case_name
) {
    if (case_name == "all") {
        return {
            CleaningType::NONE,
            CleaningType::HYPERBOLIC_GLM,
            CleaningType::MIXED_GLM,
            CleaningType::PARABOLIC,
            CleaningType::ELLIPTIC_PROJECTION,
            CleaningType::POWELL_SOURCE,
            CleaningType::MIXED_EGLM,
            CleaningType::GI_MIXED_EGLM
        };
    }

    return {parse_cleaning_type_2d(case_name)};
}

double max_cleaning_dt(
    CleaningType method,
    double dx,
    double dy,
    const GLM2DParams& params
) {
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;

    if (method == CleaningType::PARABOLIC) {
        constexpr double Cdiff = 0.20;
        const double h = std::min(dx, dy);
        return Cdiff * h * h / (params.cp * params.cp);
    }

    if (method == CleaningType::HYPERBOLIC_GLM ||
        method == CleaningType::MIXED_GLM ||
        method == CleaningType::MIXED_EGLM ||
        method == CleaningType::GI_MIXED_EGLM) {
        return params.cfl / (params.ch * (inv_dx + inv_dy));
    }

    return std::numeric_limits<double>::infinity();
}

void advance_glm_2d_one_step(
    std::vector<State>& U,
    CleaningType type,
    const GLM2DParams& params
) {
    if (type == CleaningType::NONE) {
        return;
    }

    if (type == CleaningType::HYPERBOLIC_GLM) {
        update_hyperbolic_glm_2d(U, params);
        return;
    }

    if (type == CleaningType::MIXED_GLM) {
        update_mixed_glm_2d(U, params);
        return;
    }

    if (type == CleaningType::PARABOLIC) {
        apply_parabolic_cleaning_2d(U, params);
        return;
    }

    if (type == CleaningType::ELLIPTIC_PROJECTION) {
        apply_elliptic_projection_2d(U, params);
        return;
    }

    if (type == CleaningType::POWELL_SOURCE) {
        apply_powell_source_2d(U, params);
        return;
    }

    if (type == CleaningType::MIXED_EGLM) {
        update_mixed_eglm_2d(U, params);
        return;
    }

    if (type == CleaningType::GI_MIXED_EGLM) {
        update_gi_mixed_eglm_2d(U, params);
        return;
    }

    throw std::runtime_error("Unsupported cleaning type in 2D GLM advance.");
}
