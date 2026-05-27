cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

# ============================================================
# HLLD solver library
# ============================================================

add_library(hlld
    src/hlld_solver.cpp
)

add_executable(test_hlld_primitive_recovery
    tests/test_hlld_primitive_recovery.cpp
)

target_link_libraries(test_hlld_primitive_recovery
    PRIVATE
        hlld
)

# ============================================================
# 1D GLM core library
# ============================================================

add_library(glm
    src/glm.cpp
)

# ============================================================
# Diagnostics library
# ============================================================

add_library(diagnostics
    include/diagnostics.hpp
    src/diagnostics.cpp
)

# ============================================================
# Divergence-cleaning plugin library
# ============================================================

add_library(cleaning_plugins
    include/cleaning_plugin.hpp
    include/cleaning_plugin_utils.hpp
    src/cleaning_plugin.cpp
    src/cleaning_plugin_utils.cpp
)

add_executable(test_cleaning_plugins
    tests/test_cleaning_plugins.cpp
)

target_link_libraries(test_cleaning_plugins
    PRIVATE
        cleaning_plugins
        hlld
)

# ============================================================
# 2D GLM library
# ============================================================

add_library(glm2d
    src/glm2d.cpp
    src/glm2d_common.cpp
    src/hyperbolic_glm2d.cpp
    src/mixed_glm2d.cpp
    src/parabolic2d.cpp
    src/powell2d.cpp
    src/projection2d.cpp
    src/eglm2d.cpp
    src/galilean_glm2d.cpp
)

target_link_libraries(glm2d
    PUBLIC
        glm
        diagnostics
        OpenMP::OpenMP_CXX
)

# ============================================================
# 1D GLM test executable
# ============================================================

add_executable(test_glm_1d
    tests/test_glm_1d.cpp
)

target_link_libraries(test_glm_1d
    PRIVATE
        glm
)

# ============================================================
# 2D GLM test executable
# ============================================================

add_executable(test_glm_2d
    tests/test_glm_2d.cpp
)

target_link_libraries(test_glm_2d
    PRIVATE
        glm2d
)

# ============================================================
# Integrated HLLD + GLM runner library
# ============================================================

add_library(mhd_runner
    src/mhd_runner.cpp
)

target_link_libraries(mhd_runner
    PUBLIC
        hlld
        glm2d
)

# ============================================================
# Integrated HLLD + GLM runner test executable
# (Orszag-Tang vortex and Brio-Wu shock tube with all cleaning methods)
# ============================================================

add_executable(test_mhd_runner
    tests/test_mhd_runner.cpp
)

target_link_libraries(test_mhd_runner
    PRIVATE
        mhd_runner
)

# ============================================================
# CTest registration
# ============================================================

enable_testing()

add_test(
    NAME hlld_primitive_recovery
    COMMAND test_hlld_primitive_recovery
)

add_test(
    NAME cleaning_plugins
    COMMAND test_cleaning_plugins
)

add_test(
    NAME glm_1d
    COMMAND test_glm_1d
)

add_test(
    NAME glm_2d
    COMMAND test_glm_2d
)

add_test(
    NAME mhd_runner
    COMMAND test_mhd_runner
)
