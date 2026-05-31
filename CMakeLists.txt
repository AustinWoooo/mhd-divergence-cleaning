cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

option(ENABLE_MPI "Build optional coarse-grained MPI sweep runner" OFF)

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
# Reconstruction (HRSC MUSCL limiter) unit test
# ============================================================

add_executable(test_mhd_reconstruction
    tests/test_mhd_reconstruction.cpp
)

target_link_libraries(test_mhd_reconstruction
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
    src/powell_control.cpp
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
        OpenMP::OpenMP_CXX
)

# ============================================================
# Integrated HLLD + divergence-control runner test executable
# (Orszag-Tang vortex, field-loop advection, and divergence advection)
# ============================================================

add_executable(test_mhd_runner
    tests/test_mhd_runner.cpp
)

target_link_libraries(test_mhd_runner
    PRIVATE
        mhd_runner
)

add_executable(mhd_runner_cli
    src/mhd_runner_cli.cpp
)

target_link_libraries(mhd_runner_cli
    PRIVATE
        mhd_runner
)

if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)

    add_executable(mhd_sweep_mpi
        src/mhd_sweep_mpi.cpp
    )

    target_link_libraries(mhd_sweep_mpi
        PRIVATE
            mhd_runner
            MPI::MPI_CXX
    )
endif()

# ============================================================
# CTest registration
# ============================================================

enable_testing()

add_test(
    NAME hlld_primitive_recovery
    COMMAND test_hlld_primitive_recovery
)

add_test(
    NAME mhd_reconstruction
    COMMAND test_mhd_reconstruction
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

add_test(
    NAME mhd_runner_smoke_sanity_generate
    COMMAND test_mhd_runner --smoke orszag_tang
            none hyperbolic_glm mixed_glm parabolic elliptic_projection
            powell_source eglm gi_mixed_eglm
)

add_test(
    NAME mhd_runner_smoke_sanity_check
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/check_mhd_runner_sanity.py
            --prefix mhd_ot_smoke
)

set_tests_properties(
    mhd_runner_smoke_sanity_check
    PROPERTIES DEPENDS mhd_runner_smoke_sanity_generate
)

add_test(
    NAME mhd_runner_divergence_reduction_generate
    COMMAND test_mhd_runner --smoke divergence_advection
            none mixed_glm elliptic_projection
)

add_test(
    NAME mhd_runner_divergence_reduction_check
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/check_mhd_runner_sanity.py
            --prefix mhd_da_smoke
            --methods none mixed_glm elliptic_projection
            --check-reduction
)

set_tests_properties(
    mhd_runner_divergence_reduction_check
    PROPERTIES DEPENDS mhd_runner_divergence_reduction_generate
)

if(ENABLE_MPI)
    add_test(
        NAME mhd_sweep_mpi_smoke_generate
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:mhd_sweep_mpi>
                ${MPIEXEC_POSTFLAGS}
                --smoke
    )

    add_test(
        NAME mhd_sweep_mpi_smoke_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/check_mpi_sweep_summary.py
                --summary-dir results/mhd_sweep_mpi/summaries
                --min-rows 8
    )

    set_tests_properties(
        mhd_sweep_mpi_smoke_check
        PROPERTIES DEPENDS mhd_sweep_mpi_smoke_generate
    )
endif()
