cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

option(ENABLE_MPI "Build with MPI: coarse-grained sweep runner + solver domain decomposition" OFF)

# MPI is used in two independent ways when ENABLE_MPI is ON:
#   1. coarse-grained sweep over independent cases (src/mhd_sweep_mpi.cpp)
#   2. fine-grained 2D domain decomposition inside the runner
#      (src/mpi_domain.cpp, gated per-translation-unit by -DENABLE_MPI)
# Without ENABLE_MPI the runner falls back to the inline no-op stubs in
# include/mpi_domain.hpp and is bit-for-bit identical to the serial build.
if(ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
endif()

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

# Domain-decomposition layer.  Compiled into glm2d (the lowest library shared by
# the cleaning kernels and the runner) so its symbols are available to every
# dependent target.  ENABLE_MPI is PUBLIC so dependents see the same header view.
if(ENABLE_MPI)
    target_sources(glm2d PRIVATE src/mpi_domain.cpp)
    target_compile_definitions(glm2d PUBLIC ENABLE_MPI)
    target_link_libraries(glm2d PUBLIC MPI::MPI_CXX)
endif()

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
# (Orszag-Tang vortex, field-loop advection, divergence advection,
#  and MHD blast wave)
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
    # Coarse-grained MPI: one independent whole-grid case per rank.
    add_executable(mhd_sweep_mpi
        src/mhd_sweep_mpi.cpp
    )

    target_link_libraries(mhd_sweep_mpi
        PRIVATE
            mhd_runner
            MPI::MPI_CXX
    )

    # Fine-grained MPI: a single simulation split across ranks (2D domain
    # decomposition).  run_mhd_2d_case detects the multi-rank communicator and
    # decomposes internally; this is just the MPI_Init/Finalize entry point.
    add_executable(mhd_runner_mpi
        src/mhd_runner_mpi.cpp
    )

    target_link_libraries(mhd_runner_mpi
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
            ${CMAKE_SOURCE_DIR}/scripts/check/check_mhd_runner_sanity.py
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
            ${CMAKE_SOURCE_DIR}/scripts/check/check_mhd_runner_sanity.py
            --prefix mhd_da_smoke
            --methods none mixed_glm elliptic_projection
            --check-reduction
)

set_tests_properties(
    mhd_runner_divergence_reduction_check
    PROPERTIES DEPENDS mhd_runner_divergence_reduction_generate
)

add_test(
    NAME mhd_runner_blast_wave_generate
    COMMAND test_mhd_runner blast_wave
            none parabolic elliptic_projection
)

add_test(
    NAME mhd_runner_blast_wave_check
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/check/check_mhd_runner_sanity.py
            --prefix mhd_blast
            --methods none parabolic elliptic_projection
)

set_tests_properties(
    mhd_runner_blast_wave_check
    PROPERTIES DEPENDS mhd_runner_blast_wave_generate
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
                ${CMAKE_SOURCE_DIR}/scripts/check/check_mpi_sweep_summary.py
                --summary-dir results/mhd_sweep_mpi/summaries
                --min-rows 8
    )

    set_tests_properties(
        mhd_sweep_mpi_smoke_check
        PROPERTIES DEPENDS mhd_sweep_mpi_smoke_generate
    )

    # Domain decomposition: a single 32x32 orszag_tang run split over a 2x2
    # process grid (16x16 per rank), then the usual runner sanity check on the
    # gathered summaries.  Exercises halos, collective reductions, and gather I/O.
    add_test(
        NAME mhd_runner_mpi_smoke_generate
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:mhd_runner_mpi>
                ${MPIEXEC_POSTFLAGS}
                --smoke --output-prefix mpidd orszag_tang none mixed_glm
    )

    add_test(
        NAME mhd_runner_mpi_smoke_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/check/check_mhd_runner_sanity.py
                --prefix mpidd_smoke
                --methods none mixed_glm
    )

    set_tests_properties(
        mhd_runner_mpi_smoke_check
        PROPERTIES DEPENDS mhd_runner_mpi_smoke_generate
    )

    add_test(
        NAME mhd_runner_mpi_projection_generate
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:mhd_runner_mpi>
                ${MPIEXEC_POSTFLAGS}
                --smoke --output-prefix mpidd_projection
                divergence_advection elliptic_projection
    )

    add_test(
        NAME mhd_runner_mpi_projection_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/scripts/check/check_mhd_runner_sanity.py
                --prefix mpidd_projection_smoke
                --methods elliptic_projection
                --expected-nx 32 --expected-ny 32
    )

    set_tests_properties(
        mhd_runner_mpi_projection_check
        PROPERTIES DEPENDS mhd_runner_mpi_projection_generate
    )
endif()
