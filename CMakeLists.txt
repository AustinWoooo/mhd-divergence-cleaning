cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# ============================================================
# HLLD solver library
# ============================================================

add_library(hlld
    src/hlld.cpp
)

# ============================================================
# Brio-Wu Shock Tube test executable
# ============================================================

add_executable(brio_wu
    src/main.cpp
    tests/Brio-Wu_Shock_Tube.cpp
)

target_link_libraries(brio_wu
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
# 2D GLM library
# ============================================================

add_library(glm2d
    include/glm2d.hpp
    src/glm2d.cpp
)

target_link_libraries(glm2d
    glm
    diagnostics
)

# ============================================================
# 1D GLM test executable
# ============================================================

add_executable(test_glm_1d
    tests/test_glm_1d.cpp
)

target_link_libraries(test_glm_1d
    glm
)

# ============================================================
# 2D GLM test executable
# ============================================================

add_executable(test_glm_2d
    tests/test_glm_2d.cpp
)

target_link_libraries(test_glm_2d
    glm2d
)
