cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

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

# ============================================================
# 1D MHD with GLM divergence cleaning (A/B test executable)
# ============================================================

add_executable(main_glm
    src/main_glm.cpp
)

target_link_libraries(main_glm
    glm
)