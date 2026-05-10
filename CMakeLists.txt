cmake_minimum_required(VERSION 3.16)

project(mhd_divergence_cleaning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(glm
    src/glm.cpp
)

add_executable(test_glm_1d
    tests/test_glm_1d.cpp
)

target_link_libraries(test_glm_1d
    glm
)