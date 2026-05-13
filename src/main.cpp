// =============================================================================
//  src/main.cpp
//  Test driver for the HLLD Riemann Solver
//
//  Directory layout expected at compile time:
//    include/HLLD_mhd_solver.hpp   -- solver implementation (header-only)
//    tests/Brio-Wu_Shock_Tube.cpp   -- test problem definitions
//    src/main.cpp                  -- this file
//
//  Build (from the project root):
//    g++ -std=c++17 -O2 -Wall -Wextra -I. src/main.cpp -o hlld_test
//
//  Run:
//    ./hlld_test
// =============================================================================

#include "../include/HLLD_mhd_solver.hpp"   // solver (header-only, namespace MHD)
#include "../tests/Brio-Wu_Shock_Tube.cpp"   // test cases, declares run_all_tests()

#include <iostream>
#include <string>

// -----------------------------------------------------------------------------
int main()
{
    const std::string sep(52, '=');

    std::cout << sep << "\n";
    std::cout << " 2D Ideal MHD -- HLLD Riemann Solver Test Suite\n";
    std::cout << " Solver  : Miyoshi & Kusano (JCP 2005)\n";
    std::cout << " Problems: Brio-Wu Shock Tube (Brio & Wu 1988)\n";
    std::cout << sep << "\n";

    // Run all five Brio-Wu test cases.
    // gamma_brio_wu = 2.0  (standard Brio-Wu setting)
    // gamma_general = 5/3  (used for Cases 3-5)
    run_all_tests(/*gamma_brio_wu=*/2.0,
                  /*gamma_general=*/5.0/3.0);

    std::cout << "\n" << sep << "\n";
    std::cout << " All tests completed.\n";
    std::cout << sep << "\n";

    return 0;
}