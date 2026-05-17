// =============================================================================
//  src/main.cpp
//  Test driver for the HLLD Riemann Solver
//
//  Build (from project root via CMake):
//    cmake -B build && cmake --build build --target brio_wu
//    ./build/brio_wu
// =============================================================================

#include <iostream>
#include <string>

// Defined in tests/Brio-Wu_Shock_Tube.cpp
void run_all_tests(double gamma_brio_wu, double gamma_general);

int main()
{
    const std::string sep(52, '=');

    std::cout << sep << "\n";
    std::cout << " 2D Ideal MHD -- HLLD Riemann Solver Test Suite\n";
    std::cout << " Solver  : Miyoshi & Kusano (JCP 2005)\n";
    std::cout << " Problems: Brio-Wu Shock Tube (Brio & Wu 1988)\n";
    std::cout << sep << "\n";

    run_all_tests(/*gamma_brio_wu=*/2.0,
                  /*gamma_general=*/5.0/3.0);

    std::cout << "\n" << sep << "\n";
    std::cout << " All tests completed.\n";
    std::cout << sep << "\n";

    return 0;
}
