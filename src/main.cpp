#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "mhd_solver.hpp"

// 輔助函式：用來漂亮地印出向量結果
void print_vector(const std::string& name, const std::vector<double>& vec) {
    std::cout << std::left << std::setw(15) << name << ": [ ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(5) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << " ]\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "   2D Ideal MHD HLL Riemann Solver Test (Brio-Wu)\n";
    std::cout << "========================================================\n\n";

    // 1. 設定 Brio-Wu Shock Tube 的初始狀態 (Left & Right States)
    // 傳統 Brio-Wu 問題的絕熱指數設為 2.0
    double gamma = 2.0;

    State W_L;
    W_L.rho = 1.0;     // 密度
    W_L.u   = 0.0;     // X 速度
    W_L.v   = 0.0;     // Y 速度
    W_L.w   = 0.0;     // Z 速度
    W_L.p   = 1.0;     // 壓力
    W_L.bx  = 0.75;    // X 磁場 (法向磁場，必須連續)
    W_L.by  = 1.0;     // Y 磁場
    W_L.bz  = 0.0;     // Z 磁場

    State W_R;
    W_R.rho = 0.125;   // 密度
    W_R.u   = 0.0;     // X 速度
    W_R.v   = 0.0;     // Y 速度
    W_R.w   = 0.0;     // Z 速度
    W_R.p   = 0.1;     // 壓力
    W_R.bx  = 0.75;    // X 磁場 (與左側保持一致)
    W_R.by  = -1.0;    // Y 磁場 (方向反轉)
    W_R.bz  = 0.0;     // Z 磁場

    // 2. 印出初始狀態以供確認
    std::cout << "--- Initial Primitive Variables (W) ---\n";
    std::cout << "Left State (W_L) : rho=" << W_L.rho << ", p=" << W_L.p 
              << ", By=" << W_L.by << " (u,v,w,Bz = 0), Bx=" << W_L.bx << "\n";
    std::cout << "Right State (W_R): rho=" << W_R.rho << ", p=" << W_R.p 
              << ", By=" << W_R.by << " (u,v,w,Bz = 0), Bx=" << W_R.bx << "\n\n";

    // 3. 計算並印出快磁聲波波速 (做為 Sanity Check)
    int direction = 0; // 沿著 X 方向計算通量
    double cf_L = MHDSolver::compute_fast_wave_speed(W_L, direction, gamma);
    double cf_R = MHDSolver::compute_fast_wave_speed(W_R, direction, gamma);
    
    std::cout << "--- Fast Magneto-acoustic Wave Speeds ---\n";
    std::cout << "cf_L (Left)  : " << cf_L << "\n";
    std::cout << "cf_R (Right) : " << cf_R << "\n\n";

    // 4. 呼叫 HLL Riemann Solver 計算數值通量
    try {
        std::vector<double> F_HLL = MHDSolver::compute_flux(W_L, W_R, direction, gamma);

        // 定義守恆變數名稱標籤以利閱讀
        std::vector<std::string> flux_names = {
            "Mass", "Momentum-X", "Momentum-Y", "Momentum-Z", 
            "Energy", "Induction-Bx", "Induction-By", "Induction-Bz"
        };

        std::cout << "--- Computed HLL Numerical Flux F(U_HLL) ---\n";
        for (size_t i = 0; i < 8; ++i) {
            std::cout << std::left << std::setw(15) << flux_names[i] 
                      << ": " << std::right << std::setw(10) << std::fixed 
                      << std::setprecision(6) << F_HLL[i] << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error computing flux: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n========================================================\n";
    return 0;
}