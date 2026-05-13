#ifndef MHD_SOLVER_HPP
#define MHD_SOLVER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ====================================================================
// 1. 狀態結構體定義 (State)
// 負責儲存原始變數 (Primitive Variables, W) 
// 並提供與守恆變數 (Conservative Variables, U) 互相轉換的介面
// ====================================================================
struct State {
    // 原始變數 W: [rho, u, v, w, p, Bx, By, Bz]
    double rho; // 密度
    double u;   // X 方向速度
    double v;   // Y 方向速度
    double w;   // Z 方向速度
    double p;   // 熱力學壓力
    double bx;  // X 方向磁場
    double by;  // Y 方向磁場
    double bz;  // Z 方向磁場

    // 計算總動能
    inline double kinetic_energy() const {
        return 0.5 * rho * (u * u + v * v + w * w);
    }

    // 計算總磁能
    inline double magnetic_energy() const {
        return 0.5 * (bx * bx + by * by + bz * bz);
    }

    // 將原始變數 (W) 轉換為守恆變數 (U)
    // U = [rho, rho*u, rho*v, rho*w, E, Bx, By, Bz]
    std::vector<double> to_conservative(double gamma) const {
        std::vector<double> U(8, 0.0);
        U[0] = rho;
        U[1] = rho * u;
        U[2] = rho * v;
        U[3] = rho * w;
        // 總能量 E = 內能 + 動能 + 磁能
        // 內能 = p / (gamma - 1)
        U[4] = (p / (gamma - 1.0)) + kinetic_energy() + magnetic_energy();
        U[5] = bx;
        U[6] = by;
        U[7] = bz;
        return U;
    }

    // 從守恆變數 (U) 轉換回原始變數 (W)
    void from_conservative(const std::vector<double>& U, double gamma) {
        rho = U[0];
        u   = U[1] / rho;
        v   = U[2] / rho;
        w   = U[3] / rho;
        bx  = U[5];
        by  = U[6];
        bz  = U[7];
        
        double E = U[4];
        // 壓力 p = (gamma - 1) * (E - 動能 - 磁能)
        p = (gamma - 1.0) * (E - kinetic_energy() - magnetic_energy());
        
        // 確保壓力與密度恆正 (避免數值崩潰)
        if (p <= 0.0 || rho <= 0.0) {
            // 在實際應用中，這裡通常會切換到一階格式或進行數值修正
            p = std::max(p, 1e-12);
            rho = std::max(rho, 1e-12);
        }
    }
};

// ====================================================================
// 2. 核心邏輯庫 (Core Logic)
// ====================================================================

namespace MHDSolver {

    // 計算快磁聲波波速 (Fast Magneto-acoustic Wave Speed)
    // 數學推導：
    // a^2 = gamma * p / rho (聲速平方)
    // vA^2 = (Bx^2 + By^2 + Bz^2) / rho (阿爾文波速平方)
    // ca^2 = Bn^2 / rho (沿著法線方向的阿爾文波速平方)
    // cf^2 = 0.5 * [ (a^2 + vA^2) + sqrt( (a^2 + vA^2)^2 - 4*a^2*ca^2 ) ]
    double compute_fast_wave_speed(const State& W, int direction, double gamma) {
        double a_sq = gamma * W.p / W.rho; // 聲速平方
        double b_sq = (W.bx * W.bx + W.by * W.by + W.bz * W.bz) / W.rho; // vA^2
        
        // 判斷法向磁場 (Bn)
        double bn = (direction == 0) ? W.bx : W.by; 
        double bn_sq = (bn * bn) / W.rho; // ca^2

        double sum = a_sq + b_sq;
        double diff = sum * sum - 4.0 * a_sq * bn_sq;
        
        // 避免浮點數誤差導致的負數開根號
        diff = std::max(diff, 0.0); 
        
        return std::sqrt(0.5 * (sum + std::sqrt(diff)));
    }

    // 計算物理通量 (Physical Flux Vector F(U) 或 G(U))
    // 這裡展示了 X 與 Y 方向切換時，動量與磁場張量的對應關係。
    std::vector<double> compute_physical_flux(const State& W, int direction, double gamma) {
        std::vector<double> F(8, 0.0);
        
        double p_tot = W.p + W.magnetic_energy(); // 總壓 (熱壓 + 磁壓)
        double E = (W.p / (gamma - 1.0)) + W.kinetic_energy() + W.magnetic_energy();
        double v_dot_B = W.u * W.bx + W.v * W.by + W.w * W.bz;

        if (direction == 0) {
            // 法向量為 X 方向 (Flux F)
            // 速度法向分量 un = u, 磁場法向分量 bn = Bx
            F[0] = W.rho * W.u;
            F[1] = W.rho * W.u * W.u + p_tot - W.bx * W.bx; // X-Momentum
            F[2] = W.rho * W.u * W.v - W.bx * W.by;         // Y-Momentum
            F[3] = W.rho * W.u * W.w - W.bx * W.bz;         // Z-Momentum
            F[4] = (E + p_tot) * W.u - W.bx * v_dot_B;      // Energy Flux
            F[5] = 0.0;                                     // div B = 0 => 1D 情況下 Bx 無通量變化
            F[6] = W.u * W.by - W.v * W.bx;                 // Induction By
            F[7] = W.u * W.bz - W.w * W.bx;                 // Induction Bz
        } else if (direction == 1) {
            // 法向量為 Y 方向 (Flux G)
            // 速度法向分量 un = v, 磁場法向分量 bn = By
            F[0] = W.rho * W.v;
            F[1] = W.rho * W.v * W.u - W.by * W.bx;         // X-Momentum
            F[2] = W.rho * W.v * W.v + p_tot - W.by * W.by; // Y-Momentum
            F[3] = W.rho * W.v * W.w - W.by * W.bz;         // Z-Momentum
            F[4] = (E + p_tot) * W.v - W.by * v_dot_B;      // Energy Flux
            F[5] = W.v * W.bx - W.u * W.by;                 // Induction Bx
            F[6] = 0.0;                                     // div B = 0 => 1D 情況下 By 無通量變化
            F[7] = W.v * W.bz - W.w * W.by;                 // Induction Bz
        } else {
            throw std::invalid_argument("Direction 必須為 0 (X方向) 或 1 (Y方向)");
        }
        
        return F;
    }

    // ====================================================================
    // 3. Riemann Solver (HLL)
    // ====================================================================
    
    // 計算單一網格介面的 HLL 數值通量
    std::vector<double> compute_flux(const State& W_L, const State& W_R, int direction, double gamma) {
        // 1. 取得左右兩側的法向速度
        double un_L = (direction == 0) ? W_L.u : W_L.v;
        double un_R = (direction == 0) ? W_R.u : W_R.v;

        // 2. 計算左右兩側的快磁聲波波速
        double cf_L = compute_fast_wave_speed(W_L, direction, gamma);
        double cf_R = compute_fast_wave_speed(W_R, direction, gamma);

        // 3. 估計系統的最外側波速 (Davis Signal Velocity Estimate)
        // S_L: 最左側的波速 (Leftmost wave speed)
        // S_R: 最右側的波速 (Rightmost wave speed)
        double S_L = std::min(un_L - cf_L, un_R - cf_R);
        double S_R = std::max(un_L + cf_L, un_R + cf_R);

        // 4. 計算左右兩側的物理通量 F(U_L) 與 F(U_R)
        std::vector<double> F_L = compute_physical_flux(W_L, direction, gamma);
        std::vector<double> F_R = compute_physical_flux(W_R, direction, gamma);

        // 5. 判斷波速區間並計算數值通量 (Upwind logic)
        std::vector<double> F_HLL(8, 0.0);

        if (S_L >= 0.0) {
            // 情況 A: 流體完全以超音速向右流動，所有波都在右側
            F_HLL = F_L;
        } else if (S_R <= 0.0) {
            // 情況 B: 流體完全以超音速向左流動，所有波都在左側
            F_HLL = F_R;
        } else {
            // 情況 C: S_L < 0 < S_R (亞音速流動，介面位於 Star State 內)
            // 中間狀態推導 (Conservation over the Riemann Fan):
            // 根據積分守恆定律，中間通量 F_HLL 必須滿足：
            // F_HLL = [ S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L) ] / (S_R - S_L)
            
            std::vector<double> U_L = W_L.to_conservative(gamma);
            std::vector<double> U_R = W_R.to_conservative(gamma);

            double inv_dS = 1.0 / (S_R - S_L);
            for (int i = 0; i < 8; ++i) {
                F_HLL[i] = (S_R * F_L[i] - S_L * F_R[i] + S_L * S_R * (U_R[i] - U_L[i])) * inv_dS;
            }
        }

        return F_HLL;
    }
}

#endif // MHD_SOLVER_HPP