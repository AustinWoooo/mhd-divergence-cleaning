// =============================================================================
//  mhd_solver.hpp
//  2D Ideal MHD HLLD Approximate Riemann Solver (Header-only)
//
//  Reference: Miyoshi & Kusano, JCP 208 (2005) 315-344
//             "A multi-state HLL approximate Riemann solver for ideal
//              magnetohydrodynamics"
//
//  設計重點:
//    1. 2.5D 架構 - 雖然空間只有 X/Y 兩個方向的通量,但速度與磁場保留 z 分量
//       以正確解析 Alfven 波(旋轉間斷)。
//    2. 方向切換 - 透過「索引旋轉」處理 X/Y 方向通量,而非重寫兩遍程式碼。
//       這是 Athena / PLUTO / FLASH 等成熟 MHD code 的標準做法。
//    3. Header-only - 所有實作皆為 inline,避免多重定義錯誤。
// =============================================================================

#ifndef MHD_SOLVER_HPP
#define MHD_SOLVER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace MHD {

// -----------------------------------------------------------------------------
//  常數定義
// -----------------------------------------------------------------------------
// 守恆變數的索引(8 個分量,2.5D MHD)
//   IDN: 質量密度 rho
//   IM1, IM2, IM3: 動量 rho*u, rho*v, rho*w
//   IEN: 總能量 E
//   IB1, IB2, IB3: 磁場 Bx, By, Bz
// 注意:在計算 X 方向通量時,IM1=rho*u(法向), IM2=rho*v(切向1), IM3=rho*w(切向2)
//                            IB1=Bx(法向),    IB2=By(切向1),    IB3=Bz(切向2)
//       在計算 Y 方向通量時,我們會「旋轉」狀態,使法向分量永遠在「1」的位置。
constexpr int IDN = 0;
constexpr int IM1 = 1;
constexpr int IM2 = 2;
constexpr int IM3 = 3;
constexpr int IEN = 4;
constexpr int IB1 = 5;
constexpr int IB2 = 6;
constexpr int IB3 = 7;
constexpr int NVAR = 8;

// 數值安全閾值,避免除以零
constexpr double TINY_NUMBER = 1.0e-20;

// -----------------------------------------------------------------------------
//  State 結構體:儲存原始變數(Primitive Variables)
// -----------------------------------------------------------------------------
//   rho: 密度
//   u, v, w: 三個速度分量(x, y, z)
//   p:      氣體壓力(thermal pressure,不含磁壓)
//   Bx, By, Bz: 三個磁場分量
//
// 注意:這裡的「u, v, w」是「物理 x, y, z」方向的速度。
//       在求解器內部,我們會旋轉成「法向 / 切向1 / 切向2」的座標系。
// -----------------------------------------------------------------------------
struct State {
    double rho;
    double u, v, w;
    double p;
    double Bx, By, Bz;

    State() : rho(0), u(0), v(0), w(0), p(0), Bx(0), By(0), Bz(0) {}

    State(double r, double uu, double vv, double ww,
          double pp, double bx, double by, double bz)
        : rho(r), u(uu), v(vv), w(ww), p(pp), Bx(bx), By(by), Bz(bz) {}

    // -------------------------------------------------------------------------
    //  原始 -> 守恆變數
    //  U = [rho, rho*u, rho*v, rho*w, E, Bx, By, Bz]
    //  E = p/(gamma-1) + (1/2)*rho*(u^2+v^2+w^2) + (1/2)*(Bx^2+By^2+Bz^2)
    //                   ^熱能              ^動能                ^磁能
    // -------------------------------------------------------------------------
    std::vector<double> to_conserved(double gamma) const {
        std::vector<double> U(NVAR);
        U[IDN] = rho;
        U[IM1] = rho * u;
        U[IM2] = rho * v;
        U[IM3] = rho * w;
        double ke = 0.5 * rho * (u*u + v*v + w*w);     // 動能密度
        double me = 0.5 * (Bx*Bx + By*By + Bz*Bz);     // 磁能密度
        U[IEN] = p / (gamma - 1.0) + ke + me;
        U[IB1] = Bx;
        U[IB2] = By;
        U[IB3] = Bz;
        return U;
    }

    // -------------------------------------------------------------------------
    //  守恆 -> 原始變數
    //  p = (gamma-1) * (E - 動能 - 磁能)
    // -------------------------------------------------------------------------
    static State from_conserved(const std::vector<double>& U, double gamma) {
        State W;
        W.rho = U[IDN];
        double inv_rho = 1.0 / std::max(W.rho, TINY_NUMBER);
        W.u  = U[IM1] * inv_rho;
        W.v  = U[IM2] * inv_rho;
        W.w  = U[IM3] * inv_rho;
        W.Bx = U[IB1];
        W.By = U[IB2];
        W.Bz = U[IB3];
        double ke = 0.5 * W.rho * (W.u*W.u + W.v*W.v + W.w*W.w);
        double me = 0.5 * (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz);
        W.p = (gamma - 1.0) * (U[IEN] - ke - me);
        return W;
    }
};

// -----------------------------------------------------------------------------
//  方向切換:將「物理座標」狀態旋轉為「法向座標」狀態
//
//  direction = 0 (X 方向):
//      法向 = x   ->  (u, Bx) 為法向分量,維持原樣
//      切向1= y       (v, By) 為切向分量
//      切向2= z       (w, Bz) 為切向分量
//
//  direction = 1 (Y 方向):
//      法向 = y   ->  將 (v, By) 換到「1」位置
//      切向1= z   ->  原 (w, Bz) 變成切向1
//      切向2= x   ->  原 (u, Bx) 變成切向2
//
//  關鍵:旋轉後我們可以用「同一段 HLLD 程式碼」處理兩個方向,
//        最後再把通量旋轉回物理座標。
// -----------------------------------------------------------------------------
inline State rotate_to_normal(const State& W, int direction) {
    if (direction == 0) {
        // X-方向:不旋轉
        return W;
    } else {
        // Y-方向:(u,v,w) -> (v,w,u),(Bx,By,Bz) -> (By,Bz,Bx)
        // 也就是把 Y 分量搬到第一個位置
        State Wr = W;
        Wr.u  = W.v;   Wr.v  = W.w;   Wr.w  = W.u;
        Wr.Bx = W.By;  Wr.By = W.Bz;  Wr.Bz = W.Bx;
        return Wr;
    }
}

// 把計算好的「法向座標」通量,旋轉回「物理座標」通量
inline std::vector<double> rotate_flux_back(const std::vector<double>& F, int direction) {
    if (direction == 0) {
        return F;  // X 方向不需轉
    } else {
        // 反向旋轉:(1,2,3) -> (3,1,2)
        // 因為原本的 (u,v,w) 是 (v,w,u),要還原回 (u,v,w)
        std::vector<double> Fb(NVAR);
        Fb[IDN] = F[IDN];
        Fb[IEN] = F[IEN];
        // 動量:F[IM1]=法向(=y) -> 應該放在 IM2
        //      F[IM2]=切1(=z)  -> 應該放在 IM3
        //      F[IM3]=切2(=x)  -> 應該放在 IM1
        Fb[IM1] = F[IM3];
        Fb[IM2] = F[IM1];
        Fb[IM3] = F[IM2];
        // 磁場同理
        Fb[IB1] = F[IB3];
        Fb[IB2] = F[IB1];
        Fb[IB3] = F[IB2];
        return Fb;
    }
}

// -----------------------------------------------------------------------------
//  快磁聲波 (Fast Magnetosonic Wave) 波速
//
//  數學推導:
//    MHD 有 7 個波(從 -cf 到 +cf):
//        fast(-cf), Alfven(-ca), slow(-cs), entropy(0),
//        slow(+cs),  Alfven(+ca), fast(+cf)
//
//    其中 fast wave 是最外側的波,色散關係:
//
//        c_f^2 = (1/2) * { a^2 + b^2 + sqrt[ (a^2+b^2)^2 - 4*a^2*b_n^2 ] }
//
//    其中:
//        a   = sqrt(gamma*p/rho)   ... 聲速
//        b^2 = (Bx^2+By^2+Bz^2)/rho ... 總 Alfven 速度平方
//        b_n = B_normal / sqrt(rho) ... 法向 Alfven 速度
//
//    使用「法向 Alfven 速度」是因為 fast wave 沿法向傳播時,
//    切向磁場提供額外的恢復力。
//
//  數值穩定性:
//    sqrt 內的判別式理論上恆 >= 0,但浮點誤差可能讓它變成微小負數,
//    所以用 std::max(0.0, ...) 保護。
// -----------------------------------------------------------------------------
inline double fast_magnetosonic_speed(const State& W, double gamma) {
    double rho = std::max(W.rho, TINY_NUMBER);
    double a2  = gamma * W.p / rho;                              // 聲速平方
    double b2  = (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz) / rho;      // 總 Alfven 速度平方
    double bn2 = W.Bx * W.Bx / rho;                              // 法向 Alfven 速度平方
                                                                  // (因為 rotate 後法向恆為 Bx)
    double term = a2 + b2;
    double disc = term * term - 4.0 * a2 * bn2;
    disc = std::max(disc, 0.0);                                  // 浮點保護
    double cf2 = 0.5 * (term + std::sqrt(disc));
    return std::sqrt(std::max(cf2, 0.0));
}

// -----------------------------------------------------------------------------
//  物理通量 F(U) 計算(已在「法向座標」下)
//
//  Ideal MHD 通量(法向方向):
//      F_rho   = rho * u
//      F_mom1  = rho*u^2 + p + (1/2)|B|^2 - Bx^2     (法向動量 + 總壓 - 磁應力)
//      F_mom2  = rho*u*v - Bx*By                      (切向動量1)
//      F_mom3  = rho*u*w - Bx*Bz                      (切向動量2)
//      F_E     = (E + p + (1/2)|B|^2)*u - Bx*(u*Bx + v*By + w*Bz)
//                ^總焓 * 法向速度          ^Poynting flux 的法向分量
//      F_B1    = 0                                    (法向磁場不變,div B = 0)
//      F_B2    = By*u - Bx*v                          (切向磁場感應)
//      F_B3    = Bz*u - Bx*w
// -----------------------------------------------------------------------------
inline std::vector<double> physical_flux(const State& W, double gamma) {
    std::vector<double> F(NVAR);
    double pmag = 0.5 * (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz);     // 磁壓
    double ptot = W.p + pmag;                                     // 總壓 = 熱壓 + 磁壓
    double vdotB = W.u*W.Bx + W.v*W.By + W.w*W.Bz;               // v.B
    double E = W.p/(gamma-1.0) + 0.5*W.rho*(W.u*W.u+W.v*W.v+W.w*W.w) + pmag;

    F[IDN] = W.rho * W.u;
    F[IM1] = W.rho * W.u * W.u + ptot - W.Bx * W.Bx;
    F[IM2] = W.rho * W.u * W.v - W.Bx * W.By;
    F[IM3] = W.rho * W.u * W.w - W.Bx * W.Bz;
    F[IEN] = (E + ptot) * W.u - W.Bx * vdotB;
    F[IB1] = 0.0;
    F[IB2] = W.By * W.u - W.Bx * W.v;
    F[IB3] = W.Bz * W.u - W.Bx * W.w;
    return F;
}

// -----------------------------------------------------------------------------
//  HLLD Riemann Solver(法向座標下)
//
//  Miyoshi & Kusano (2005) 的核心思想:
//  -------------------------------------
//    波結構從左到右為:
//        U_L  |  U_L*  |  U_L**  |  U_R**  |  U_R*  |  U_R
//             S_L      S_L*       S_M       S_R*    S_R
//
//    - S_L, S_R: 最外側的快磁聲波(將外部未擾動區域與內部分開)
//    - S_L*, S_R*: 阿爾文波(讓切向磁場與切向速度旋轉)
//    - S_M: 中間接觸間斷(密度跳變,但 u, p_total 連續)
//
//  推導步驟(關鍵方程式):
//  -------------------------------------
//  (1) 估計 S_L, S_R:用 Davis 估計
//          S_L = min(u_L - cf_L,  u_R - cf_R)
//          S_R = max(u_L + cf_L,  u_R + cf_R)
//
//  (2) S_M 來自「跨越所有內部波」的 Rankine-Hugoniot:
//      法向動量守恆,p_total 在 S_M 兩側相等
//          S_M = [(S_R - u_R)*rho_R*u_R - (S_L - u_L)*rho_L*u_L - pT_R + pT_L]
//                / [(S_R - u_R)*rho_R - (S_L - u_L)*rho_L]
//
//      其中 pT = p + |B|^2/2 (總壓)
//      且 S_M 區域內 u* = u** = S_M, pT* = pT** 為常數
//
//  (3) Star 區密度(從質量守恆 R-H 關係):
//          rho_L* = rho_L * (S_L - u_L) / (S_L - S_M)
//          rho_R* = rho_R * (S_R - u_R) / (S_R - S_M)
//
//  (4) 切向速度 / 磁場(由動量與感應方程的 R-H 推導):
//          v* = v - Bx*By*(S_M - u) / [rho*(S_L - u)(S_L - S_M) - Bx^2]
//          B*y = By * [rho*(S_L - u)^2 - Bx^2] / [rho*(S_L - u)(S_L - S_M) - Bx^2]
//      (w, Bz 同理)
//
//      當分母 -> 0(即 Bx^2 -> rho*(S_L-u)(S_L-S_M),退化情形):
//      切向分量不變,v* = v, B*y = By
//
//  (5) Star 區能量:
//          E_L* = [ (S_L - u_L)*E_L - pT_L*u_L + pT*S_M
//                   + Bx*(v.B - v*.B*) ] / (S_L - S_M)
//
//  (6) 阿爾文波速:
//          S_L* = S_M - |Bx| / sqrt(rho_L*)
//          S_R* = S_M + |Bx| / sqrt(rho_R*)
//
//  (7) Double-star 區(S_L* 與 S_R* 之間):
//      密度不變 rho** = rho*,但切向 v, B 由阿爾文波關係決定:
//          v** = (sqrt(rho_L*)*v_L* + sqrt(rho_R*)*v_R* + (B_R* - B_L*)*sign(Bx))
//                / (sqrt(rho_L*) + sqrt(rho_R*))
//          B** = (sqrt(rho_L*)*B_R* + sqrt(rho_R*)*B_L*
//                 + sqrt(rho_L*)*sqrt(rho_R*)*(v_R* - v_L*)*sign(Bx))
//                / (sqrt(rho_L*) + sqrt(rho_R*))
//
//  (8) 通量公式:
//          F = F_L                                  if S_L > 0
//          F = F_L + S_L*(U_L* - U_L)               if S_L <= 0 < S_L*
//          F = F_L* + S_L**(U_L** - U_L*)
//                  (= F_L + S_L*(U_L*-U_L) + S_L** *(U_L**-U_L*))   if S_L* <= 0 < S_M
//          F = F_R* + S_R**(U_R** - U_R*)
//                                                   if S_M <= 0 < S_R*
//          F = F_R + S_R*(U_R* - U_R)               if S_R* <= 0 < S_R
//          F = F_R                                  if S_R <= 0
//
//  退化處理(Bx ~ 0):
//      當 Bx -> 0 時,沒有阿爾文波,Star 區 = Double-star 區,
//      切向分量直接取對應的單側 Star 值,程式碼透過 if 分支處理。
// -----------------------------------------------------------------------------
inline std::vector<double> hlld_flux_normal(const State& WL, const State& WR, double gamma) {
    // ---- 第 0 步:法向磁場 Bx 必須左右一致(由 div B = 0 保證)----
    // 在無清理機制下,我們取平均;若使用 CT/GLM,Bx 在介面上是唯一定義的。
    double Bn = 0.5 * (WL.Bx + WR.Bx);
    State L = WL; L.Bx = Bn;
    State R = WR; R.Bx = Bn;
    double Bn2 = Bn * Bn;

    // ---- 第 1 步:計算快磁聲波波速,估計 S_L, S_R ----
    double cfL = fast_magnetosonic_speed(L, gamma);
    double cfR = fast_magnetosonic_speed(R, gamma);
    double SL = std::min(L.u - cfL, R.u - cfR);
    double SR = std::max(L.u + cfL, R.u + cfR);

    // ---- 第 2 步:左右態的守恆變數與物理通量 ----
    std::vector<double> UL = L.to_conserved(gamma);
    std::vector<double> UR = R.to_conserved(gamma);
    std::vector<double> FL = physical_flux(L, gamma);
    std::vector<double> FR = physical_flux(R, gamma);

    // 早期返回:全超音速向右 / 向左
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    // 左右態總壓
    double pmagL = 0.5*(L.Bx*L.Bx + L.By*L.By + L.Bz*L.Bz);
    double pmagR = 0.5*(R.Bx*R.Bx + R.By*R.By + R.Bz*R.Bz);
    double pTL = L.p + pmagL;
    double pTR = R.p + pmagR;

    // ---- 第 3 步:接觸間斷速度 S_M ----
    //   S_M = [(SR-uR)*rhoR*uR - (SL-uL)*rhoL*uL - pTR + pTL]
    //         / [(SR-uR)*rhoR - (SL-uL)*rhoL]
    double sLmuL = SL - L.u;
    double sRmuR = SR - R.u;
    double denom_SM = sRmuR * R.rho - sLmuL * L.rho;
    double SM = (sRmuR * R.rho * R.u - sLmuL * L.rho * L.u - pTR + pTL) / denom_SM;

    // 中間區常數總壓 pT*(左右相等)
    double pT_star = pTL + L.rho * sLmuL * (SM - L.u);
    // 等價於: pTR + R.rho*sRmuR*(SM - R.u),用任一個即可

    // ---- 第 4 步:Star 區密度 ----
    double rhoL_star = L.rho * sLmuL / (SL - SM);
    double rhoR_star = R.rho * sRmuR / (SR - SM);
    double sqrtRhoLs = std::sqrt(rhoL_star);
    double sqrtRhoRs = std::sqrt(rhoR_star);

    // ---- 第 5 步:阿爾文波速 ----
    double SLs = SM - std::abs(Bn) / sqrtRhoLs;   // S_L*
    double SRs = SM + std::abs(Bn) / sqrtRhoRs;   // S_R*

    // ---- 第 6 步:Star 區切向速度與磁場 ----
    //   分母:rho*(S - u)*(S - SM) - Bn^2
    //   當 Bn ~ 0 或分母 ~ 0 時退化,切向分量不變。
    auto compute_star_tangential = [&](const State& W, double S, double sMu,
                                       double& vs, double& ws,
                                       double& Bys, double& Bzs) {
        double denom = W.rho * sMu * (S - SM) - Bn2;
        if (std::abs(denom) < TINY_NUMBER * W.rho * sMu * sMu) {
            // 退化:切向分量保持不變
            vs = W.v;  ws = W.w;
            Bys = W.By; Bzs = W.Bz;
        } else {
            double inv = 1.0 / denom;
            // v* = v - Bn*By*(SM - u) / denom
            vs = W.v - Bn * W.By * (SM - W.u) * inv;
            ws = W.w - Bn * W.Bz * (SM - W.u) * inv;
            // B*y = By * (rho*(S-u)^2 - Bn^2) / denom
            double num = W.rho * sMu * sMu - Bn2;
            Bys = W.By * num * inv;
            Bzs = W.Bz * num * inv;
        }
    };

    double vL_s, wL_s, ByL_s, BzL_s;
    double vR_s, wR_s, ByR_s, BzR_s;
    compute_star_tangential(L, SL, sLmuL, vL_s, wL_s, ByL_s, BzL_s);
    compute_star_tangential(R, SR, sRmuR, vR_s, wR_s, ByR_s, BzR_s);

    // ---- 第 7 步:Star 區能量 ----
    //   E* = [ (S - u)*E - pT*u + pT_star*SM + Bn*(v.B - v*.B*) ] / (S - SM)
    double vBL  = L.u*Bn + L.v*L.By + L.w*L.Bz;
    double vBR  = R.u*Bn + R.v*R.By + R.w*R.Bz;
    double vBLs = SM*Bn + vL_s*ByL_s + wL_s*BzL_s;
    double vBRs = SM*Bn + vR_s*ByR_s + wR_s*BzR_s;

    double EL = UL[IEN];
    double ER = UR[IEN];
    double EL_s = (sLmuL * EL - pTL * L.u + pT_star * SM + Bn * (vBL - vBLs)) / (SL - SM);
    double ER_s = (sRmuR * ER - pTR * R.u + pT_star * SM + Bn * (vBR - vBRs)) / (SR - SM);

    // 組裝 Star 區守恆變數
    auto make_U_star = [&](double rho_s, double v_s, double w_s,
                           double By_s, double Bz_s, double E_s) {
        std::vector<double> Us(NVAR);
        Us[IDN] = rho_s;
        Us[IM1] = rho_s * SM;          // 法向動量:u* = SM
        Us[IM2] = rho_s * v_s;
        Us[IM3] = rho_s * w_s;
        Us[IEN] = E_s;
        Us[IB1] = Bn;
        Us[IB2] = By_s;
        Us[IB3] = Bz_s;
        return Us;
    };
    std::vector<double> UL_s = make_U_star(rhoL_star, vL_s, wL_s, ByL_s, BzL_s, EL_s);
    std::vector<double> UR_s = make_U_star(rhoR_star, vR_s, wR_s, ByR_s, BzR_s, ER_s);

    // ---- 第 8 步:根據 0 與各波速的相對位置,輸出通量 ----
    //   一個重要的退化情形:當 Bn ~ 0 時,SL* = SR* = SM,
    //   沒有阿爾文波分隔的 double-star 區,直接用 star 區即可。
    bool degenerate = (Bn2 < TINY_NUMBER);

    if (!degenerate) {
        // ---- 計算 Double-Star (**) 區 ----
        //   rho** = rho*
        //   v**, w**, By**, Bz** 由阿爾文跳躍條件給出
        double sgn_Bn = (Bn >= 0.0) ? 1.0 : -1.0;
        double sum_sqrt = sqrtRhoLs + sqrtRhoRs;
        double inv_sum  = 1.0 / sum_sqrt;

        double v_ss  = (sqrtRhoLs * vL_s + sqrtRhoRs * vR_s
                        + (ByR_s - ByL_s) * sgn_Bn) * inv_sum;
        double w_ss  = (sqrtRhoLs * wL_s + sqrtRhoRs * wR_s
                        + (BzR_s - BzL_s) * sgn_Bn) * inv_sum;
        double By_ss = (sqrtRhoLs * ByR_s + sqrtRhoRs * ByL_s
                        + sqrtRhoLs * sqrtRhoRs * (vR_s - vL_s) * sgn_Bn) * inv_sum;
        double Bz_ss = (sqrtRhoLs * BzR_s + sqrtRhoRs * BzL_s
                        + sqrtRhoLs * sqrtRhoRs * (wR_s - wL_s) * sgn_Bn) * inv_sum;

        // 能量跳躍:E** = E* +/- sqrt(rho*) * (v*.B* - v**.B**) * sign(Bn)
        double vB_ss = SM*Bn + v_ss*By_ss + w_ss*Bz_ss;
        double EL_ss = EL_s - sqrtRhoLs * (vBLs - vB_ss) * sgn_Bn;
        double ER_ss = ER_s + sqrtRhoRs * (vBRs - vB_ss) * sgn_Bn;

        auto make_U_ss = [&](double rho_ss, double E_ss) {
            std::vector<double> Uss(NVAR);
            Uss[IDN] = rho_ss;
            Uss[IM1] = rho_ss * SM;
            Uss[IM2] = rho_ss * v_ss;
            Uss[IM3] = rho_ss * w_ss;
            Uss[IEN] = E_ss;
            Uss[IB1] = Bn;
            Uss[IB2] = By_ss;
            Uss[IB3] = Bz_ss;
            return Uss;
        };
        std::vector<double> UL_ss = make_U_ss(rhoL_star, EL_ss);
        std::vector<double> UR_ss = make_U_ss(rhoR_star, ER_ss);

        // 通量分支(六種情況)
        std::vector<double> F(NVAR);
        if (SLs >= 0.0) {
            // 0 落在 S_L 與 S_L* 之間,使用左 star 通量
            // F_L* = F_L + S_L*(U_L* - U_L)
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]);
        } else if (SM >= 0.0) {
            // 0 落在 S_L* 與 S_M 之間,使用左 double-star 通量
            // F_L** = F_L + S_L*(U_L* - U_L) + S_L**(U_L** - U_L*)
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]) + SLs * (UL_ss[i] - UL_s[i]);
        } else if (SRs >= 0.0) {
            // 0 落在 S_M 與 S_R* 之間,使用右 double-star 通量
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]) + SRs * (UR_ss[i] - UR_s[i]);
        } else {
            // 0 落在 S_R* 與 S_R 之間,使用右 star 通量
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]);
        }
        return F;
    } else {
        // ---- 退化情形:Bn ~ 0,沒有阿爾文波(SL* = SM = SR*)----
        // 退化為 HLLC-like 兩中間狀態
        std::vector<double> F(NVAR);
        if (SM >= 0.0) {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]);
        } else {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]);
        }
        return F;
    }
}

// -----------------------------------------------------------------------------
//  對外介面:compute_flux
//
//  輸入:
//     W_L, W_R:  左右原始變數狀態(物理 x, y, z 座標系)
//     direction: 0 = X 方向通量,1 = Y 方向通量
//     gamma:     絕熱指數
//
//  輸出:8 分量的數值通量向量(物理座標系)
//
//  做法:
//     1. 把左右 state 「旋轉」到法向座標系
//     2. 在法向座標系下執行 HLLD
//     3. 把通量「旋轉回」物理座標系
// -----------------------------------------------------------------------------
inline std::vector<double> compute_flux(const State& W_L, const State& W_R,
                                        int direction, double gamma) {
    if (direction != 0 && direction != 1) {
        throw std::invalid_argument("direction must be 0 (X) or 1 (Y)");
    }
    State L_rot = rotate_to_normal(W_L, direction);
    State R_rot = rotate_to_normal(W_R, direction);
    std::vector<double> F_rot = hlld_flux_normal(L_rot, R_rot, gamma);
    return rotate_flux_back(F_rot, direction);
}

} // namespace MHD

#endif // MHD_SOLVER_HPP
