// =============================================================================
//  mhd_solver.hpp
//  2D Ideal MHD HLLD Approximate Riemann Solver (Header-only)
//
//  Reference: Miyoshi & Kusano, JCP 208 (2005) 315-344
//             "A multi-state HLL approximate Riemann solver for ideal
//              magnetohydrodynamics"
//
//  Design notes:
//    1. 2.5D layout -- spatial fluxes in X/Y only, but velocity and magnetic
//       field retain their Z components to correctly resolve Alfven waves.
//    2. Direction switching via index rotation: a single HLLD kernel handles
//       both X and Y fluxes (standard practice in Athena / PLUTO / FLASH).
//    3. Header-only: all functions are inline to avoid multiple-definition errors.
// =============================================================================

#ifndef MHD_SOLVER_HPP
#define MHD_SOLVER_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace MHD {

// -----------------------------------------------------------------------------
//  Conservative variable indices (8 components, 2.5D MHD)
//    IDN       : mass density rho
//    IM1/2/3   : momenta rho*u, rho*v, rho*w
//    IEN       : total energy E
//    IB1/2/3   : magnetic field Bx, By, Bz
//
//  Convention: after rotating to the normal frame, index "1" always holds
//  the normal component and "2","3" hold the two tangential components.
// -----------------------------------------------------------------------------
constexpr int IDN = 0;
constexpr int IM1 = 1;
constexpr int IM2 = 2;
constexpr int IM3 = 3;
constexpr int IEN = 4;
constexpr int IB1 = 5;
constexpr int IB2 = 6;
constexpr int IB3 = 7;
constexpr int NVAR = 8;

// Small floor to guard against division by zero
constexpr double TINY_NUMBER = 1.0e-20;

// -----------------------------------------------------------------------------
//  State -- primitive variables W = [rho, u, v, w, p, Bx, By, Bz]
//
//  u/v/w and Bx/By/Bz are in the physical (x,y,z) frame on input/output.
//  Inside the solver they are temporarily rotated to the normal frame.
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
    //  Primitive -> conserved
    //  U = [rho, rho*u, rho*v, rho*w, E, Bx, By, Bz]
    //  E = p/(gamma-1)  +  (1/2)*rho*|v|^2  +  (1/2)*|B|^2
    //      thermal          kinetic              magnetic
    // -------------------------------------------------------------------------
    std::vector<double> to_conserved(double gamma) const {
        std::vector<double> U(NVAR);
        U[IDN] = rho;
        U[IM1] = rho * u;
        U[IM2] = rho * v;
        U[IM3] = rho * w;
        double ke = 0.5 * rho * (u*u + v*v + w*w);     // kinetic energy density
        double me = 0.5 * (Bx*Bx + By*By + Bz*Bz);     // magnetic energy density
        U[IEN] = p / (gamma - 1.0) + ke + me;
        U[IB1] = Bx;
        U[IB2] = By;
        U[IB3] = Bz;
        return U;
    }

    // -------------------------------------------------------------------------
    //  Conserved -> primitive
    //  p = (gamma-1) * (E - kinetic - magnetic)
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
//  Rotate state from the physical frame to the normal frame.
//
//  direction = 0 (X-flux): no-op; (u,v,w) and (Bx,By,Bz) are unchanged.
//  direction = 1 (Y-flux): cycle left so that the Y component moves to slot 1:
//      (u,v,w)    ->  (v,w,u)
//      (Bx,By,Bz) ->  (By,Bz,Bx)
//
//  This lets a single HLLD kernel handle both directions; the normal component
//  is always in slot "1" after rotation.
// -----------------------------------------------------------------------------
inline State rotate_to_normal(const State& W, int direction) {
    if (direction == 0) {
        // X-direction: no rotation needed
        return W;
    } else {
        // Y-direction: move Y component into the normal (slot-1) position
        State Wr = W;
        Wr.u  = W.v;   Wr.v  = W.w;   Wr.w  = W.u;
        Wr.Bx = W.By;  Wr.By = W.Bz;  Wr.Bz = W.Bx;
        return Wr;
    }
}

// Inverse rotation: map the normal-frame flux back to the physical frame.
// Reverses the left-cycle applied in rotate_to_normal: (1,2,3) -> (3,1,2)
inline std::vector<double> rotate_flux_back(const std::vector<double>& F, int direction) {
    if (direction == 0) {
        return F;  // X-direction: no rotation needed
    } else {
        // Right-cycle to undo (u,v,w)->(v,w,u): F_normal=(1,2,3) -> F_phys=(3,1,2)
        std::vector<double> Fb(NVAR);
        Fb[IDN] = F[IDN];
        Fb[IEN] = F[IEN];
        // F[IM1]=normal(y)->IM2, F[IM2]=tang1(z)->IM3, F[IM3]=tang2(x)->IM1
        Fb[IM1] = F[IM3];
        Fb[IM2] = F[IM1];
        Fb[IM3] = F[IM2];
        // Same permutation for magnetic flux components
        Fb[IB1] = F[IB3];
        Fb[IB2] = F[IB1];
        Fb[IB3] = F[IB2];
        return Fb;
    }
}

// -----------------------------------------------------------------------------
//  Fast magnetosonic wave speed (evaluated in the normal frame, so Bx = Bn)
//
//  Dispersion relation for the outermost MHD characteristic:
//      cf^2 = (1/2) * { (a^2 + b^2) + sqrt[(a^2 + b^2)^2 - 4*a^2*bn^2] }
//  where:
//      a^2  = gamma*p/rho   -- sound speed squared
//      b^2  = |B|^2/rho     -- total Alfven speed squared
//      bn^2 = Bx^2/rho      -- normal Alfven speed squared (Bx is normal after rotation)
//
//  The discriminant is non-negative in exact arithmetic; clamp to 0 to guard
//  against tiny negative values from floating-point rounding.
// -----------------------------------------------------------------------------
inline double fast_magnetosonic_speed(const State& W, double gamma) {
    double rho = std::max(W.rho, TINY_NUMBER);
    double a2  = gamma * W.p / rho;                              // sound speed^2
    double b2  = (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz) / rho;   // total Alfven speed^2
    double bn2 = W.Bx * W.Bx / rho;                             // normal Alfven speed^2
    double term = a2 + b2;
    double disc = term * term - 4.0 * a2 * bn2;
    disc = std::max(disc, 0.0);                                  // floating-point guard
    double cf2 = 0.5 * (term + std::sqrt(disc));
    return std::sqrt(std::max(cf2, 0.0));
}

// -----------------------------------------------------------------------------
//  Physical flux F(U) evaluated in the normal frame (normal direction = x)
//
//  F_rho  = rho*u
//  F_mom1 = rho*u^2 + pT - Bx^2          (normal momentum; pT = p + |B|^2/2)
//  F_mom2 = rho*u*v - Bx*By              (tangential momentum 1)
//  F_mom3 = rho*u*w - Bx*Bz              (tangential momentum 2)
//  F_E    = (E + pT)*u - Bx*(v.B)        (energy + normal Poynting flux)
//  F_B1   = 0                             (normal B is frozen; enforces div B = 0)
//  F_B2   = By*u - Bx*v                  (induction, tangential 1)
//  F_B3   = Bz*u - Bx*w                  (induction, tangential 2)
// -----------------------------------------------------------------------------
inline std::vector<double> physical_flux(const State& W, double gamma) {
    std::vector<double> F(NVAR);
    double pmag = 0.5 * (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz);
    double ptot = W.p + pmag;                                    // total pressure
    double vdotB = W.u*W.Bx + W.v*W.By + W.w*W.Bz;
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
//  HLLD Riemann solver -- normal frame (Miyoshi & Kusano 2005)
//
//  Wave structure (left to right):
//      U_L | U_L* | U_L** | U_R** | U_R* | U_R
//          SL     SL*      SM      SR*    SR
//
//  SL, SR   -- outermost fast magnetosonic waves (Davis estimate)
//  SL*, SR* -- rotational (Alfven) discontinuities
//  SM       -- contact discontinuity (entropy wave)
//
//  Key equations (Miyoshi & Kusano eqs. 16-40):
//
//  (1) Outer wave speeds (Davis):
//        SL = min(uL - cfL, uR - cfR)
//        SR = max(uL + cfL, uR + cfR)
//
//  (2) Contact speed SM from normal-momentum R-H across both outer waves:
//        SM = [(SR-uR)*rhoR*uR - (SL-uL)*rhoL*uL - pTR + pTL]
//             / [(SR-uR)*rhoR   - (SL-uL)*rhoL]
//      Total pressure is uniform across all inner waves:
//        pT* = pTL + rhoL*(SL-uL)*(SM-uL)
//
//  (3) Star-state density from mass R-H:
//        rho_K* = rho_K*(S_K - u_K) / (S_K - SM),  K in {L, R}
//
//  (4) Alfven speeds:
//        SL* = SM - |Bn| / sqrt(rhoL*)
//        SR* = SM + |Bn| / sqrt(rhoR*)
//
//  (5) Tangential star-state from momentum + induction R-H:
//        denom = rho*(S_K - u_K)*(S_K - SM) - Bn^2
//        v*  = v  - Bn*By*(SM - u) / denom
//        B*y = By * [rho*(S_K - u_K)^2 - Bn^2] / denom
//        (w* and Bz* analogous; unchanged when denom -> 0)
//
//  (6) Star-state energy from energy R-H:
//        E* = [(S_K-u_K)*E - pT*u_K + pT**SM + Bn*(v.B - v*.B*)] / (S_K - SM)
//
//  (7) Double-star state from Alfven R-H across SL* / SR*:
//        v**  = (sqrt(rhoL*)*vL*  + sqrt(rhoR*)*vR*  + (ByR* - ByL*)*sgn(Bn))
//               / (sqrt(rhoL*) + sqrt(rhoR*))
//        By** = (sqrt(rhoL*)*ByR* + sqrt(rhoR*)*ByL*
//                + sqrt(rhoL*)*sqrt(rhoR*)*(vR* - vL*)*sgn(Bn))
//               / (sqrt(rhoL*) + sqrt(rhoR*))
//        E**  = E* -/+ sqrt(rho*) * (v*.B* - v**.B**) * sgn(Bn)
//
//  (8) Flux selection (6 regions):
//        F = FL                                        if SL  >= 0
//        F = FL + SL*(UL* - UL)                        if SL  <= 0 < SL*
//        F = FL + SL*(UL*-UL)  + SL** *(UL**-UL*)      if SL* <= 0 < SM
//        F = FR + SR*(UR*-UR)  + SR** *(UR**-UR*)       if SM  <= 0 < SR*
//        F = FR + SR*(UR* - UR)                        if SR* <= 0 < SR
//        F = FR                                        if SR  <= 0
//
//  Degenerate case (Bn ~ 0): no Alfven waves; SL* = SM = SR*; the double-star
//  region vanishes and HLLD reduces to an HLLC-like two-intermediate-state solver.
// -----------------------------------------------------------------------------
inline std::vector<double> hlld_flux_normal(const State& WL, const State& WR, double gamma) {
    // Step 0: enforce a unique normal B at the interface (average L and R).
    // With CT/GLM Bn would already be uniquely defined; the average is the
    // correct fallback when no divergence-cleaning is applied.
    double Bn = 0.5 * (WL.Bx + WR.Bx);
    State L = WL; L.Bx = Bn;
    State R = WR; R.Bx = Bn;
    double Bn2 = Bn * Bn;

    // Step 1: fast-wave speed estimates (Davis)
    double cfL = fast_magnetosonic_speed(L, gamma);
    double cfR = fast_magnetosonic_speed(R, gamma);
    double SL = std::min(L.u - cfL, R.u - cfR);
    double SR = std::max(L.u + cfL, R.u + cfR);

    // Step 2: conserved variables and physical fluxes for L and R states
    std::vector<double> UL = L.to_conserved(gamma);
    std::vector<double> UR = R.to_conserved(gamma);
    std::vector<double> FL = physical_flux(L, gamma);
    std::vector<double> FR = physical_flux(R, gamma);

    // Early exit for supersonic flow
    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    // Total pressure for L and R states
    double pmagL = 0.5*(L.Bx*L.Bx + L.By*L.By + L.Bz*L.Bz);
    double pmagR = 0.5*(R.Bx*R.Bx + R.By*R.By + R.Bz*R.Bz);
    double pTL = L.p + pmagL;
    double pTR = R.p + pmagR;

    // Step 3: contact speed SM and uniform intermediate total pressure pT*
    //   SM = [(SR-uR)*rhoR*uR - (SL-uL)*rhoL*uL - pTR + pTL]
    //        / [(SR-uR)*rhoR   - (SL-uL)*rhoL]
    double sLmuL = SL - L.u;
    double sRmuR = SR - R.u;
    double denom_SM = sRmuR * R.rho - sLmuL * L.rho;
    double SM = (sRmuR * R.rho * R.u - sLmuL * L.rho * L.u - pTR + pTL) / denom_SM;

    // pT* is equal on both sides of SM; compute from the left side
    double pT_star = pTL + L.rho * sLmuL * (SM - L.u);

    // Step 4: star-state densities from mass R-H: rho* = rho*(S-u)/(S-SM)
    double rhoL_star = L.rho * sLmuL / (SL - SM);
    double rhoR_star = R.rho * sRmuR / (SR - SM);
    double sqrtRhoLs = std::sqrt(rhoL_star);
    double sqrtRhoRs = std::sqrt(rhoR_star);

    // Step 5: Alfven wave speeds
    double SLs = SM - std::abs(Bn) / sqrtRhoLs;   // S_L*
    double SRs = SM + std::abs(Bn) / sqrtRhoRs;   // S_R*

    // Step 6: star-state tangential velocity and magnetic field
    //   denom = rho*(S-u)*(S-SM) - Bn^2
    //   When denom -> 0 (Alfven resonance or Bn = 0), tangential components
    //   are continuous across the wave and pass through unchanged.
    auto compute_star_tangential = [&](const State& W, double S, double sMu,
                                       double& vs, double& ws,
                                       double& Bys, double& Bzs) {
        double denom = W.rho * sMu * (S - SM) - Bn2;
        if (std::abs(denom) < TINY_NUMBER * W.rho * sMu * sMu) {
            // Degenerate: tangential components unchanged across the wave
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

    // Step 7: star-state energy from energy R-H
    //   E* = [(S-u)*E - pT*u + pT**SM + Bn*(v.B - v*.B*)] / (S - SM)
    double vBL  = L.u*Bn + L.v*L.By + L.w*L.Bz;
    double vBR  = R.u*Bn + R.v*R.By + R.w*R.Bz;
    double vBLs = SM*Bn + vL_s*ByL_s + wL_s*BzL_s;
    double vBRs = SM*Bn + vR_s*ByR_s + wR_s*BzR_s;

    double EL = UL[IEN];
    double ER = UR[IEN];
    double EL_s = (sLmuL * EL - pTL * L.u + pT_star * SM + Bn * (vBL - vBLs)) / (SL - SM);
    double ER_s = (sRmuR * ER - pTR * R.u + pT_star * SM + Bn * (vBR - vBRs)) / (SR - SM);

    // Assemble star-state conserved vectors (normal velocity u* = SM by construction)
    auto make_U_star = [&](double rho_s, double v_s, double w_s,
                           double By_s, double Bz_s, double E_s) {
        std::vector<double> Us(NVAR);
        Us[IDN] = rho_s;
        Us[IM1] = rho_s * SM;          // normal momentum: u* = SM
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

    // Step 8: select flux region.
    // When Bn ~ 0 there are no Alfven waves (SL* = SR* = SM); the double-star
    // region vanishes and we fall back to HLLC-like two-state treatment.
    bool degenerate = (Bn2 < TINY_NUMBER);

    if (!degenerate) {
        // Compute double-star (**) state from Alfven R-H conditions.
        // rho** = rho*; tangential v and B from the Alfven jump relations:
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

        // Energy jump: E** = E* -/+ sqrt(rho*) * (v*.B* - v**.B**) * sgn(Bn)
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

        // Six-region flux selection
        std::vector<double> F(NVAR);
        if (SLs >= 0.0) {
            // Region between SL and SL*: left star flux
            // F_L* = F_L + SL*(U_L* - U_L)
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]);
        } else if (SM >= 0.0) {
            // Region between SL* and SM: left double-star flux
            // F_L** = F_L + SL*(U_L*-U_L) + SL**(U_L**-U_L*)
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]) + SLs * (UL_ss[i] - UL_s[i]);
        } else if (SRs >= 0.0) {
            // Region between SM and SR*: right double-star flux
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]) + SRs * (UR_ss[i] - UR_s[i]);
        } else {
            // Region between SR* and SR: right star flux
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]);
        }
        return F;
    } else {
        // Degenerate path: Bn ~ 0, no Alfven waves; reduce to HLLC-like two states
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
//  Public interface: compute_flux
//
//  Inputs:
//    W_L, W_R  -- left and right primitive states in the physical (x,y,z) frame
//    direction -- 0 = X-flux, 1 = Y-flux
//    gamma     -- adiabatic index
//
//  Output: 8-component numerical flux vector in the physical frame
//
//  Steps: rotate to normal frame -> HLLD -> rotate flux back
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