// =============================================================================
//  src/hlld.cpp
//  HLLD Approximate Riemann Solver -- implementations
//
//  Reference: Miyoshi & Kusano, JCP 208 (2005) 315-344
// =============================================================================

#include "HLLD_mhd_solver.hpp"

namespace MHD {

// -----------------------------------------------------------------------------
//  PrimState member functions
// -----------------------------------------------------------------------------

State PrimState::to_conserved(double gamma) const {
    State U{};
    U[IDN] = rho;
    U[IM1] = rho * u;
    U[IM2] = rho * v;
    U[IM3] = rho * w;
    double ke = 0.5 * rho * (u*u + v*v + w*w);
    double me = 0.5 * (Bx*Bx + By*By + Bz*Bz);
    U[IEN] = p / (gamma - 1.0) + ke + me;
    U[IB1] = Bx;
    U[IB2] = By;
    U[IB3] = Bz;
    U[IPSI] = psi;
    return U;
}

PrimState PrimState::from_conserved(const State& U, double gamma) {
    PrimState W;
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
    W.psi = U[IPSI];
    return W;
}

// -----------------------------------------------------------------------------
//  Frame rotation helpers
// -----------------------------------------------------------------------------

PrimState rotate_to_normal(const PrimState& W, int direction) {
    if (direction == 0) {
        return W;
    } else {
        PrimState Wr = W;
        Wr.u  = W.v;   Wr.v  = W.w;   Wr.w  = W.u;
        Wr.Bx = W.By;  Wr.By = W.Bz;  Wr.Bz = W.Bx;
        return Wr;
    }
}

State rotate_flux_back(const State& F, int direction) {
    if (direction == 0) {
        return F;
    } else {
        State Fb{};
        Fb[IDN] = F[IDN];
        Fb[IEN] = F[IEN];
        Fb[IM1] = F[IM3];
        Fb[IM2] = F[IM1];
        Fb[IM3] = F[IM2];
        Fb[IB1] = F[IB3];
        Fb[IB2] = F[IB1];
        Fb[IB3] = F[IB2];
        Fb[IPSI] = F[IPSI];
        return Fb;
    }
}

// -----------------------------------------------------------------------------
//  Wave speed and flux utilities
// -----------------------------------------------------------------------------

double fast_magnetosonic_speed(const PrimState& W, double gamma) {
    double rho = std::max(W.rho, TINY_NUMBER);
    double a2  = gamma * W.p / rho;
    double b2  = (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz) / rho;
    double bn2 = W.Bx * W.Bx / rho;
    double term = a2 + b2;
    double disc = std::max(term * term - 4.0 * a2 * bn2, 0.0);
    return std::sqrt(std::max(0.5 * (term + std::sqrt(disc)), 0.0));
}

State physical_flux(const PrimState& W, double gamma) {
    State F{};
    double pmag = 0.5 * (W.Bx*W.Bx + W.By*W.By + W.Bz*W.Bz);
    double ptot = W.p + pmag;
    double vdotB = W.u*W.Bx + W.v*W.By + W.w*W.Bz;
    double Etot = W.p/(gamma-1.0) + 0.5*W.rho*(W.u*W.u+W.v*W.v+W.w*W.w) + pmag;

    F[IDN] = W.rho * W.u;
    F[IM1] = W.rho * W.u * W.u + ptot - W.Bx * W.Bx;
    F[IM2] = W.rho * W.u * W.v - W.Bx * W.By;
    F[IM3] = W.rho * W.u * W.w - W.Bx * W.Bz;
    F[IEN] = (Etot + ptot) * W.u - W.Bx * vdotB;
    F[IB1] = 0.0;
    F[IB2] = W.By * W.u - W.Bx * W.v;
    F[IB3] = W.Bz * W.u - W.Bx * W.w;
    F[IPSI] = 0.0;
    return F;
}

// -----------------------------------------------------------------------------
//  HLLD Riemann solver (normal frame)
// -----------------------------------------------------------------------------

State hlld_flux_normal(const PrimState& WL, const PrimState& WR, double gamma) {
    // Enforce a unique normal B at the interface.
    double Bn = 0.5 * (WL.Bx + WR.Bx);
    PrimState L = WL; L.Bx = Bn;
    PrimState R = WR; R.Bx = Bn;
    double Bn2 = Bn * Bn;

    // Step 1: fast-wave speed estimates (Davis)
    double cfL = fast_magnetosonic_speed(L, gamma);
    double cfR = fast_magnetosonic_speed(R, gamma);
    double SL = std::min(L.u - cfL, R.u - cfR);
    double SR = std::max(L.u + cfL, R.u + cfR);

    // Step 2: conserved variables and physical fluxes
    State UL = L.to_conserved(gamma);
    State UR = R.to_conserved(gamma);
    State FL = physical_flux(L, gamma);
    State FR = physical_flux(R, gamma);

    if (SL >= 0.0) return FL;
    if (SR <= 0.0) return FR;

    double pmagL = 0.5*(L.Bx*L.Bx + L.By*L.By + L.Bz*L.Bz);
    double pmagR = 0.5*(R.Bx*R.Bx + R.By*R.By + R.Bz*R.Bz);
    double pTL = L.p + pmagL;
    double pTR = R.p + pmagR;

    // Step 3: contact speed SM and intermediate total pressure pT*
    double sLmuL = SL - L.u;
    double sRmuR = SR - R.u;
    double denom_SM = sRmuR * R.rho - sLmuL * L.rho;
    double SM = (sRmuR * R.rho * R.u - sLmuL * L.rho * L.u - pTR + pTL) / denom_SM;
    double pT_star = pTL + L.rho * sLmuL * (SM - L.u);

    // Step 4: star-state densities
    double rhoL_star = L.rho * sLmuL / (SL - SM);
    double rhoR_star = R.rho * sRmuR / (SR - SM);
    double sqrtRhoLs = std::sqrt(rhoL_star);
    double sqrtRhoRs = std::sqrt(rhoR_star);

    // Step 5: Alfven wave speeds
    double SLs = SM - std::abs(Bn) / sqrtRhoLs;
    double SRs = SM + std::abs(Bn) / sqrtRhoRs;

    // Step 6: star-state tangential velocity and magnetic field
    auto compute_star_tangential = [&](const PrimState& W, double S, double sMu,
                                       double& vs, double& ws,
                                       double& Bys, double& Bzs) {
        double denom = W.rho * sMu * (S - SM) - Bn2;
        if (std::abs(denom) < TINY_NUMBER * W.rho * sMu * sMu) {
            vs = W.v;  ws = W.w;
            Bys = W.By; Bzs = W.Bz;
        } else {
            double inv = 1.0 / denom;
            vs = W.v - Bn * W.By * (SM - W.u) * inv;
            ws = W.w - Bn * W.Bz * (SM - W.u) * inv;
            double num = W.rho * sMu * sMu - Bn2;
            Bys = W.By * num * inv;
            Bzs = W.Bz * num * inv;
        }
    };

    double vL_s, wL_s, ByL_s, BzL_s;
    double vR_s, wR_s, ByR_s, BzR_s;
    compute_star_tangential(L, SL, sLmuL, vL_s, wL_s, ByL_s, BzL_s);
    compute_star_tangential(R, SR, sRmuR, vR_s, wR_s, ByR_s, BzR_s);

    // Step 7: star-state energy
    double vBL  = L.u*Bn + L.v*L.By + L.w*L.Bz;
    double vBR  = R.u*Bn + R.v*R.By + R.w*R.Bz;
    double vBLs = SM*Bn + vL_s*ByL_s + wL_s*BzL_s;
    double vBRs = SM*Bn + vR_s*ByR_s + wR_s*BzR_s;

    double EL = UL[IEN];
    double ER = UR[IEN];
    double EL_s = (sLmuL * EL - pTL * L.u + pT_star * SM + Bn * (vBL - vBLs)) / (SL - SM);
    double ER_s = (sRmuR * ER - pTR * R.u + pT_star * SM + Bn * (vBR - vBRs)) / (SR - SM);

    auto make_U_star = [&](double rho_s, double v_s, double w_s,
                           double By_s, double Bz_s, double E_s) {
        State Us{};
        Us[IDN] = rho_s;
        Us[IM1] = rho_s * SM;
        Us[IM2] = rho_s * v_s;
        Us[IM3] = rho_s * w_s;
        Us[IEN] = E_s;
        Us[IB1] = Bn;
        Us[IB2] = By_s;
        Us[IB3] = Bz_s;
        Us[IPSI] = 0.0;
        return Us;
    };
    State UL_s = make_U_star(rhoL_star, vL_s, wL_s, ByL_s, BzL_s, EL_s);
    State UR_s = make_U_star(rhoR_star, vR_s, wR_s, ByR_s, BzR_s, ER_s);

    // Step 8: flux selection
    bool degenerate = (Bn2 < TINY_NUMBER);

    if (!degenerate) {
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

        double vB_ss = SM*Bn + v_ss*By_ss + w_ss*Bz_ss;
        double EL_ss = EL_s - sqrtRhoLs * (vBLs - vB_ss) * sgn_Bn;
        double ER_ss = ER_s + sqrtRhoRs * (vBRs - vB_ss) * sgn_Bn;

        auto make_U_ss = [&](double rho_ss, double E_ss) {
            State Uss{};
            Uss[IDN] = rho_ss;
            Uss[IM1] = rho_ss * SM;
            Uss[IM2] = rho_ss * v_ss;
            Uss[IM3] = rho_ss * w_ss;
            Uss[IEN] = E_ss;
            Uss[IB1] = Bn;
            Uss[IB2] = By_ss;
            Uss[IB3] = Bz_ss;
            Uss[IPSI] = 0.0;
            return Uss;
        };
        State UL_ss = make_U_ss(rhoL_star, EL_ss);
        State UR_ss = make_U_ss(rhoR_star, ER_ss);

        State F{};
        if (SLs >= 0.0) {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]);
        } else if (SM >= 0.0) {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]) + SLs * (UL_ss[i] - UL_s[i]);
        } else if (SRs >= 0.0) {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]) + SRs * (UR_ss[i] - UR_s[i]);
        } else {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]);
        }
        F[IPSI] = 0.0;
        return F;
    } else {
        State F{};
        if (SM >= 0.0) {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FL[i] + SL * (UL_s[i] - UL[i]);
        } else {
            for (int i = 0; i < NVAR; ++i)
                F[i] = FR[i] + SR * (UR_s[i] - UR[i]);
        }
        F[IPSI] = 0.0;
        return F;
    }
}

// -----------------------------------------------------------------------------
//  Public interface
// -----------------------------------------------------------------------------

State compute_flux(const PrimState& W_L, const PrimState& W_R,
                   int direction, double gamma) {
    if (direction != 0 && direction != 1) {
        throw std::invalid_argument("direction must be 0 (X) or 1 (Y)");
    }
    PrimState L_rot = rotate_to_normal(W_L, direction);
    PrimState R_rot = rotate_to_normal(W_R, direction);
    State F_rot = hlld_flux_normal(L_rot, R_rot, gamma);
    return rotate_flux_back(F_rot, direction);
}

} // namespace MHD
