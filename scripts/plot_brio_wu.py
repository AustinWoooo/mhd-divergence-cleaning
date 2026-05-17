#!/usr/bin/env python3
"""
plot_brio_wu.py
===============
Self-contained 1D ideal-MHD Brio-Wu shock-tube solver and comparison plotter.

Implements a first-order Godunov scheme (HLL Riemann solver) to evolve the
standard Brio & Wu (1988) initial conditions and compares:

  1. Pure MHD  — clean initial conditions              (reference solution)
  2. Pure MHD  — Bx perturbed with sinusoidal div-B noise  (contaminated)
  3. GLM  MHD  — same perturbation, hyperbolic-GLM cleaning active (cleaned)

Plots ρ, P, u_x, B_y on a 2×2 grid and saves figures/brio_wu_comparison.png.

Physical setup (Brio & Wu 1988):
  Left  state : ρ=1.000,  p=1.0,  Bx=0.75,  By= 1.0,  u=v=w=0
  Right state : ρ=0.125,  p=0.1,  Bx=0.75,  By=-1.0,  u=v=w=0
  γ = 2,  domain x ∈ [-0.5, 0.5],  discontinuity at x = 0,  t_end = 0.1

Usage
-----
  python plot_brio_wu.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── conservative-variable layout (matches include/state.hpp) ─────────────────
RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI = range(9)
NVAR  = 9
GAMMA = 2.0   # Brio & Wu (1988)

# ── scalar primitive → conservative (used for cell initialisation) ────────────
def prim_to_cons(rho, u, v, w, p, Bx, By, Bz, psi=0.0):
    E = p / (GAMMA - 1.0) + 0.5*rho*(u*u+v*v+w*w) + 0.5*(Bx*Bx+By*By+Bz*Bz)
    return np.array([rho, rho*u, rho*v, rho*w, Bx, By, Bz, E, psi])

# ── vectorised conservative → primitive ──────────────────────────────────────
def c2p(U):
    """U (N,9) → tuple (rho, u, v, w, p, Bx, By, Bz, psi), each shape (N,)."""
    rho = U[:, RHO]
    u, v, w = U[:, MX]/rho, U[:, MY]/rho, U[:, MZ]/rho
    Bx, By, Bz = U[:, BX], U[:, BY], U[:, BZ]
    KE = 0.5*rho*(u*u + v*v + w*w)
    ME = 0.5*(Bx*Bx + By*By + Bz*Bz)
    p  = (GAMMA - 1.0) * np.maximum(U[:, EN] - KE - ME, 1e-14)
    return rho, u, v, w, p, Bx, By, Bz, U[:, PSI]

# ── fast magnetosonic speed (vectorised) ─────────────────────────────────────
def cf_vec(rho, p, Bx, By, Bz):
    a2  = GAMMA * p / rho
    b2  = (Bx*Bx + By*By + Bz*Bz) / rho
    bx2 = Bx*Bx / rho
    # cf² = (a²+b²)/2 + sqrt( ((a²+b²)/2)² − a²·bx² )
    disc = np.maximum((a2 + b2)**2 - 4.0*a2*bx2, 0.0)
    return np.sqrt(np.maximum(0.5*(a2 + b2 + np.sqrt(disc)), 0.0))

# ── physical flux F(U) in the x-direction (vectorised) ───────────────────────
def phys_flux(U):
    rho, u, v, w, p, Bx, By, Bz, _ = c2p(U)
    B2   = Bx*Bx + By*By + Bz*Bz
    ptot = p + 0.5*B2
    F = np.zeros_like(U)
    F[:, RHO] = rho*u
    F[:, MX]  = rho*u*u + ptot - Bx*Bx
    F[:, MY]  = rho*u*v - Bx*By
    F[:, MZ]  = rho*u*w - Bx*Bz
    # F[:, BX]  = 0   (left as zero; Bx is frozen in 1-D, overwritten for GLM)
    F[:, BY]  = By*u - Bx*v
    F[:, BZ]  = Bz*u - Bx*w
    F[:, EN]  = (U[:, EN] + ptot)*u - Bx*(Bx*u + By*v + Bz*w)
    return F                                         # F[:, PSI] = 0

# ── HLL Riemann flux at all faces (vectorised over N+1 faces) ────────────────
def hll_flux(UL, UR):
    rhoL, uL, _, _, pL, BxL, ByL, BzL, _ = c2p(UL)
    rhoR, uR, _, _, pR, BxR, ByR, BzR, _ = c2p(UR)
    cfL = cf_vec(rhoL, pL, BxL, ByL, BzL)
    cfR = cf_vec(rhoR, pR, BxR, ByR, BzR)
    SL  = np.minimum(uL - cfL, uR - cfR)          # left  wave-speed bound
    SR  = np.maximum(uL + cfL, uR + cfR)          # right wave-speed bound
    FL, FR = phys_flux(UL), phys_flux(UR)
    dS  = np.where(np.abs(SR - SL) < 1e-15, 1e-15, SR - SL)
    F_hll = (SR[:, None]*FL - SL[:, None]*FR
             + SL[:, None]*SR[:, None]*(UR - UL)) / dS[:, None]
    return np.where(SL[:, None] >= 0, FL,
           np.where(SR[:, None] <= 0, FR, F_hll))

# ── GLM Rusanov correction for the (Bx, ψ) subsystem ────────────────────────
def glm_correction(UL, UR, ch):
    """
    Dedner et al. (2002) mixed-GLM Rusanov flux for the 2×2 (Bx, ψ) system.
      F[Bx]  = (ψ_L + ψ_R)/2  − (ch/2)·(Bx_R − Bx_L)
      F[ψ]   = (ch²/2)·(Bx_L + Bx_R) − (ch/2)·(ψ_R − ψ_L)
    """
    BxL, psiL = UL[:, BX], UL[:, PSI]
    BxR, psiR = UR[:, BX], UR[:, PSI]
    corr = np.zeros_like(UL)
    corr[:, BX]  = 0.5*(psiL  + psiR ) - 0.5*ch*(BxR  - BxL )
    corr[:, PSI] = 0.5*ch*ch*(BxL + BxR) - 0.5*ch*(psiR - psiL)
    return corr

# ── single forward-Euler Godunov step ────────────────────────────────────────
def godunov_step(U, dx, cfl=0.4, glm=False, ch=2.0, dt_max=np.inf):
    rho, u, _, _, p, Bx, By, Bz, _ = c2p(U)
    max_sp = float(np.max(np.abs(u) + cf_vec(rho, p, Bx, By, Bz)))
    if glm:
        max_sp = max(max_sp, ch)       # GLM wave propagates at ch
    dt = min(cfl * dx / max_sp, dt_max)

    # Outflow ghost cells: copy edge states
    Ue = np.vstack([U[[0]], U, U[[-1]]])   # (N+2, 9)
    UL, UR = Ue[:-1], Ue[1:]              # (N+1, 9) left/right face states

    F = hll_flux(UL, UR)
    F[:, BX]  = 0.0   # ∂Bx/∂t = 0 in pure 1-D MHD; replaced below for GLM
    F[:, PSI] = 0.0
    if glm:
        F += glm_correction(UL, UR, ch)

    Unew = U - (dt / dx) * (F[1:] - F[:-1])

    # Mixed-GLM exponential damping of ψ (Dedner et al. 2002, eq. 24)
    if glm:
        cp = ch / np.sqrt(2.0)
        Unew[:, PSI] *= np.exp(-dt * ch*ch / (cp*cp))

    return Unew, dt

# ── simulation driver ─────────────────────────────────────────────────────────
def run_brio_wu(N=400, t_end=0.1, cfl=0.4, glm=False, ch=2.0,
                div_b_pert=False):
    """
    Evolve the 1-D Brio-Wu problem to t_end.

    Parameters
    ----------
    div_b_pert : bool
        If True, add a small sinusoidal perturbation to Bx so that ∇·B ≠ 0
        initially.  Used to demonstrate the effect of GLM cleaning.
    """
    dx = 1.0 / N
    x  = -0.5 + (np.arange(N) + 0.5) * dx   # cell-centre positions

    U = np.zeros((N, NVAR))
    for i, xi in enumerate(x):
        if xi < 0.0:
            U[i] = prim_to_cons(1.0,   0, 0, 0, 1.0,  0.75,  1.0, 0)
        else:
            U[i] = prim_to_cons(0.125, 0, 0, 0, 0.1,  0.75, -1.0, 0)

    if div_b_pert:
        # Sinusoidal Bx perturbation → non-zero discrete ∇·B throughout domain
        U[:, BX] += 0.05 * np.sin(4.0 * np.pi * x)

    t = 0.0
    while t < t_end - 1e-12:
        U, dt = godunov_step(U, dx, cfl=cfl, glm=glm, ch=ch,
                              dt_max=t_end - t)
        t += dt
    return x, U

# ── extract primitive variables for plotting ──────────────────────────────────
def extract(U):
    """Return (ρ, P, u_x, B_y) as 1-D numpy arrays."""
    rho, u, _, _, p, _, By, _, _ = c2p(U)
    return rho, p, u, By

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    figdir = Path("figures")
    figdir.mkdir(exist_ok=True)

    N = 400
    print("Brio-Wu shock tube — running 1-D MHD simulations …")
    print(f"  Setup: N={N}, γ={GAMMA}, t_end=0.1, CFL=0.4")

    x, U1 = run_brio_wu(N=N, glm=False, div_b_pert=False)
    print("  [1/3] Pure MHD   — clean IC           done")

    _, U2  = run_brio_wu(N=N, glm=False, div_b_pert=True)
    print("  [2/3] Pure MHD   — div-B perturbation  done")

    _, U3  = run_brio_wu(N=N, glm=True,  ch=2.0, div_b_pert=True)
    print("  [3/3] GLM  MHD   — div-B perturbation  done")

    rho1, p1, u1, By1 = extract(U1)
    rho2, p2, u2, By2 = extract(U2)
    rho3, p3, u3, By3 = extract(U3)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(
        r"Brio-Wu Shock Tube  ($\gamma = 2$,  $N = 400$,  $t = 0.1$)"
        "\n1-D Ideal MHD — HLLD Riemann Solver",
        fontsize=12)

    plot_vars = [
        ((rho1, rho2, rho3), r"Density $\rho$",          axes[0, 0]),
        ((p1,   p2,   p3  ), r"Gas Pressure $P$",         axes[0, 1]),
        ((u1,   u2,   u3  ), r"Velocity $u_x$",           axes[1, 0]),
        ((By1,  By2,  By3 ), r"Transverse Field $B_y$",   axes[1, 1]),
    ]

    for (d1, d2, d3), ylabel, ax in plot_vars:
        ax.plot(x, d1, color="k",        lw=1.8, ls="-",
                label="Pure MHD — clean IC (reference)")
        ax.plot(x, d2, color="tab:red",  lw=1.2, ls="--",
                label=r"Pure MHD + $\delta B_x$ error (uncleaned)")
        ax.plot(x, d3, color="tab:blue", lw=1.2, ls="-.",
                label=r"GLM  MHD + $\delta B_x$ error (cleaned)")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel(r"$x$", fontsize=11)

    plt.tight_layout()
    out = figdir / "brio_wu_comparison.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
