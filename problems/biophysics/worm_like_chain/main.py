"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

N_FORCES = 12  # pulled forces from 0.02 to 10 pN, plus one free chain at F = 0
SHOW = [3, 6, 9, 12]  # forces whose first chain is drawn: about 0.06, 0.3, 2 and 10 pN
FIT_RANGE = (0.5, 4.0)  # pN: well stretched (fast to equilibrate) but far below K k_B T / a (discreteness)


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    P, L = params["persistence"], params["contour_length"]
    a = tensors.SEGMENT
    n_seg = int(round(L / a))
    K = tensors.stiffness_for(P, a)
    F = tensors.forces(N_FORCES, dev)  # (B,) with F[0] = 0
    f = tensors.reduced_forces(F, a)  # (B,)
    M = 2 if fast_mode() else params["chains"]
    sweeps = 40 if fast_mode() else params["sweeps"]
    t0 = tensors.relaxed_chains(len(F), M, n_seg, K, seed=1, device=dev)  # (B, M, N, 3)
    print(f"P = {P:g} nm, L = {L:g} nm → N = {n_seg} segments of a = {a} nm, bending stiffness K = κ/kT = {K:.2f}")
    print(f"{len(F)} forces × {M} chains × {sweeps} Monte Carlo sweeps")

    out = solve.simulate(t0, K, f, sweeps, n_records=60, show=SHOW, seed=0)
    z, z_err = out["z_mean"], out["z_err"]
    z_ms = equation.marko_siggia_extension(F[1:], P)
    z_fjc = equation.fjc_extension(F[1:], 2 * P)
    P_b = equation.persistence_from_force(F[1:], z[1:])  # (B-1,) Marko–Siggia solved for P at each force
    print("   F (pN)   z/L sim         WLC     FJC     P from this point")
    for b in range(1, len(F)):
        print(
            f"{F[b].item():8.3f}   {z[b].item():.3f} ± {z_err[b].item():.3f}   {z_ms[b - 1].item():.3f}   "
            f"{z_fjc[b - 1].item():.3f}   {P_b[b - 1].item():6.1f} nm"
        )
    in_fit = (F[1:] >= FIT_RANGE[0]) & (F[1:] <= FIT_RANGE[1])
    P_fit = torch.exp(torch.log(P_b[in_fit]).mean()).item()  # least squares in log F → geometric mean of P_b
    P_corr = equation.persistence_from_correlation(out["correlation"][: max(2, int(P / a))], a)
    print(
        f"free chain: ⟨t_i·t_i+1⟩ = {out['correlation'][1].item():.4f}  (exact {equation.segment_correlation(K):.4f})"
    )
    print(f"persistence length from the tangent correlation at F = 0: {P_corr:.1f} nm")
    print(f"persistence length from a Marko–Siggia fit, {FIT_RANGE[0]}–{FIT_RANGE[1]} pN: {P_fit:.1f} nm")
    print(f"input persistence length: {P:g} nm")
    return plot.animate(out, F, P, a, SHOW, FIT_RANGE, P_corr, out_dir / "worm_like_chain.gif")


if __name__ == "__main__":
    cli(run, __file__)
