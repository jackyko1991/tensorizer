"""Main loop: parameters → tensors → Monte Carlo → measurements → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import strong_coupling_u, weak_coupling


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    group, dim, size = params["group"], params["dim"], params["size"] // 2 * 2  # even L for the checkerboard
    n_sweeps = 60 if fast_mode() else params["sweeps"]
    n_therm = n_sweeps // 5
    beta0 = params["beta"]
    betas = torch.tensor(tensors.coupling_grid(beta0, group), dtype=torch.float64, device=dev)  # (B,)
    i0 = int((betas - beta0).abs().argmin())
    n_frames = 8 if fast_mode() else 45

    out = solve.simulate(group, dim, size, betas, n_sweeps, n_therm, seed=0, measure_every=10, n_snapshots=n_frames)
    u = strong_coupling_u(betas, group)
    print(
        f"{group} Wilson action on a periodic {size}^{dim} lattice, {len(betas)} couplings in one batch, "
        f"{n_sweeps} Metropolis sweeps ({n_therm} for thermalisation)"
    )
    print(f"acceptance: {out['acceptance'].min():.2f} … {out['acceptance'].max():.2f}")

    # 1. average plaquette against strong and weak coupling
    plaq, plaq_err = solve.jackknife(out["plaquette"], lambda m: m)
    print(f"{'β':>6} {'⟨½TrP⟩':>14} {'strong u(β)':>12} {'weak 1−n/(Dβ)':>14}")
    for b in range(len(betas)):
        print(
            f"{betas[b]:6.3f} {plaq[b]:9.5f}±{plaq_err[b]:.5f} {u[b]:12.5f} {weak_coupling(betas[b], group, dim):14.5f}"
        )

    # 2. Wilson loops and Creutz ratios at the chosen β
    w, w_err = solve.jackknife(out["wilson"][:, i0], lambda m: m)  # (R, T)
    creutz = lambda m: -torch.log(m[1:, 1:] * m[:-1, :-1] / (m[1:, :-1] * m[:-1, 1:]))  # noqa: E731  χ(R, T)
    chi, chi_err = solve.jackknife(out["wilson"][:, i0].clamp_min(1e-12), creutz)
    print(
        f"β = {beta0}: W(R,T) = "
        + ", ".join(f"W({r},{t}) = {w[r - 1, t - 1]:.4f}" for r, t in [(1, 1), (2, 2), (3, 3)])
    )
    print(
        f"Creutz ratio χ(2,2) = {chi[0, 0]:.3f}±{chi_err[0, 0]:.3f} (string tension σa²);"
        f"  strong coupling −ln u = {-math.log(u[i0]):.3f}"
    )

    # 3. glueball: connected correlator of the spatial-plaquette operator between time slices
    t_max = 2  # Δ = 0, 1: beyond that the signal drowns in noise at these statistics
    if out["operator"] is not None:
        samples = solve.correlator_samples(out["operator"][:, i0], t_max)  # (S, Δ + 2)
        mass, mass_err = solve.jackknife(samples, solve.effective_mass)
        lead = -4 * math.log(u[i0])  # strong coupling: the smallest tube between the two plaquettes has 4Δ faces
        print(
            "glueball effective mass a·m_eff(Δ) = ln C(Δ)/C(Δ+1): "
            + ", ".join(f"Δ={d}: {m:.2f}±{e:.2f}" for d, (m, e) in enumerate(zip(mass.tolist(), mass_err.tolist())))
        )
        print(
            f"leading strong-coupling estimate a·m = −4 ln u(β) = {lead:.2f} (lattice estimate, not a continuum proof)"
        )

    running = solve.running(out, i0, n_therm, out["snapshot_sweeps"], t_max, measure_every=10)
    return plot.animate(group, dim, betas, i0, out, plaq_err, u, running, n_therm, out_dir / "yang_mills_mass_gap.gif")


if __name__ == "__main__":
    cli(run, __file__)
