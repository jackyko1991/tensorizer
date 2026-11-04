"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import analyticity_strip, bkm_integral


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = 16 if fast_mode() else params["n"] // 2 * 2  # even grid
    nu_list = [0.0, params["nu"], 2 * params["nu"]]  # Euler and two viscous runs, solved as one batch
    nu = torch.tensor(nu_list, dtype=torch.float64, device=dev)
    t_max = 1.0 if fast_mode() else params["t_max"]

    k, k2, mask = tensors.wavenumbers(n, dev)  # (3, N, N, M), (N, N, M), (N, N, M)
    u0 = tensors.taylor_green(n, dev)[None].repeat(len(nu_list), 1, 1, 1, 1)  # (B, 3, N, N, N)
    dt = 1.6 / n  # CFL: |u| ≤ 1, Δx = 2π/N
    out = solve.simulate(u0, nu, k, k2, mask, t_max, dt, n_save=8 if fast_mode() else 60)
    t, e, z, wmax = out["t"], out["energy"], out["enstrophy"], out["max_vorticity"]
    bkm = bkm_integral(t, wmax)  # (T, B)

    k_max = n // 3  # largest wavenumber kept by the 2/3 rule
    delta = torch.stack([analyticity_strip(s, 3, k_max - 1) for s in out["spectra"]])  # (S, B)
    resolved = delta[:, 0] * k_max >= 1.0  # a common criterion: the strip is wider than one shortest wave
    t_lost = out["t_save"][~resolved][0].item() if (~resolved).any() else None

    de_dt = (e[2:] - e[:-2]) / (t[2:] - t[:-2])[:, None]
    err = (de_dt + 2 * nu * z[1:-1]).abs().max(dim=0).values / (2 * nu * z).max(dim=0).values.clamp_min(1e-30)
    print(f"Taylor–Green vortex on {n}³ points (2/3 dealiasing, k_max = {k_max}), RK4 with dt = {dt:.4f}")
    for b, v in enumerate(nu_list):
        name = "Euler (ν = 0)" if v == 0 else f"ν = {v:.4f} (Re = {1 / v:.0f})"
        z_peak = z[:, b].max().item()
        t_peak = t[z[:, b].argmax()].item()
        print(
            f"  {name:<22} max Ω = {z_peak:7.3f} at t = {t_peak:4.2f},  ‖ω‖_∞(end) = {wmax[-1, b]:7.3f},  "
            f"BKM ∫‖ω‖_∞ dt = {bkm[-1, b]:6.2f},  E(end)/E₀ = {e[-1, b] / e[0, b]:.4f}"
        )
    print(
        f"energy budget: Euler |E/E₀ − 1| = {abs(e[-1, 0] / e[0, 0] - 1):.1e};  "
        f"viscous max|dE/dt + 2νΩ| / max 2νΩ = {err[1]:.1e}, {err[2]:.1e}"
    )
    ts = out["t_save"]
    picks = [int(torch.searchsorted(ts, torch.tensor(float(c), dtype=ts.dtype))) for c in range(1, int(t_max) + 1)]
    strip = ", ".join(f"δ({ts[i]:.1f}) = {delta[i, 0]:.3f}" for i in picks if i < len(ts))
    print(f"analyticity strip of the Euler run (shrinking, but no sign of reaching 0 while resolved): {strip}")
    viscous_ok = (delta[1:, 1:] * k_max).min().item()
    print(f"resolution of the viscous runs: min δ·k_max = {viscous_ok:.2f} (≥ 1: resolved throughout)")
    if t_lost is not None:
        print(f"Euler run under-resolved after t ≈ {t_lost:.2f} (δ·k_max < 1): later growth is a grid artefact")
    pvd = plot.export_vtk(out, t_lost, out_dir)  # interactive 3D results: the |ω| iso-surface over time
    size = sum(p.stat().st_size for p in pvd.parent.iterdir()) / 1e6
    print(f"interactive results: {pvd.relative_to(out_dir)} and its layers ({size:.1f} MB)")
    return plot.animate(out, nu_list, bkm, delta, k_max, t_lost, out_dir / "navier_stokes_blowup.gif")


if __name__ == "__main__":
    cli(run, __file__)
