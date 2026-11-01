"""Main loop: a temporal interface (time reflection) and a periodic modulation (momentum gap), plus Floquet bands."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors

EPS0 = 1.0  # unmodulated permittivity (vacuum-like, so c = 1 before the jump)
DT = 0.05


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    alpha, omega, eps2 = params["alpha"], params["omega"], params["eps2"]
    t_max = 12.0 if fast_mode() else params["t_max"]
    t_jump = min(25.0, t_max / 3)
    every = max(1, round(t_max / DT / 200))  # keep about 200 snapshots
    x, k = tensors.grid(dev)
    k_gap = math.sqrt(EPS0) * omega / 2  # centre of the first momentum gap

    # 1. Temporal interface: ε jumps from EPS0 to eps2 at t_jump.
    n1, n2 = math.sqrt(EPS0), math.sqrt(eps2)
    D0, B0 = tensors.forward_pulse(x, k_gap, n1)
    times, u_jump = solve.interface(D0, B0, k, EPS0, eps2, t_jump, t_max, DT, every)
    D_jump = solve.to_field(u_jump)  # (S, X)
    T_f, T_b = equation.temporal_fresnel(n1, n2)
    D_fwd, D_bwd = equation.split_directions(D_jump[-1], solve.to_field(u_jump[-1], component=1), n2)
    print(f"temporal interface ε: {EPS0} → {eps2} at t = {t_jump:g}:  ω2/ω1 = n1/n2 = {n1 / n2:.3f}")
    print(f"  forward  D amplitude: Fresnel {T_f:+.3f}   simulated |D| {solve.envelope(D_fwd, k).max().item():.3f}")
    print(f"  backward D amplitude: Fresnel {T_b:+.3f}   simulated |D| {solve.envelope(D_bwd, k).max().item():.3f}")

    # 2. Periodic modulation ε(t) = EPS0 (1 + α cos Ωt), same pulse.
    def eps_periodic(t: float) -> float:
        return equation.periodic_eps(t, EPS0, alpha, omega)

    D0p, B0p = tensors.forward_pulse(x, k_gap, math.sqrt(eps_periodic(0.0)))
    times_p, u_per = solve.evolve(solve.to_modes(D0p, B0p), k, eps_periodic, 0.0, DT, round(t_max / DT), every)
    times_p = torch.tensor(times_p)
    D_per = solve.to_field(u_per)  # (S, X)
    spectra = u_per[..., 0].abs() / len(x)  # |D_k(t)|  (S, K)

    # 3. Floquet band structure: one monodromy matrix per k, eigenvalues in one batched call.
    period = 2 * math.pi / omega
    k_band = tensors.band_wavenumbers(2.4 * k_gap, 60 if fast_mode() else 480, dev)
    M = solve.monodromy(k_band, eps_periodic, period)  # (K, 2, 2)
    w_F = equation.floquet_frequencies(torch.linalg.eigvals(M), period)  # (K, 2)
    growth = w_F.imag.max(dim=-1).values  # (K,)
    lo, hi = equation.perturbative_gap(EPS0, alpha, omega)
    inside = k_band[growth > 1e-6]
    i_gap = torch.argmin((k - k_gap).abs()).item()
    measured = solve.growth_rate_measured(spectra, times_p, i_gap, t_max / 2)
    print(f"periodic modulation α = {alpha}, Ω = {omega}:  first momentum gap")
    if len(inside):
        first = inside[inside < 1.5 * k_gap]
        print(f"  Floquet:      k ∈ [{first.min().item():.4f}, {first.max().item():.4f}]   "
              f"max growth Im ω = {growth.max().item():.4f}")  # fmt: skip
    print(f"  perturbative: k ∈ [{lo:.4f}, {hi:.4f}]   max growth αΩ/8 = {alpha * omega / 8:.4f}")
    print(f"  simulated growth rate of the mode k = {k[i_gap].item():.4f}: {measured:.4f}")

    return plot.animate(
        x.cpu(), times, D_jump.cpu(), times_p, D_per.cpu(), spectra.cpu(), k.cpu(), k_band.cpu(), w_F.cpu(),
        (lo, hi), (T_f, T_b), (EPS0, eps2, t_jump), (alpha, omega), out_dir / "photonic_time_crystal.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
