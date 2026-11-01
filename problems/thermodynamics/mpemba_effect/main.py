"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import evolve, kl_divergence, mode_amplitudes


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T_bath, T_hot = params["T_bath"], params["T_hot"]
    m = solve.build(params["barrier"], T_bath, dev)  # u, W, π, λ, r, ℓ
    u, lam = m["u"], m["lam"]
    print(f"{len(u)} sites, bath T_b = {T_bath:.2f}")
    print(f"columns of W sum to {m['W'].sum(dim=0).abs().max():.1e};  λ_1 = {lam[0]:.1e}")
    print(f"slowest relaxation λ_2 = {lam[1]:.3e} (τ_2 = {-1 / lam[1]:.0f})")
    print(f"next mode          λ_3 = {lam[2]:.3e} (τ_3 = {-1 / lam[2]:.0f})")

    # a_2(T) on a fine log grid: one batched einsum over 300 temperatures
    T_grid = torch.logspace(math.log10(0.8 * T_bath), math.log10(100 * T_bath), 300, dtype=u.dtype, device=dev)
    a2_grid = solve.a2_curve(u, T_grid, m["left"])
    T_star = solve.mpemba_point(u, T_bath, m["left"])
    print(f"strong Mpemba point: a_2(T*) = 0 at T* = {T_star:.3f}")

    # the batch of starts: log-spaced temperatures, the Mpemba point, the warmest |a_2| below T*, and T_hot
    between = (T_grid > T_bath) & (T_grid < T_star)
    T_warm = T_grid[between][a2_grid[between].abs().argmax()].item() if between.any() else 1.5 * T_bath
    T0 = tensors.start_temperatures(T_bath, T_star, 9, dev)
    T0 = torch.cat([T0, torch.tensor([T_warm, T_hot], dtype=u.dtype, device=dev)]).unique()  # sorted (B,)
    p0 = tensors.boltzmann(u, T0)  # (B, N)
    a = mode_amplitudes(p0, m["left"])  # (B, N)

    times = torch.cat([torch.zeros(1), torch.logspace(-1, math.log10(8 / -lam[1].item()), 299)]).to(u)
    p = evolve(a, lam, m["right"], times)  # (T, B, N)
    kl = kl_divergence(p, m["pi"])  # (T, B)

    show = {k: int((T0 - v).abs().argmin()) for k, v in {"warm": T_warm, "T*": T_star, "hot": T_hot}.items()}
    for name, b in show.items():
        print(f"{name:>4}: T0 = {T0[b]:6.3f}  a_2 = {a[b, 1]:+.4f}  D(p‖π) = {kl[0, b]:.3f} → {kl[-1, b]:.1e}")
    w, h = show["warm"], show["hot"]
    if abs(a[h, 1]) < abs(a[w, 1]):
        cross = (kl[:, h] < kl[:, w]) & (times > 0)
        t_x = times[cross.nonzero()[0, 0]].item() if cross.any() else float("nan")
        print(f"Mpemba effect: the T0 = {T0[h]:.2f} start overtakes the T0 = {T0[w]:.2f} start at t ≈ {t_x:.0f}")
    else:
        print(f"no Mpemba effect between T0 = {T0[w]:.2f} and T0 = {T0[h]:.2f}: |a_2| is larger for the hotter start")
    return plot.animate(
        u, m["pi"], times, p, kl, T0, show, T_grid, a2_grid, T_bath, T_star, out_dir / "mpemba_effect.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
