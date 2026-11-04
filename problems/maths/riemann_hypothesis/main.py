"""Main loop: parameters → Z(t) on a long grid → zeros → counting, statistics, explicit formula → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    t_max = 600.0 if fast_mode() else params["t_max"]

    # 1. thousands of zeros of Z(t) on the critical line (Riemann–Siegel formula, sign changes, bisection)
    t = tensors.t_grid(t_max, 0.02, dev)  # (T,)
    z = solve.z_on_grid(t)
    gammas = solve.find_zeros(t, z)
    print(f"Z(t) at {len(t):,} heights 10 ≤ t ≤ {t_max:.0f}: {len(gammas):,} sign changes = zeros on the line")
    for k, known in tensors.KNOWN_ZEROS.items():
        if k <= len(gammas):
            print(f"  γ_{k:<5d} = {gammas[k - 1].item():.9f}   published {known:.9f}")
    s = solve.s_function(gammas, t[::10])
    print(f"N(T) = {len(gammas)}; smooth count θ(T)/π + 1 = {equation.smooth_count(torch.tensor(t_max)).item():.2f}")
    print(f"S(t) = N(t) − θ(t)/π − 1: mean {s.mean().item():+.3f}, range [{s.min().item():.2f}, {s.max().item():.2f}]")

    # 2. Gram points and Gram's law
    n_gram = int(equation.smooth_count(torch.tensor(t_max)).item()) - 2
    gram = tensors.gram_points(n_gram, dev)
    gram = gram[gram < t_max]
    bad = solve.gram_violations(equation.riemann_siegel_z(gram))
    print(
        f"Gram points below T: {len(gram)}; Gram's law (−1)^n Z(g_n) > 0 fails at {len(bad)} "
        f"({100 * len(bad) / len(gram):.1f}%), first at n = {bad[0].item() if len(bad) else '–'}"
    )

    # 3. statistics of the unfolded zeros versus GUE random matrices
    x = solve.unfold(gammas)
    spacing = x.diff()
    k = int(spacing.argmin())
    print(
        f"unfolded spacings: mean {spacing.mean().item():.4f}, closest pair γ = {gammas[k].item():.4f}, "
        f"{gammas[k + 1].item():.4f} (spacing {spacing[k].item():.3f} of the mean)"
    )
    edges = torch.linspace(0, 3, 31, dtype=torch.float64, device=dev)
    r2 = solve.pair_correlation(x, edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    dev_r2 = (r2 - equation.montgomery_r2(mid)).abs().mean().item()
    print(f"pair correlation: mean |R₂ − (1 − (sin πu/πu)²)| = {dev_r2:.3f} on 0 ≤ u ≤ 3 (Montgomery–Odlyzko)")

    # 4. explicit formula: ψ(x) rebuilt from the zeros below each height
    x_max = params["x_max"]
    xs = torch.linspace(1.5, x_max, 1200, dtype=torch.float64, device=dev)
    psi_exact = tensors.chebyshev_psi(int(x_max), dev)  # (x_max + 1,)
    exact = psi_exact[xs.floor().long()]
    psi_k = solve.psi_from_zeros(xs, gammas)  # (K + 1, X)
    half = torch.arange(2.5, x_max, 1.0, dtype=torch.float64, device=dev)
    err = (solve.psi_from_zeros(half, gammas)[-1] - psi_exact[half.floor().long()]).abs()
    print(f"explicit formula with all {len(gammas)} zeros: max |ψ_K(x) − ψ(x)| = {err.max():.3f} at half-integers")

    heights = torch.logspace(math.log10(25), math.log10(t_max), 48).tolist()
    path = out_dir / "riemann_hypothesis.gif"
    return plot.animate(t, z, gammas, gram, bad, s, t[::10], spacing, x, edges, xs, exact, psi_k, heights, path)


if __name__ == "__main__":
    cli(run, __file__)
