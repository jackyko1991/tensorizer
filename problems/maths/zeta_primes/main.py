"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import gram_r, li, riemann_siegel_z, theta


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_zeros = 20 if fast_mode() else params["n_zeros"]

    # 1. zeros of Z(t) on the critical line
    t_max = tensors.height_for(n_zeros)
    n_terms = solve.n_terms_for(t_max)
    t = tensors.t_grid(t_max, 0.01, dev)  # (T,)
    z = solve.z_on_grid(t, n_terms)
    gammas = solve.find_zeros(t, z, n_terms)
    expected = int(theta(torch.tensor(t_max, dtype=torch.float64)) / math.pi + 1)
    print(f"Z(t) on {len(t)} points of 10 ≤ t ≤ {t_max:.1f} (Euler–Maclaurin, N = {n_terms})")
    print(f"sign changes found: {len(gammas)}; zero count N(T) ≈ θ(T)/π + 1 = {expected}")
    for g, known in zip(gammas[:5].tolist(), tensors.KNOWN_ZEROS):
        print(f"  γ = {g:.9f}   known {known:.9f}   difference {abs(g - known):.1e}")
    rs = riemann_siegel_z(gammas[:5])
    print(f"Riemann–Siegel (main sum + C₀) at those zeros: |Z| ≤ {rs.abs().max():.1e} instead of 0")
    gammas = gammas[:n_zeros]

    # 2. π(x) from the sieve, compared with x/ln x, li(x) and R(x)
    x_max = 10 ** (5 if fast_mode() else params["log10_x"])
    pi = tensors.prime_pi(tensors.sieve(x_max, dev))  # (x_max + 1,)
    zeta_int = tensors.zeta_at_integers(100, dev)
    print(f"{'x':>9} {'π(x)':>8} {'x/ln x':>10} {'li(x)':>10} {'R(x)':>10}")
    for e in range(1, int(math.log10(x_max)) + 1):
        x = torch.tensor([10.0**e], dtype=torch.float64, device=dev)
        print(
            f"{10**e:>9} {int(pi[10**e]):>8} {(x / torch.log(x)).item():>10.1f} "
            f"{li(x).item():>10.1f} {gram_r(x, zeta_int).item():>10.1f}"
        )
    x_log = torch.unique(torch.logspace(1, math.log10(x_max), 400, dtype=torch.float64, device=dev).floor())
    approx = {
        "x / ln x": x_log / torch.log(x_log),
        "li(x)": li(x_log),
        "R(x)": gram_r(x_log, zeta_int),
    }

    # 3. explicit formula: π(x) rebuilt from the first K zeros, K = 0 … n_zeros
    x_lin = torch.linspace(2, params["x_explicit"], 1500, dtype=torch.float64, device=dev)
    mu = tensors.mobius(64)
    pi_k = solve.prime_count_from_zeros(x_lin, gammas, mu)  # (K + 1, X)
    exact = pi[x_lin.floor().long()].double()
    for k in (0, min(10, n_zeros), n_zeros):
        err = (pi_k[k] - exact).abs()
        print(f"explicit formula with {k:3d} zeros: mean |π_K(x) − π(x)| = {err.mean():.3f} on 2 ≤ x ≤ {x_lin[-1]:.0f}")
    return plot.animate(t, z, gammas, x_log, pi[x_log.long()], approx, x_lin, exact, pi_k, out_dir / "zeta_primes.gif")


if __name__ == "__main__":
    cli(run, __file__)
