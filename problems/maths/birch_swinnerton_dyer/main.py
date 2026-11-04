"""Main loop: curves → a_p for every prime up to X → partial products, Sato–Tate, L(E, s) → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    x_max = 3000 if fast_mode() else params["x_max"]
    coeffs = tensors.curve_coefficients(dev)  # (C, 5)
    names = list(tensors.CURVES)

    # 1. point counts over F_p for every prime p ≤ X and every curve at once
    primes = tensors.primes_up_to(x_max, dev)  # (P,)
    a_p = equation.frobenius_traces(coeffs, primes)  # (C, P)
    hasse = (a_p.abs().double() / (2 * primes.double().sqrt())).max().item()
    print(f"{len(names)} curves × {len(primes):,} primes ≤ {x_max:,}: max |a_p| / 2√p = {hasse:.4f} (Hasse: ≤ 1)")
    print("a_p for p = 2, 3, 5, 7, 11, 13:")
    for name, row in zip(names, a_p[:, :6].tolist()):
        print(f"  {name:>9}: {row}")

    # 2. Birch and Swinnerton-Dyer's experiment: ∏_{p ≤ X} N_p / p ≈ C (log X)^r
    log_prod = solve.log_partial_products(coeffs, primes)  # (C, P)
    slope = solve.rank_estimate(log_prod, primes)

    # 3. the L-function at s = 1: root number, order of vanishing (analytic rank), leading coefficient
    lf = solve.l_function(coeffs, tensors.CONDUCTORS, n_max=3000)
    omega = equation.real_period(coeffs)
    print(f"{'curve':>9} {'rank':>4} {'slope':>6} {'ε':>3} {'ord L':>5} {'L^(r)(1)/r!':>12} {'Ω':>9}")
    for c, name in enumerate(names):
        print(
            f"{name:>9} {tensors.RANKS[c]:>4} {slope[c].item():6.2f} {lf['root_number'][c].item():+3d} "
            f"{lf['order'][c].item():>5} {lf['taylor'][c].item():12.6f} {omega[c].item():9.5f}"
        )
    print(f"11a1: L(E,1)/Ω = {lf['leading'][0].item() / omega[0].item():.10f}  (BSD: #Sha·c_11/#tors² = 1·5/5² = 1/5)")

    # Sato–Tate: the angles a_p = 2√p cos θ_p are distributed like (2/π) sin²θ
    x = (a_p.double() / (2 * primes.double().sqrt())).cpu()  # (C, P) in [−1, 1]
    moments = [(x**k).mean().item() for k in (2, 4)]
    print(f"Sato–Tate moments of a_p/2√p: E[x²] = {moments[0]:.3f} (1/4), E[x⁴] = {moments[1]:.3f} (1/8)")

    frames = torch.logspace(math.log10(60), math.log10(x_max), 45).tolist()
    return plot.animate(
        names, tensors.RANKS, primes.cpu(), x, log_prod.cpu(), lf, frames, out_dir / "birch_swinnerton_dyer.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
