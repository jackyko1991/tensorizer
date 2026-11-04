"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    side = 61 if fast_mode() else params["side"]
    start = params["start"]
    x_max = 10 ** (5 if fast_mode() else params["log10_x"])
    g_max = 20000 if fast_mode() else params["goldbach_max"]
    k_max = 300 if fast_mode() else params["k_poly"]

    # 1. one sieve big enough for everything
    coef = tensors.quadratic_family(8, 60, dev)  # (F, 3): 4k² + bk + c
    q_max = int(equation.quadratic(coef, torch.tensor([k_max], device=dev)).max())
    n_max = max(x_max, g_max, q_max, start + side * side)
    is_prime = tensors.sieve(n_max, dev)
    primes = torch.nonzero(is_prime).flatten()
    pi = torch.cumsum(is_prime.long(), 0)
    print(f"sieve up to {n_max:,}: {len(primes):,} primes")
    print(f"{'x':>11} {'π(x)':>9} {'π₂(x)':>8} {'HL twins':>9} {'ratio':>6}")
    c2 = equation.twin_constant(primes[primes < 10**6])
    twins = equation.twin_count(is_prime[: x_max + 1])
    for e in range(3, int(math.log10(x_max)) + 1):
        pred = equation.twin_prediction(torch.tensor([10.0**e], dtype=torch.float64, device=dev), c2).item()
        print(f"{10**e:>11,} {int(pi[10**e]):>9,} {int(twins[10**e]):>8,} {pred:>9.0f} {twins[10**e] / pred:>6.3f}")
    print(f"twin-prime constant C₂ = {c2:.8f} (product over p < 10⁶)")

    # 2. the spirals
    n = torch.arange(start, start + side * side, device=dev)
    ulam = tensors.ulam_xy(n, start)  # (N, 2)
    sacks = tensors.sacks_xy(n)  # (N, 2)
    k = torch.arange(0, side // 2 + 1, device=dev)
    diag = torch.cat([k * k + k + start, k * k - k + start])  # the two halves of the main diagonal
    diag_frac = is_prime[diag].double().mean().item()
    print(
        f"Ulam spiral {side}×{side} from {start}: {is_prime[n].double().mean():.1%} primes overall, "
        f"{diag_frac:.1%} on the main diagonal k² + k + {start}"
    )
    if start == 41:
        print(
            "  Euler: k² + k + 41 is prime for k = 0 … 39 (Rabinowitsch: k² + k + p works for p = 2, 3, 5, 11, 17, 41)"
        )

    # 3. Conjecture F on every diagonal-type quadratic 4k² + bk + c
    small_primes = primes[primes < 10**5]
    fit = solve.conjecture_f(coef, is_prime, k_max, small_primes)
    alive = fit["constant"] > 0
    ratio = (fit["observed"][:, -1] / fit["predicted"][:, -1])[alive]
    print(
        f"Conjecture F on {int(alive.sum())} quadratics 4k² + bk + c (k ≤ {k_max}): observed / predicted primes "
        f"median {ratio.median():.3f}, middle 90% {ratio.quantile(0.05):.3f} … {ratio.quantile(0.95):.3f}"
    )
    pair = torch.stack([fit["observed"][alive, -1].double(), fit["constant"][alive]])
    print(f"  correlation of the prime count with C(f): {torch.corrcoef(pair)[0, 1]:.3f}")
    euler = solve.conjecture_f(torch.tensor([[1, 1, 41]], device=dev), is_prime, 2 * k_max, small_primes)
    print(
        f"  Euler n² + n + 41: C = {euler['constant'][0]:.5f} (= 2 × 3.31977…), {int(euler['observed'][0, -1])} primes "
        f"for 3 ≤ n ≤ {2 * k_max}, predicted {euler['predicted'][0, -1]:.0f}"
    )

    # 4. Goldbach's comet
    r = equation.goldbach_count(is_prime[: g_max + 1])
    n_even = torch.arange(6, g_max + 1, 2, device=dev)
    r_even = r[n_even].double()
    hl = equation.goldbach_prediction(n_even, primes, c2)
    upper = n_even > g_max // 2
    print(
        f"Goldbach: every even 4 ≤ N ≤ {g_max:,} has r(N) ≥ 1 (min {int(r_even.min())}); r / prediction for "
        f"N > {g_max // 2:,}: mean {(r_even / hl)[upper].mean():.4f}, std {(r_even / hl)[upper].std():.4f}"
    )

    # 5. gaps against Cramér's model
    gaps, norm = equation.normalised_gaps(is_prime[: x_max + 1])
    cramer = equation.cramer_model(x_max, 7, dev)
    _, norm_c = equation.normalised_gaps(cramer)
    twins_c = equation.twin_count(cramer)
    p_rec, rec_ratio = equation.ln_squared_ratio(is_prime[: x_max + 1])
    print(f"gaps up to {x_max:,}: mean g/ln p = {norm.mean():.3f} (Cramér model {norm_c.mean():.3f})")
    print(
        f"  record gaps: largest g/ln² p = {rec_ratio[p_rec > 100].max():.3f} (Cramér's conjecture ≤ 1, "
        f"Granville's 2e^(−γ) = {equation.GRANVILLE:.3f})"
    )
    print(
        f"  twins in Cramér's model: {int(twins_c[-1]):,} vs {int(twins[-1]):,} real: the model misses the factor "
        f"2C₂ = {2 * c2:.3f} (it ignores that p and p + 2 are both odd)"
    )

    data = {
        "side": side,
        "start": start,
        "ulam": ulam,
        "sacks": sacks,
        "prime_mask": is_prime[n],
        "diag_mask": torch.isin(n, diag),
        "fit": fit,
        "alive": alive,
        "x_max": x_max,
        "twins": twins,
        "twins_cramer": twins_c,
        "c2": c2,
        "n_even": n_even,
        "r_even": r_even,
        "hl": hl,
        "primes": primes[primes <= x_max],
        "norm": norm,
        "norm_cramer": norm_c,
        "cramer_primes": torch.nonzero(cramer).flatten(),
    }
    return plot.animate(data, out_dir / "prime_spirals.gif")


if __name__ == "__main__":
    cli(run, __file__)
