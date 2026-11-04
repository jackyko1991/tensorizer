"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_max = 1000 if fast_mode() else params["n_max"]
    n_terms = params["terms"]
    n_rad = 200  # Rademacher in float64: beyond n ≈ 240 the rounding of the large sinh/cosh terms reaches ±0.5

    # 1. exact values in int64
    exact = solve.partitions(tensors.N_EXACT, dev, torch.int64)
    print(f"p(100) = {int(exact[100]):,}   p(200) = {int(exact[200]):,}")
    print(f"p({tensors.N_EXACT}) = {int(exact[-1]):,}  (the largest p(n) below 2^63)")

    # 2. far beyond int64: exact residues modulo primes near 2^31, then the Chinese remainder theorem
    primes = tensors.crt_primes(n_max, dev)
    p = solve.partitions_float(n_max, dev)
    print(f"p({n_max:,}) ≈ {p[-1]:.6e}  (from residues modulo {len(primes)} primes near 2^31)")
    n = torch.arange(1, n_max + 1, device=dev)
    ratio = p[1:] / equation.hardy_ramanujan(n.double())
    c1 = 1 / equation.C + equation.C / 48
    print(f"{'n':>7} {'p(n) / Hardy–Ramanujan':>24} {'(1 − ratio)·√n':>15}")
    for m in [10, 100, 1000, 10000, 20000]:
        if m <= n_max:
            r = ratio[m - 1].item()
            print(f"{m:>7,} {r:>24.6f} {(1 - r) * m**0.5:>15.4f}")
    print(f"  the ratio → 1 like 1 − c/√n with c = 1/C + C/48 = {c1:.4f}, C = π√(2/3)")

    # 3. Ramanujan's congruences, checked on exact residues
    a, b, m = tensors.congruences(dev).T
    p_mod = solve.partitions(n_max, dev, torch.int64, m)  # (N + 1, M)
    held, checked = equation.congruence_holds(p_mod, a, b)
    for i in range(len(m)):
        status = "holds" if held[i] else "FAILS"
        print(f"  p({int(a[i])}k + {int(b[i])}) ≡ 0 (mod {int(m[i])}): {status} for k = 0 … {int(checked[i]) - 1}")
    print(
        f"Ramanujan noticed that p(5k+4) is always divisible by 5 — checked up to k = {int(checked[0]) - 1:,}"
        f" (p({5 * (int(checked[0]) - 1) + 4:,}) included)"
    )

    # 4. Rademacher's exact series
    n_r = torch.arange(1, n_rad + 1, device=dev)
    terms = equation.rademacher_terms(n_r, max(n_terms, 5))  # (N, K)
    partial = torch.cumsum(terms, dim=1)
    target = exact[1 : n_rad + 1].double()
    err = (partial - target[:, None]).abs()  # (N, K)
    used = partial[:, n_terms - 1]
    exact_hits = (used.round() == target).sum().item()
    print(
        f"Rademacher with {n_terms} terms: max |error| for n ≤ {n_rad} = {err[:, n_terms - 1].max():.3g}; "
        f"rounding gives the exact p(n) for {exact_hits}/{n_rad} values"
    )
    print(f"  p(200) from Rademacher: {used[199]:.4f} → {round(used[199].item()):,}")

    data = {
        "n": n,
        "ratio": ratio,
        "c1": c1,
        "n_rad": n_r,
        "err": err,
        "n_terms": n_terms,
        "p_mod": p_mod,
        "congruences": torch.stack([a, b, m], dim=1),
        "grid_n": min(n_max, params["grid_n"]),
    }
    return plot.animate(data, out_dir / "ramanujan_partitions.gif")


if __name__ == "__main__":
    cli(run, __file__)
