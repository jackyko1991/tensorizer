"""Main loop: parameters → roots of unity and Gauss periods → nested radical → Richmond's construction → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    dtype = torch.float64 if params["precision"] == "float64" else torch.float32
    g = int(params["generator"])
    n_max = params["n_max"]

    # 1. roots of unity and the period tree, all five levels in one einsum
    zeta = tensors.roots_of_unity(17, torch.float64, dev)  # (17,) complex128
    order = tensors.cyclic_order(g)
    print(f"17th roots of unity: |Σ ζ^k| = {zeta.sum().abs():.1e}")
    print(f"generator g = {g}: exponents g^j mod 17 = {order.tolist()}")
    eta = equation.periods(zeta, order)  # (5, 16)
    worst = 0.0
    for m in range(4):
        n = 2**m
        const, coef = equation.product_in_periods(order, m)
        a, b = eta[m + 1, :n], eta[m + 1, n : 2 * n]
        prod = const.to(eta.dtype) + coef.to(eta.dtype) @ eta[m, :n]
        worst = max(worst, (a + b - eta[m, :n]).abs().max().item(), (a * b - prod).abs().max().item())
        terms = [f"{int(c) if c != 1 else ''}η{m},{k}" for k, c in enumerate(coef[0].tolist()) if c]
        terms = " + ".join(([str(int(const[0]))] if const[0] else []) + terms)
        print(
            f"level {m + 1}: {2 * n:>2} periods of {16 // (2 * n):>2} terms, e.g. "
            f"η{m + 1},0 + η{m + 1},{n} = η{m},0 and η{m + 1},0·η{m + 1},{n} = {terms}"
        )
    print(f"  all 15 quadratic relations hold to {worst:.1e}")
    print(f"  η1,0 = {eta[1, 0].real:.12f} = (−1 + √17)/2 = {(-1 + math.sqrt(17)) / 2:.12f}")

    # 2. the tower by square roots, and Gauss's closed formula
    tower = solve.periods_by_radicals(eta, order)
    print(f"tower of square roots reproduces every period to {(tower - eta).abs().max():.1e}")
    gauss = equation.gauss_formula(torch.float64)
    exact = math.cos(2 * math.pi / 17)
    print(
        f"Gauss: 16 cos(2π/17) = {16 * gauss:.15f}; cos(2π/17) = {gauss:.16f}, torch.cos error {abs(gauss - exact):.1e}"
    )
    try:
        import sympy as sp

        s17 = sp.sqrt(17)
        expr = (
            -1
            + s17
            + sp.sqrt(34 - 2 * s17)
            + 2 * sp.sqrt(17 + 3 * s17 - sp.sqrt(34 - 2 * s17) - 2 * sp.sqrt(34 + 2 * s17))
        ) / 16
        print(f"  sympy, 60 digits: |formula − cos(2π/17)| = {float(abs(sp.N(expr - sp.cos(2 * sp.pi / 17), 60))):.1e}")
    except ImportError:
        print("  (sympy not installed: symbolic check skipped)")

    # 3. Richmond's construction
    con = solve.richmond(dtype, dev)
    k = torch.arange(17, dtype=torch.float64, device=dev)
    exact_v = torch.stack([torch.cos(2 * math.pi * k / 17), torch.sin(2 * math.pi * k / 17)], 1)
    err = (con["vertices"].double() - exact_v).norm(dim=1)
    n_circ = sum(s["kind"] == "circle" for s in con["steps"])
    pts = con["points"]
    print(
        f"Richmond (1893) in {params['precision']}: {len(con['steps'])} steps "
        f"({n_circ} circles, {len(con['steps']) - n_circ} lines)"
    )
    print(f"  N3 = {pts['N3'][0]:.15f}  vs cos(6π/17)  = {math.cos(6 * math.pi / 17):.15f}")
    print(f"  N5 = {pts['N5'][0]:.15f} vs cos(10π/17) = {math.cos(10 * math.pi / 17):.15f}")
    closure = (pts["P0'"] - pts["P0"]).norm()
    print(f"  17 vertices: max error {err.max():.1e}; stepping back from P1 lands on P0 within {closure:.1e}")

    # 4. Gauss–Wantzel
    grid = solve.constructibility(n_max, dev)
    yes = grid["n"][grid["fermat"]]
    agree = bool((grid["phi_test"] == grid["fermat"]).all())
    print(f"constructible n-gons, n ≤ {n_max}: {len(yes)} of {n_max - 2}; φ(n) a power of 2 agrees: {agree}")
    print("  " + ", ".join(str(int(v)) for v in yes))
    print("  257-gon: Richelot (1832); 65537-gon: Hermes (about ten years' work, finished 1894)")

    data = {
        "con": con,
        "err": err,
        "eta": eta,
        "tower": tower,
        "order": order,
        "gauss": float(gauss),
        "grid": grid,
        "precision": params["precision"],
    }
    return plot.animate(data, out_dir / "gauss_heptadecagon.gif")


if __name__ == "__main__":
    cli(run, __file__)
