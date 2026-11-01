"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    r = tensors.reaction(params["reaction"], dev)
    nu, dH, dS = r["nu"], r["dH"], r["dS"]
    T_list = r["T_curves"] + [params["T"]]  # the chosen temperature is the last curve
    P_grid = tensors.pressures(dev)  # (P,) bar
    T_b, P_b = tensors.condition_grid(T_list, P_grid)  # (B,), (B,) with B = nT · P
    n0 = r["n0"].expand(len(T_b), -1)  # (B, S) same feed everywhere

    # 1) equilibrium for every (T, P) at once
    lnK = equation.log_K(T_b, dH, dS)  # (B,)
    xi, history = solve.equilibrium_extent(n0, nu, P_b, lnK)  # (B,)
    n_eq = equation.moles(n0, nu, xi)  # (B, S)
    residual = (equation.log_Q(n_eq, P_b, nu) - lnK).abs().max().item()
    _, xi_max = equation.extent_bounds(n0, nu)
    conversion = (xi / xi_max).reshape(len(T_list), len(P_grid))  # (nT, P) fraction of the possible extent
    x_eq = (n_eq / n_eq.sum(-1, keepdim=True)).reshape(len(T_list), len(P_grid), -1)  # (nT, P, S)

    T0 = params["T"]
    K0 = math.exp(equation.log_K(torch.tensor(T0, dtype=tensors.DTYPE), dH, dS).item())
    print(f"{r['label']}   ΔH° ≈ {dH / 1e3:+.1f} kJ/mol   ΔS° ≈ {dS:+.1f} J/(mol K)   (approximate 298 K values)")
    print(f"T = {T0:.0f} K:  ΔG° = {(dH - T0 * dS) / 1e3:+.2f} kJ/mol   K = {K0:.4g}")
    print(f"bisection on {len(T_b)} conditions: max |ln Q − ln K| after 10/30/60 halvings = "
          f"{history[9]:.1e} / {history[29]:.1e} / {history[-1]:.1e}   (final check {residual:.1e})")  # fmt: skip

    print("\npressure (Le Chatelier): conversion ξ/ξ_max at the chosen T")
    for P in (0.1, 1.0, 10.0, 100.0):
        p = int(torch.argmin((P_grid - P).abs()))
        print(f"  P = {P_grid[p].item():6.1f} bar:  {conversion[-1, p].item():.4f}")

    p1 = int(torch.argmin((P_grid - params["P"]).abs()))
    print(f"\ntemperature (van 't Hoff, d ln K/d(1/T) = −ΔH°/R = {equation.van_t_hoff_slope(dH):.0f} K):"
          f" conversion at P = {P_grid[p1].item():.3g} bar")  # fmt: skip
    for t, T in enumerate(T_list[:-1]):
        print(f"  T = {T:5.0f} K:  K = {math.exp(lnK.reshape(len(T_list), -1)[t, 0].item()):.4g}"
              f"   conversion = {conversion[t, p1].item():.4f}")  # fmt: skip

    # 2) Le Chatelier with added product at the chosen (T, P)
    P1 = torch.tensor([params["P"]], device=dev, dtype=tensors.DTYPE)
    lnK1 = equation.log_K(torch.tensor([T0], device=dev, dtype=tensors.DTYPE), dH, dS)
    xi1, _ = solve.equilibrium_extent(r["n0"][None], nu, P1, lnK1)
    n1 = equation.moles(r["n0"][None], nu, xi1)  # (1, S)
    n_add, gap, n_new = solve.add_product_and_reequilibrate(n1, nu, P1, lnK1, params["add_product"])
    prod = r["species"][-1]
    print(f"\nadd {params['add_product']:.2f} mol {prod} at T = {T0:.0f} K, P = {params['P']:.3g} bar:")
    print("  species   " + "".join(f"{s:>9}" for s in r["species"]))
    for name, n in (("equilibrium", n1), ("just after", n_add), ("new equil.", n_new)):
        print(f"  {name:<11}" + "".join(f"{v:9.4f}" for v in n[0].tolist()) + "  mol")
    shift = n_new[0, -1].item() - n_add[0, -1].item()
    print(f"  Q/K just after adding = {math.exp(gap.item()):.3f}  →  net change {shift:+.4f} mol {prod}"
          " (Q > K: back towards reactants)")  # fmt: skip

    T_line = torch.linspace(min(T_list) - 10, max(T_list) + 10, 100, device=dev, dtype=tensors.DTYPE)
    return plot.animate(
        r, T_list, P_grid, conversion, x_eq, T_line, equation.log_K(T_line, dH, dS), params["P"],
        (n1[0], n_add[0], n_new[0]), out_dir / "chemical_equilibrium.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
