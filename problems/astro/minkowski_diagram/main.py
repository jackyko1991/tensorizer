"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # A handful of 2×2 matrices: everything stays on the CPU, in float64.
    v, distance, beta_view = params["v"], params["distance"], params["beta_view"]
    home, traveller = tensors.twin_world_lines(v, distance)
    T = float(home[-1, 0])
    grid = tensors.frame_grid(T + 2, -2 * T, 2 * T, step=1.0)

    sol = solve.solve(home, traveller, grid, beta_view)

    eta = equation.metric()
    reunion = traveller[-1] - traveller[0]
    s2 = equation.interval(reunion, eta)
    s2_boosted = equation.interval(equation.transform(sol["Lam"], reunion), eta)
    print(f"interval departure → reunion: Δs² = {s2:.6f} in S, {s2_boosted:.6f} in S′ (invariant)")
    b = torch.tensor(v, dtype=torch.float64)
    combined = equation.boost(b) @ equation.boost(b)
    print(
        f"Λ(v)Λ(v) = Λ({-combined[0, 1] / combined[0, 0]:.4f});  (v+v)/(1+v²) = {equation.velocity_addition(b, b):.4f}"
    )
    print(f"home twin ages {sol['tau_home'][-1]:.3f} yr, traveller ages {sol['tau_trav'][-1]:.3f} yr")
    print(f"expected traveller age 2L/v·√(1−v²) = {T * (1 - v**2) ** 0.5:.3f} yr")
    return plot.animate(sol, v, beta_view, out_dir / "minkowski_diagram.gif")


if __name__ == "__main__":
    cli(run, __file__)
