"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    GM = tensors.GM
    y0 = tensors.initial_state(params["r0"], params["ring_radius"], params["n_particles"], dev)
    sol = solve.solve(y0, GM, params["r_stop"], params["n_steps"])

    # the tidal tensor at the start, from the formula and from autograd
    x0 = y0[:1, 0]  # (1, 3)
    T_formula = equation.tidal_tensor(x0, GM)[0]
    T_autograd = equation.tidal_tensor_autograd(x0, GM)[0]
    r0 = float(x0.norm())
    print(f"T_ij at r = {r0:g} (units GM/r³):\n{(T_formula * r0**3 / GM).cpu().numpy().round(6)}")
    print(f"autograd Hessian vs formula: max difference {(T_autograd - T_formula).abs().max():.1e}")
    print(f"trace T = {torch.trace(T_formula):.1e}   eigenvalues · r³/GM = {torch.linalg.eigvalsh(T_formula) * r0**3}")

    r_t = equation.tidal_radius(params["ring_radius"], params["mass_ratio"])
    along, across = solve.stretch(sol["xi_exact"])
    s = params["ring_radius"]
    print(f"tidal radius for a body with m/M = {params['mass_ratio']:g} and radius {s:g}: r_t = {r_t:.2f} GM/c²")
    print(f"at r = {params['r_stop']:g}: stretched ×{along[-1] / s:.2f} radially, squeezed ×{across[-1] / s:.2f}")
    return plot.animate(sol, GM, r_t, out_dir / "tidal_tensor.gif")


if __name__ == "__main__":
    cli(run, __file__)
