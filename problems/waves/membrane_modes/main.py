"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

# exact frequency ratios ω_m / ω_1 of the first six modes
SQUARE = [math.sqrt((m * m + n * n) / 2) for m, n in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]]
BESSEL_ZEROS = [2.40483, 3.83171, 3.83171, 5.13562, 5.13562, 5.52008]  # j_{0,1}, j_{1,1}, j_{1,1}, j_{2,1}, …
CIRCLE = [j / BESSEL_ZEROS[0] for j in BESSEL_ZEROS]


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n = params["n_grid"]
    X, Y, h = tensors.grid(n, dev)
    D = tensors.second_difference(n, h, dev)
    inside = tensors.drum_mask(X, Y, params["shape"])

    omega, shapes = solve.solve(D, inside, tensors.C)

    exact = SQUARE if params["shape"] == "square" else CIRCLE
    print(f"{int(inside.sum())} grid points inside the drum")
    for m in range(6):
        print(f"mode {m + 1}:  omega/omega_1 = {(omega[m] / omega[0]).item():.4f}   exact {exact[m]:.4f}")
    if params["shape"] == "square":
        print(f"fundamental omega_1 = {omega[0].item():.4f}   exact c*pi*sqrt(2)/L = {math.pi * math.sqrt(2):.4f}")
    else:
        print(f"fundamental omega_1 = {omega[0].item():.4f}   exact 2.40483 c/R = {2 * 2.40483:.4f}")
    flat = shapes.flatten(1)[:, inside].T  # (P, 6) back in the eigenvector layout
    error = (equation.overlap(flat) - torch.eye(6, dtype=flat.dtype, device=dev)).abs().max()
    print(f"orthonormality error max|phi^T phi - I| = {error:.1e}")
    return plot.animate(shapes, omega, inside, params["periods"], out_dir / "membrane_modes.gif")


if __name__ == "__main__":
    cli(run, __file__)
