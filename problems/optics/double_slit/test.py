"""Physics checks, written first (TDD): what any correct double-slit code must satisfy.

Run with `pytest problems/optics/double_slit/test.py`. The Laplacian is second-order accurate, the CFL stability limit
is enforced, and the fringes are spaced λL/d with a bright centre.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_fd_laplacian_matches_analytic():
    h = 0.1
    y, x = torch.meshgrid(
        torch.arange(100, dtype=torch.float64) * h, torch.arange(120, dtype=torch.float64) * h, indexing="ij"
    )
    kx, ky = 0.7, 1.3
    u = torch.sin(kx * x) * torch.sin(ky * y)
    lap = equation.laplacian(u, h)
    exact = -(kx**2 + ky**2) * u
    truncation = h**2 / 12 * (kx**4 + ky**4)  # leading error term of the 5-point stencil
    assert (lap - exact)[1:-1, 1:-1].abs().max() <= 1.01 * truncation  # edges use zero padding


def test_cfl_condition_respected_and_enforced():
    assert equation.courant(1.0, 0.5, 1.0) <= equation.CFL_LIMIT
    setup = tensors.setup(48.0, 6.0, 150.0, CPU)
    try:
        solve.solve(setup, 12.0, t_max=10.0, dt=0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("dt above the CFL limit must be rejected")


def test_fringe_spacing():
    lam, d, dist = 12.0, 48.0, 150.0
    setup = tensors.setup(d, 6.0, dist, CPU)
    _, _, intensity = solve.solve(setup, lam, t_max=450.0, save_every=100)
    screen = intensity[-1]  # (Ny,)
    y = torch.arange(tensors.NY) - tensors.NY / 2
    centre = (y.abs() < 10).nonzero().flatten()
    first = ((y > 20) & (y < 60)).nonzero().flatten()
    y0 = y[centre[screen[centre].argmax()]]
    y1 = y[first[screen[first].argmax()]]
    assert abs(y0) <= 3  # bright central fringe on the axis
    assert math.isclose((y1 - y0).item(), lam * dist / d, rel_tol=0.15)  # ≈ λL/d
