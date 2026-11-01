"""Physics checks, written first (TDD): what any correct electrostatics (Poisson) code must satisfy.

Run with `pytest problems/em/electrostatics/test.py`. The discrete Laplacian matches the analytic one, the solution
satisfies Poisson's equation ∇²φ = −ρ, and Gauss's law gives the enclosed charge as the flux of E.
"""

import math

import torch

from . import equation, solve, tensors


def test_discrete_laplacian_matches_analytic():
    n = 40
    x, h = tensors.grid(n)
    X, Y = torch.meshgrid(x, x, indexing="xy")  # rows = y, columns = x
    phi = torch.sin(math.pi * X) * torch.sin(math.pi * Y)  # vanishes on the walls
    lap = equation.apply_laplacian(equation.laplacian(n, h), phi)
    assert torch.allclose(lap, -2 * math.pi**2 * phi, atol=2e-2 * 2 * math.pi**2)


def test_poisson_solution_satisfies_equation():
    n = 30
    _, h = tensors.grid(n)
    rho = tensors.charge_density("quadrupole", n, 1.0)
    phi, _ = solve.solve(rho, h)
    lap = equation.apply_laplacian(equation.laplacian(n, h), phi)
    assert torch.allclose(lap, -rho, atol=1e-6 * rho.abs().max())  # Poisson: ∇²φ = −ρ


def test_gauss_law_flux_around_one_charge():
    n = 60
    _, h = tensors.grid(n)
    rho = tensors.charge_density("dipole", n, 1.0)  # +1 at x≈0.35, −1 at x≈0.65
    _, E = solve.solve(rho, h)
    a, b = (rho > 0).nonzero()[0].tolist()
    flux = solve.flux_through_square(E, h, a - 8, a + 8, b - 8, b + 8)  # box around the + charge only
    assert abs(flux.item() - 1.0) < 0.1  # Gauss's law: flux = enclosed charge
    whole = solve.flux_through_square(E, h, 2, n - 3, 2, n - 3)  # encloses both: net charge 0
    assert abs(whole.item()) < 0.1
