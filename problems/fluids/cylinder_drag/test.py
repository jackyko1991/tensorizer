"""Physics checks, written first (TDD): what any correct lattice-Boltzmann cylinder code must satisfy.

Run with `pytest problems/fluids/cylinder_drag/test.py`. The equilibrium carries the right density and momentum,
collisions conserve mass in a closed box, and at Re = 100 the cylinder feels positive drag and sheds vortices with
a Strouhal number near 0.16.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_equilibrium_moments():
    e, w, _ = tensors.lattice(CPU)
    gen = torch.Generator().manual_seed(2)
    rho = 1 + 0.1 * torch.rand(6, 5, generator=gen)
    u = 0.1 * torch.randn(6, 5, 2, generator=gen)
    f_eq = equation.equilibrium(rho, u, e, w)
    assert torch.allclose(equation.density(f_eq), rho, atol=1e-6)  # zeroth moment: mass
    assert torch.allclose(equation.momentum(f_eq, e), rho[..., None] * u, atol=1e-6)  # first moment: momentum
    assert w.sum().item() == pytest.approx(1.0)


def test_conserves_mass_in_a_closed_box():
    nx, ny = 40, 30
    e, w, opp = tensors.lattice(CPU)
    solid = tensors.cylinder(nx, ny, 15, 15, 4, CPU)
    links = tensors.boundary_links(solid, e)
    f = tensors.initial_populations(nx, ny, 0.08, e, w, CPU)
    mass0 = f.sum().item()
    result = solve.simulate(f, solid, links, e, w, opp, 0.6, 0.08, 300, 100, open_ends=False)
    assert result["f"].sum().item() == pytest.approx(mass0, rel=1e-5)  # mass conservation (bounce-back walls)


@pytest.fixture(scope="module")
def cylinder_run():
    nx, ny, D, U = 180, 56, 10, 0.12
    e, w, opp = tensors.lattice(CPU)
    solid = tensors.cylinder(nx, ny, 40.0, ny / 2, D / 2, CPU)
    links = tensors.boundary_links(solid, e)
    f = tensors.initial_populations(nx, ny, U, e, w, CPU)
    tau = equation.viscosity_to_tau(U * D / 100.0)
    result = solve.simulate(f, solid, links, e, w, opp, tau, U, 5000, 1000)
    return solve.coefficients(result["force"], U, D), U, D


def test_drag_positive_and_right_order(cylinder_run):
    coeffs, _, _ = cylinder_run
    cd = coeffs[2500:, 0]
    assert (cd > 0).all()  # second law: the flow always pushes the cylinder downstream
    assert 1.0 < cd.mean().item() < 3.0


def test_lift_oscillates_with_zero_mean_and_sheds_periodically(cylinder_run):
    coeffs, U, D = cylinder_run
    cl = coeffs[2500:, 1]
    assert cl.abs().max().item() > 0.1  # the wake really oscillates
    assert abs(cl.mean().item()) < 0.1 * cl.abs().max().item()  # up–down symmetry: zero mean lift
    assert 0.12 < solve.strouhal(coeffs[:, 1], U, D) < 0.3  # von Kármán street: St ≈ 0.16 at Re = 100
