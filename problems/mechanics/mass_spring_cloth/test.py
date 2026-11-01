"""Physics checks, written first (TDD): what any correct mass-spring cloth code must satisfy.

Run with `pytest problems/mechanics/mass_spring_cloth/test.py`. A hanging chain settles to Hooke's-law static
equilibrium, internal spring forces sum to zero (Newton's third law), and energy is conserved without damping and only
decreases with it.
"""

import torch

from . import equation, solve, tensors


def hanging_chain(n: int, spacing: float):
    """A vertical chain of n nodes hanging from node 0: positions, D, rest lengths, free mask."""
    x0 = torch.zeros(n, 3, dtype=torch.float64)
    x0[:, 2] = -spacing * torch.arange(n, dtype=torch.float64)
    edges = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=-1)
    free = torch.ones(n, dtype=torch.float64)
    free[0] = 0.0
    return x0, tensors.incidence(edges, n), tensors.rest_lengths(x0, edges), free


def test_chain_static_equilibrium_matches_analytic_stretch():
    n, m, k, g = 6, 0.1, 50.0, 9.81
    x0, D, rest, free = hanging_chain(n, 0.2)
    cloth = tensors.parameters(k, gamma=4.0, node_mass=m, g=g)
    _, traj = solve.solve(x0, D, rest, free, cloth, t_max=15.0, dt=0.005)
    x_end = traj[-1, 0]
    stretch = (x_end[1:] - x_end[:-1]).norm(dim=-1) - rest  # (n-1,)
    below = torch.arange(n - 1, 0, -1, dtype=torch.float64)  # spring e carries the n-1-e masses under it
    assert torch.allclose(stretch, below * m * g / k, rtol=1e-4)
    assert traj[-1, 1].abs().max() < 1e-5  # at rest


def test_einsum_forces_match_index_add():
    n = 5
    x0 = tensors.grid(n, 0.1)
    edges = tensors.springs(n)
    D, rest = tensors.incidence(edges, n * n), tensors.rest_lengths(x0, edges)
    x = x0 + 0.02 * torch.randn(n * n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    F1 = equation.spring_forces(x, D, rest, 100.0)
    F2 = equation.spring_forces_scatter(x, edges, rest, 100.0)
    assert torch.allclose(F1, F2, atol=1e-12)
    assert torch.allclose(F1.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)  # internal forces cancel


def test_energy_decreases_with_damping_and_is_conserved_without():
    n, m, k = 6, 0.01, 100.0
    x0 = tensors.grid(n, 0.1)
    edges = tensors.springs(n)
    D, rest, free = tensors.incidence(edges, n * n), tensors.rest_lengths(x0, edges), tensors.free_mask(n)

    damped = tensors.parameters(k, gamma=1.0, node_mass=m, g=9.81)
    _, traj = solve.solve(x0, D, rest, free, damped, t_max=1.5, dt=0.001)
    E = equation.energy(traj, D, rest, damped)
    assert (E.diff() <= 1e-9).all()  # never goes up
    assert E[-1] < E[0] - 0.05

    undamped = tensors.parameters(k, gamma=0.0, node_mass=m, g=9.81)
    _, traj = solve.solve(x0, D, rest, free, undamped, t_max=0.5, dt=0.0005)
    E = equation.energy(traj, D, rest, undamped)
    kinetic_max = 0.5 * m * torch.einsum("tai,tai->t", traj[:, 1], traj[:, 1]).max()
    assert (E - E[0]).abs().max() < 1e-4 * kinetic_max
