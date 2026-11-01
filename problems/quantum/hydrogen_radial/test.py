"""Physics checks, written first (TDD): what any correct hydrogen code must satisfy.

Run with `pytest problems/quantum/hydrogen_radial/test.py`. Each test is a physical law, not an implementation
detail: the levels are E_n = −1/(2n²) with the l-degeneracy of the 1/r potential, radial functions have n − l − 1
nodes and every state is normalised.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def hydrogen(n_max: int = 3, n: int = 1500, r_max: float = 60.0):
    r, dr = tensors.radial_grid(n, r_max, CPU)
    ls = tensors.angular_momenta(n_max, CPU)
    H = equation.hamiltonian(tensors.second_derivative(n, dr, CPU), equation.effective_potential(r, ls))
    return r, dr, *solve.eigenstates(H, dr)


def test_energies_degeneracy_and_nodes():
    r, dr, energies, u = hydrogen()
    for n, ell in solve.orbitals(3):
        E, u_nl = solve.pick(energies, u, n, ell)
        assert abs(E + 0.5 / n**2) < 5e-4  # atomic units: E_n = −1/(2n²)
        assert solve.count_nodes(u_nl) == n - ell - 1  # node theorem: n − l − 1 radial nodes
        assert abs((u_nl**2).sum() * dr - 1) < 1e-10  # normalised radial probability
    # same n, different l: eigenvalues of different matrices, yet (nearly) equal
    assert abs(energies[0, 1] - energies[1, 0]) < 1e-4  # 2s vs 2p
    assert abs(energies[1, 1] - energies[2, 0]) < 1e-4  # 3p vs 3d


def test_angular_factor_is_normalised():
    # ∫ |Y_l0|² dΩ = 2π ∫ |Y_l0|² d(cos θ) = 1 for the Legendre factor used in the 2D slices
    cos_theta = torch.linspace(-1, 1, 20001, dtype=torch.float64)
    for ell in range(4):
        Y = equation.spherical_harmonic_m0(ell, cos_theta)
        assert abs(2 * math.pi * torch.trapezoid(Y**2, cos_theta) - 1) < 1e-6
