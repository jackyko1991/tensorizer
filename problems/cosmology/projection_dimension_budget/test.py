"""Checks, written first (TDD): the dimension budget of the projection hypothesis must obey these laws.

Run with `pytest problems/cosmology/projection_dimension_budget/test.py`. The hypothesis is speculative; these tests
check the physics used to budget its dimensions. Gauss's law in 3 + n space dimensions fixes the force of a point
mass, compact dimensions turn it back into Newton's law beyond their size, the ADD relation
M_Pl^2 = 8 pi R^n M_D^(n+2) reproduces the numbers quoted by the Particle Data Group, and each experimental bound
excludes exactly the points it should.
"""

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def test_gauss_flux_in_3_plus_n_dimensions_is_constant():
    g = torch.Generator().manual_seed(0)
    radii = torch.tensor([0.3, 1.0, 4.0], dtype=torch.float64)
    for n in range(0, 5):
        flux = equation.gauss_flux(n, radii, points=4000, generator=g)  # area x mean outward field, autograd
        exact = equation.sphere_area(n + 2)  # |S^(n+2)| G_D M with G_D M = 1
        assert torch.allclose(flux, torch.full_like(flux, exact), rtol=1e-10)


def test_compact_dimensions_give_back_newton_far_away():
    for n, K in ((1, 4000), (2, 250)):
        r = torch.tensor([0.01, 0.02, 20.0, 30.0], dtype=torch.float64)  # units of R
        g = equation.image_force(r, n, K)  # sum over images of the mass around the n circles
        near = 1 / r[:2] ** (n + 2)  # G_D M / r^(n+2): gravity in 3 + n dimensions
        far = equation.g4_over_gd(n) / r[2:] ** 2  # G_4 M / r^2 with G_4 = |S^(n+2)| G_D / (4 pi (2 pi R)^n)
        assert torch.allclose(g[:2], near, rtol=1e-3)
        assert torch.allclose(g[2:], far, rtol=1e-3)


def test_add_relation_reproduces_the_pdg_numbers():
    n2 = torch.tensor(2.0, dtype=torch.float64)
    assert abs(equation.add_md_tev(n2, torch.tensor(30e-6)).item() - 4.0) < 0.05  # PDG: R < 30 um <=> M_D > 4.0 TeV
    n = torch.arange(1, 7, dtype=torch.float64)
    R = equation.add_radius_m(n, torch.tensor(1.0))  # PDG eq. (85.11), M_D = 1 TeV
    assert 0.9e12 < R[0] < 1.4e12  # n = 1: about 10^9 km (the Solar System)
    assert 0.4e-3 < R[1] < 0.6e-3  # n = 2: about 0.5 mm
    assert 0.08 < R[5] / tensors.HBARC_M / 1e3 < 0.2  # n = 6: about 0.1 MeV^-1
    M = torch.tensor([1.0, 7.0, 300.0], dtype=torch.float64)
    back = equation.add_md_tev(n[:, None], equation.add_radius_m(n[:, None], M))
    assert torch.allclose(back, M.expand(6, -1), rtol=1e-10)  # the relation inverts


def test_torus_yukawa_strength_makes_torsion_bounds_apply():
    n = torch.arange(1, 11, dtype=torch.float64)
    alpha = equation.torus_yukawa_alpha(n)
    assert alpha[1] == pytest.approx(16 / 3)  # PDG: 2-torus, alpha = 16/3
    assert (alpha >= 1).all()  # so |alpha| = 1 limits exclude R > lambda_max for every n


def test_bounds_are_applied_correctly_to_toy_points():
    b = tensors.bounds()
    toys = {  # (n, R in metres): bounds expected to exclude it
        (2, 1e-3): {"torsion", "lhc", "sn1987a", "neutron_star"},
        (2, 1e-12): set(),
        (3, 1e-11): {"neutron_star"},
        (3, 1e-13): set(),
        (5, 1e-12): {"lhc"},
        (8, 1e-3): {"torsion"},
        (8, 1e-6): set(),
        (0, 1.0): set(),  # no extra dimension: nothing to bound
    }
    n = torch.tensor([k[0] for k in toys], dtype=torch.float64)
    R = torch.tensor([k[1] for k in toys], dtype=torch.float64)
    masks = equation.exclusion_masks(n[:, None], R[:, None], b)  # {bound: (T, 1)}
    for i, expected in enumerate(toys.values()):
        got = {name for name, m in masks.items() if m[i, 0]}
        assert got == expected


def test_largest_allowed_radius_matches_the_map():
    b = tensors.bounds()
    n = torch.arange(0, 9, dtype=torch.float64)
    R = torch.logspace(-35, 0, 701, dtype=torch.float64)
    excluded = equation.excluded(n[:, None], R[None, :], b)  # (N, R)
    Rmax = equation.r_max(n, b)  # (N,)
    assert torch.isinf(Rmax[0]) and not excluded[0].any()
    for i in range(1, 9):  # allowed exactly below R_max(n)
        assert torch.equal(excluded[i], R > Rmax[i])
    assert Rmax[1] == pytest.approx(38.6e-6)  # n = 1: torsion balance (Lee 2020)
    assert Rmax[2] == pytest.approx(equation.add_radius_m(n[2], torch.tensor(1700.0)).item())  # neutron stars
    assert 1e-10 < Rmax[2] < 3e-10 and 1e-12 < Rmax[3] < 5e-12


def test_quantum_budget_counts():
    assert equation.race_dims(3, 4) == 7  # d observed + one hidden complex dimension per outcome (step 1)
    N = torch.arange(1, 11, dtype=torch.float64)
    assert torch.equal(equation.hilbert_log2_dim(N), N)  # D >= 2^N
    assert torch.equal(equation.spatial_reading_extra_dims(N), 2 ** (N + 1) - 3)  # real coordinates beyond 3


@pytest.mark.slow
def test_run_leaves_the_number_of_dimensions_open(tmp_path):
    from . import main

    params = {"n_max": 6, "R_points": 120, "M_star": 1.0, "qubits": 2}
    out = main.run(params, tmp_path)
    assert out.exists() and out.stat().st_size > 0
