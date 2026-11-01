"""Physics checks, written first (TDD): what any correct PET reconstruction code must satisfy.

Run with `pytest problems/medical/pet_mlem/test.py`. Each test is a physical law: annihilation photons are back-to-
back at 511 keV, MLEM conserves counts and never decreases the Poisson likelihood, and noiseless data converge to
the true activity.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_annihilation_photons_back_to_back_511_kev():
    g = torch.Generator().manual_seed(0)
    phi = 2 * math.pi * torch.rand(1000, generator=g)
    u, energy = equation.annihilation_photons(phi)
    cos = torch.einsum("ei,ei->e", u[:, 0], u[:, 1])
    assert torch.allclose(cos, torch.full_like(cos, -1.0))  # 180° apart
    assert torch.allclose(u.norm(dim=-1), torch.ones(1000, 2))
    assert torch.all(energy == 511.0)
    # both photons reach the ring, on a straight line through the annihilation point
    a = 10 * (torch.rand(1000, 2, generator=g) - 0.5)
    h = equation.ring_hits(a, u, 40.0)
    assert torch.allclose(h.norm(dim=-1), torch.full((1000, 2), 40.0), atol=1e-3)
    d1, d2 = h[:, 0] - a, h[:, 1] - a
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    assert cross.abs().max() < 1e-2 and (torch.einsum("ei,ei->e", d1, d2) < 0).all()


def test_positron_range_blurs_point_source():
    g = torch.Generator().manual_seed(1)
    sigma = 0.3  # cm
    a = equation.positron_range(torch.zeros(200000, 2), sigma, g)
    assert a.mean(0).abs().max() < 0.01
    assert torch.allclose(a.std(0), torch.full((2,), sigma), rtol=0.02)
    rms_radius = a.norm(dim=1).pow(2).mean().sqrt().item()  # 2D Gaussian: √2 σ
    assert abs(rms_radius - math.sqrt(2) * sigma) < 0.01


def _small_problem(n: int = 32):
    activity, mu = tensors.phantom(n, CPU)
    A = tensors.system_matrix(n, CPU)
    truth = activity.flatten() * 1e5 / activity.sum()
    return A, truth


def test_system_matrix_columns_are_probabilities():
    A, truth = _small_problem()
    n = 32
    centre = tensors.pixel_centres(n, CPU).view(-1, 2).norm(dim=1) < 12  # well inside the sinogram range
    col = A.sum(0)[centre]
    assert abs(col.mean().item() - 1.0) < 0.05


def test_mlem_conserves_counts_and_likelihood_increases():
    A, truth = _small_problem()
    g = torch.Generator().manual_seed(2)
    y = torch.poisson(A @ truth, generator=g)
    r = torch.zeros_like(y)
    images, ll, totals = solve.mlem(A, y, r, 30)
    assert torch.allclose(totals, torch.full_like(totals, y.sum().item()), rtol=1e-4)
    assert (images >= 0).all()
    assert (ll.diff() >= -1e-6 * ll.abs()[1:]).all()
    assert ll[-1] > ll[1] > ll[0]


def test_mlem_noiseless_converges_to_truth():
    A, truth = _small_problem()
    y = A @ truth  # exact, noise-free data
    images, ll, _ = solve.mlem(A, y, torch.zeros_like(y), 150)
    err = solve.relative_error(images, truth)
    assert err[150] < err[50] < err[10] < err[1]
    assert err[150] < 0.25 * err[0]


def test_lor_coordinates_of_a_known_line():
    # horizontal line y = 5 → normal along +y (θ = 90°), s = 5
    theta, s = equation.lor_coordinates(torch.tensor([[-40.0, 5.0]]), torch.tensor([[40.0, 5.0]]))
    assert abs(theta.item() - math.pi / 2) < 1e-6 and abs(s.item() - 5.0) < 1e-5
