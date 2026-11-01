"""Checks for the maths problems: Newton fractal, SVD image compression and PageRank."""

import pytest
import torch

from problems.maths.markov_pagerank import equation as pr_eq
from problems.maths.markov_pagerank import solve as pr_solve
from problems.maths.markov_pagerank import tensors as pr_tensors
from problems.maths.newton_fractal import equation as nf_eq
from problems.maths.newton_fractal import solve as nf_solve
from problems.maths.newton_fractal import tensors as nf_tensors
from problems.maths.svd_compression import equation as svd_eq
from problems.maths.svd_compression import solve as svd_solve
from problems.maths.svd_compression import tensors as svd_tensors

CPU = torch.device("cpu")


# ---------------------------------------------------------------- Newton fractal


def test_roots_are_roots():
    for name in nf_tensors.POLYNOMIALS:
        c = nf_tensors.coefficients(name, CPU, torch.complex128)
        r = nf_tensors.roots(c)
        assert len(r) == len(c) - 1
        assert nf_eq.poly(r, c).abs().max() < 1e-12


def test_derivative_coefficients():
    c = torch.tensor([2.0, -2.0, 0.0, 1.0], dtype=torch.complex128)  # z³ − 2z + 2
    z = torch.tensor([0.3 + 0.7j, -1.2 + 0.1j], dtype=torch.complex128)
    assert torch.allclose(nf_eq.poly(z, nf_eq.derivative(c)), 3 * z**2 - 2)


@pytest.mark.parametrize("name", ["z3_minus_1", "z4_minus_1", "z3_minus_2z_plus_2"])
def test_pixels_near_each_root_converge_to_it(name):
    c = nf_tensors.coefficients(name, CPU)
    roots = nf_tensors.roots(c)
    for j, r in enumerate(roots.tolist()):
        z0 = nf_tensors.grid(complex(r), 0.1, 16, CPU)  # a small patch around root j
        labels, conv = nf_solve.iterate(z0, c, roots, 15, 1e-4)
        assert (labels[-1] == j).all()
        assert conv.max() <= 6  # quadratic convergence: only a few steps from close by


@pytest.mark.parametrize(("name", "expected"), [("z3_minus_1", 1.0), ("z4_minus_1", 1.5)])
def test_newton_error_ratio_tends_to_p2_over_2p1(name, expected):
    """e_{k+1}/e_k² → |p''(r)/(2p'(r))|: for z^n − 1 at r = 1 this is (n − 1)/2."""
    c = nf_tensors.coefficients(name, CPU)
    errors, root = nf_solve.error_trace(1.3 + 0.2j, c)
    assert abs(root.item() - 1) < 1e-12
    limit = nf_solve.newton_constant(root, c)
    assert limit == pytest.approx(expected, rel=1e-10)
    k = int(((errors > 1e-7) & (errors < 1e-3)).nonzero()[-1])  # close to the root but above round-off
    assert (errors[k + 1] / errors[k] ** 2).item() == pytest.approx(limit, rel=0.01)


def test_basins_have_threefold_symmetry_and_wada_boundary():
    c = nf_tensors.coefficients("z3_minus_1", CPU)
    roots = nf_tensors.roots(c)
    labels, _ = nf_solve.iterate(nf_tensors.grid(0j, 1.5, 120, CPU), c, roots, 40, 1e-4)
    share = torch.stack([(labels[-1] == j).float().mean() for j in range(3)])
    assert share.sum() > 0.99 and share.min() > 0.3  # every basin takes about a third of a centred square
    _, wada = nf_solve.boundary_stats(labels[-1].long(), 3)
    assert wada > 0.4  # most boundary pixels touch all three basins


# ---------------------------------------------------------------- SVD compression


@pytest.fixture(scope="module")
def image():
    return svd_tensors.test_image(64, CPU)


def test_eckart_young_equality(image):
    a = svd_tensors.grey(image).double()
    u, s, v = svd_eq.svd(a)
    ks = [1, 3, 10, 30]
    direct = svd_solve.direct_errors(a, u, s, v, ks)
    tail = svd_eq.tail_energy(s)
    assert torch.allclose(direct, tail[ks], rtol=1e-8, atol=1e-10)
    assert torch.allclose(svd_eq.reconstruct(u, s, v, len(s)), a, atol=1e-10)


def test_rank_k_error_falls_monotonically(image):
    a = svd_tensors.grey(image).double()
    u, s, v = svd_eq.svd(a)
    ks = list(range(1, len(s) + 1, 4))
    direct = svd_solve.direct_errors(a, u, s, v, ks)
    assert (direct[1:] <= direct[:-1] + 1e-12).all()
    assert (svd_eq.tail_energy(s).diff() <= 0).all()


def test_eckart_young_is_optimal_against_random_rank_k(image):
    a = svd_tensors.grey(image).double()
    u, s, v = svd_eq.svd(a)
    k, gen = 5, torch.Generator().manual_seed(0)
    best = svd_eq.tail_energy(s)[k]
    for _ in range(5):  # project onto a random k-dimensional column space: never better than the SVD
        q, _ = torch.linalg.qr(torch.randn(a.shape[0], k, generator=gen, dtype=a.dtype))
        assert ((a - q @ (q.T @ a)) ** 2).sum() >= best


def test_colour_svd_is_batched_per_channel(image):
    u_c, s_c, v_c = svd_eq.svd(image)
    for c in range(3):
        _, s, _ = svd_eq.svd(image[c])
        assert torch.allclose(s_c[c], s, atol=1e-4)
    assert torch.allclose(svd_eq.reconstruct(u_c, s_c, v_c, s_c.shape[-1]), image, atol=1e-4)
    ks, psnr, ratio = svd_solve.quality_curve(s_c, 64, 64)
    assert (psnr.diff() >= 0).all() and (ratio.diff() < 0).all()


# ---------------------------------------------------------------- PageRank


@pytest.fixture(scope="module")
def chain():
    links, _, _ = pr_tensors.web_graph(25, 3, 1, CPU)
    links = links.double()
    s = pr_eq.transition(links)
    ds = torch.tensor([0.5, 0.85, 0.95], dtype=s.dtype)
    g = pr_eq.google(s, ds)
    history, residual = pr_solve.power_iteration(g, 200)
    return links, s, ds, g, history, residual


def test_transition_is_column_stochastic_with_dangling_pages(chain):
    links, s, _, g, _, _ = chain
    assert (links.sum(0) == 0).sum() == 3
    assert torch.allclose(s.sum(0), torch.ones(25, dtype=s.dtype))
    assert torch.allclose(g.sum(1), torch.ones(3, 25, dtype=g.dtype))


def test_pagerank_sums_to_one_and_is_non_negative(chain):
    _, _, ds, _, history, _ = chain
    pi = history[-1]  # (B, N)
    assert torch.allclose(pi.sum(-1), torch.ones(3, dtype=pi.dtype), atol=1e-12)
    assert (pi >= (1 - ds[:, None]) / 25 - 1e-12).all()  # teleporting alone gives every page (1 − d)/N


def test_power_iteration_matches_eigenvector(chain):
    _, _, _, g, history, _ = chain
    for b in range(3):
        pi_eig, mags = pr_solve.eig_stationary(g[b])
        assert mags[0].item() == pytest.approx(1.0, abs=1e-12)
        assert torch.allclose(history[-1, b], pi_eig, atol=1e-10)


def test_convergence_factor_is_lambda2_and_at_most_d(chain):
    _, _, ds, g, _, residual = chain
    factor = pr_solve.convergence_factor(residual)
    for b, d in enumerate(ds.tolist()):
        _, mags = pr_solve.eig_stationary(g[b])
        assert mags[1].item() <= d + 1e-9
        assert factor[b].item() <= d + 0.02
        assert factor[b].item() == pytest.approx(mags[1].item(), abs=0.05)


def test_monte_carlo_surfers_match_power_iteration(chain):
    _, s, ds, _, history, _ = chain
    occupancy = pr_solve.monte_carlo(s, 0.85, 5000, 60, seed=0)
    assert torch.allclose(occupancy.sum(-1), torch.ones(61))
    estimate = occupancy[30:].mean(0).double()
    pi = history[-1, 1]
    assert torch.corrcoef(torch.stack([estimate, pi]))[0, 1] > 0.99
    assert pr_solve.spearman(estimate, pi) > 0.9
    assert (estimate - pi).abs().sum() < 0.05
