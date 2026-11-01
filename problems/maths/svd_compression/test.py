"""Physics checks, written first (TDD): what any correct SVD compression code must satisfy.

Run with `pytest problems/maths/svd_compression/test.py`. Each test is a mathematical invariant, not an implementation
detail: the full SVD reconstructs the image exactly and the rank-k error is the tail energy of the singular values
(Eckart–Young), which no other rank-k approximation beats.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def image():
    return tensors.test_image(64, CPU)


def test_eckart_young_equality(image):
    a = tensors.grey(image).double()
    u, s, v = equation.svd(a)
    ks = [1, 3, 10, 30]
    direct = solve.direct_errors(a, u, s, v, ks)
    tail = equation.tail_energy(s)
    assert torch.allclose(direct, tail[ks], rtol=1e-8, atol=1e-10)  # Eckart–Young: error = Σ_{i>k} σ_i²
    assert torch.allclose(equation.reconstruct(u, s, v, len(s)), a, atol=1e-10)  # full rank: exact reconstruction


def test_rank_k_error_falls_monotonically(image):
    a = tensors.grey(image).double()
    u, s, v = equation.svd(a)
    ks = list(range(1, len(s) + 1, 4))
    direct = solve.direct_errors(a, u, s, v, ks)
    assert (direct[1:] <= direct[:-1] + 1e-12).all()
    assert (equation.tail_energy(s).diff() <= 0).all()


def test_eckart_young_is_optimal_against_random_rank_k(image):
    a = tensors.grey(image).double()
    u, s, v = equation.svd(a)
    k, gen = 5, torch.Generator().manual_seed(0)
    best = equation.tail_energy(s)[k]
    for _ in range(5):  # project onto a random k-dimensional column space: never better than the SVD
        q, _ = torch.linalg.qr(torch.randn(a.shape[0], k, generator=gen, dtype=a.dtype))
        assert ((a - q @ (q.T @ a)) ** 2).sum() >= best


def test_colour_svd_is_batched_per_channel(image):
    u_c, s_c, v_c = equation.svd(image)
    for c in range(3):
        _, s, _ = equation.svd(image[c])
        assert torch.allclose(s_c[c], s, atol=1e-4)
    assert torch.allclose(equation.reconstruct(u_c, s_c, v_c, s_c.shape[-1]), image, atol=1e-4)
    ks, psnr, ratio = solve.quality_curve(s_c, 64, 64)
    assert (psnr.diff() >= 0).all() and (ratio.diff() < 0).all()
