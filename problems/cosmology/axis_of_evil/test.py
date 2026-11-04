"""Checks written first (TDD): what any correct quadrupole–octopole alignment code must satisfy.

Run with `pytest problems/cosmology/axis_of_evil/test.py`. The harmonics must be orthonormal on the sphere, the
least-squares fit must give back a band-limited sky exactly, the statistic must not care how the sky is rotated,
isotropic random skies must give a uniform |cos| distribution, and a sky with a planted common axis must be detected.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pixels():
    dirs, T, w = tensors.load_sky_map()
    return dirs, w


def test_loader_reads_bundled_map(tmp_path):
    dirs, T, w = tensors.load_sky_map()
    assert dirs.shape == (3072, 3) and T.shape == (3072,)  # HEALPix N_side = 16
    assert torch.allclose(dirs.norm(dim=1), torch.ones(3072, dtype=dirs.dtype))
    assert dirs.sum(0).abs().max() < 1e-6 and w.sum().item() == pytest.approx(4 * math.pi)
    assert 30 < T.std().item() < 150  # CMB fluctuations of tens of μK
    path = tmp_path / "map.csv"  # an uploaded map: other column order and names
    path.write_text("temperature,lat,lon\n1.5,90,0\n-2.0,0,90\n", encoding="utf-8")
    d2, T2, _ = tensors.load_sky_map(path)
    assert torch.allclose(d2, torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=d2.dtype), atol=1e-12)
    assert T2.tolist() == [1.5, -2.0]


def test_harmonics_are_orthonormal():
    # Gauss–Legendre in cos θ × uniform φ integrates Y*Y exactly up to ℓ = 6
    x, wx = equation.gauss_legendre(16)
    phi = torch.arange(32, dtype=torch.float64) * 2 * math.pi / 32
    st = torch.sqrt(1 - x**2)
    n = torch.stack([st[:, None] * torch.cos(phi), st[:, None] * torch.sin(phi), x[:, None].expand(-1, 32)], -1)
    w = (wx[:, None] * 2 * math.pi / 32).expand(-1, 32).reshape(-1)
    Y = equation.spherical_harmonics(n.reshape(-1, 3), 6)  # (P, 49)
    gram = torch.einsum("p,pa,pb->ab", w.to(Y.dtype), Y.conj(), Y)
    assert torch.allclose(gram, torch.eye(49, dtype=Y.dtype), atol=1e-12)
    R = equation.real_harmonics(n.reshape(-1, 3), 6)
    assert torch.allclose(torch.einsum("p,pa,pb->ab", w, R, R), torch.eye(49, dtype=R.dtype), atol=1e-12)


def test_fit_recovers_band_limited_sky_and_reality_condition(pixels):
    dirs, w = pixels
    gen = torch.Generator().manual_seed(1)
    r = torch.randn(36, dtype=torch.float64, generator=gen)  # ℓ ≤ 5
    T = equation.real_harmonics(dirs, 5) @ r
    alm = solve.fit_alm(dirs, T, w, lmax=8)
    for ell in (2, 3):
        exact = equation.real_to_complex(r[ell**2 : (ell + 1) ** 2], ell)
        assert torch.allclose(alm[ell], exact, atol=1e-10)
        m = torch.arange(-ell, ell + 1)
        assert torch.allclose(alm[ell].flip(0), (-1.0) ** m * alm[ell].conj(), atol=1e-12)  # a_{ℓ,−m} = (−1)^m a*_{ℓm}


def test_angular_momentum_algebra():
    for ell in (2, 3):
        L = equation.angular_momentum(ell)  # (3, 2ℓ+1, 2ℓ+1)
        comm = L[0] @ L[1] - L[1] @ L[0]
        assert torch.allclose(comm, 1j * L[2], atol=1e-12)  # [L_x, L_y] = i L_z
        casimir = torch.einsum("imn,ink->mk", L, L)
        assert torch.allclose(casimir, ell * (ell + 1) * torch.eye(2 * ell + 1, dtype=L.dtype), atol=1e-12)


def test_statistic_is_rotation_invariant(pixels):
    dirs, w = pixels
    gen = torch.Generator().manual_seed(7)
    r = torch.randn(16, dtype=torch.float64, generator=gen)  # ℓ ≤ 3 sky
    R = tensors.random_rotation(gen)
    T = equation.real_harmonics(dirs, 3) @ r
    T_rot = equation.real_harmonics(dirs @ R, 3) @ r  # T'(n) = T(Rᵀ n): the same sky turned by R
    a, a_rot = solve.fit_alm(dirs, T, w, 6), solve.fit_alm(dirs, T_rot, w, 6)
    for ell in (2, 3):
        n = equation.principal_axis(equation.dispersion_tensor(a[ell], ell))
        n_rot = equation.principal_axis(equation.dispersion_tensor(a_rot[ell], ell))
        assert abs(torch.dot(R @ n, n_rot).item()) == pytest.approx(1.0, abs=1e-8)  # the axis turns with the sky
    s = solve.alignment_of(a)
    s_rot = solve.alignment_of(a_rot)
    assert s["cos"] == pytest.approx(s_rot["cos"], abs=1e-8)


def test_isotropic_skies_give_uniform_alignment():
    gen = torch.Generator().manual_seed(0)
    cos = solve.monte_carlo(40000, gen)
    assert cos.min() >= 0 and cos.max() <= 1
    sorted_cos = cos.sort().values
    ecdf = torch.arange(1, len(cos) + 1, dtype=torch.float64) / len(cos)
    assert (sorted_cos - ecdf).abs().max() < 0.012  # Kolmogorov–Smirnov distance to U(0, 1)
    assert cos.mean().item() == pytest.approx(0.5, abs=0.006)


def test_planted_common_axis_is_detected(pixels):
    dirs, w = pixels
    u = torch.nn.functional.normalize(torch.tensor([0.3, -0.5, 0.8], dtype=torch.float64), dim=0)
    T = tensors.planted_sky(dirs, u, amplitudes=(30.0, 25.0))
    a = solve.fit_alm(dirs, T, w, 10)
    s = solve.alignment_of(a)
    assert s["cos"] > 0.999
    assert abs(torch.dot(s["axis2"], u).item()) > 0.999
    cos_sims = solve.monte_carlo(20000, torch.Generator().manual_seed(2))
    assert solve.p_value(s["cos"], cos_sims) < 2e-3


def test_real_map_axes_are_well_defined():
    dirs, T, w = tensors.load_sky_map()
    s = solve.alignment_of(solve.fit_alm(dirs, T, w, 10))
    assert 0.0 <= s["cos"] <= 1.0
    assert s["gap2"] > 0 and s["gap3"] > 0  # the largest eigenvalue is not degenerate: the axis is unique
