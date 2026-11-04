"""Checks, written first (TDD): embeddings, curvature and fold exponents behind step 4 of the projection hypothesis.

Run with `pytest problems/cosmology/projection_singularity_fold/test.py`. The hypothesis (P4: a singularity is where
the projection of a flat higher-dimensional embedding folds) is speculative; these tests check the established
mathematics it is compared with. The Flamm paraboloid and Fronsdal's 6D embedding reproduce the Schwarzschild metric
(induced metric by autograd), the Kretschmann scalar is 12 r_s^2 / r^6, a fold caustic has the universal density
exponent -1/2 and a cusp -2/3, and a shell-shaped fold leaves the exterior field exactly Schwarzschild.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")
RS = 1.0  # Schwarzschild radius r_s = 2GM/c^2, the unit of length


def test_flamm_paraboloid_induces_the_schwarzschild_space_metric():
    x = tensors.flamm_points(RS, CPU)  # (B, 2) = (r, phi) outside the horizon
    g = equation.induced_metric(lambda q: equation.flamm_embedding(q, RS), x, torch.eye(3, dtype=torch.float64))
    r = x[:, 0]
    target = torch.zeros_like(g)
    target[:, 0, 0] = 1 / (1 - RS / r)  # dr^2 / (1 - r_s / r)
    target[:, 1, 1] = r**2  # r^2 dphi^2
    assert torch.allclose(g, target, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("region", ["exterior", "interior"])
def test_fronsdal_embedding_induces_schwarzschild(region):
    x = tensors.spacetime_points(RS, region, CPU)  # (B, 4) = (t, r, theta, phi)
    g = equation.induced_metric(lambda q: equation.fronsdal_embedding(q, RS), x, equation.ETA6)
    target = torch.vmap(lambda q: equation.schwarzschild_metric(q, RS))(x)
    assert torch.allclose(g, target, rtol=1e-9, atol=1e-9)


def test_fronsdal_in_kruskal_coordinates_has_full_rank_across_the_horizon():
    x = tensors.kruskal_points(RS, CPU)  # (B, 4) = (T, X, theta, phi), r from 0.2 r_s to 4 r_s
    r = equation.kruskal_r(x[:, 0], x[:, 1], RS)
    assert (r < RS).any() and (r > RS).any()
    g = equation.induced_metric(lambda q: equation.fronsdal_kruskal(q, RS), x, equation.ETA6)
    target = torch.vmap(lambda q: equation.kruskal_metric(q, RS))(x)
    assert torch.allclose(g, target, rtol=1e-8, atol=1e-9)
    sv = equation.jacobian_singular_values(lambda q: equation.fronsdal_kruskal(q, RS), x)  # (B, 4)
    assert sv.min() > 1e-3  # no fold at the horizon: the drop seen in (t, r) is a coordinate artefact
    xs = torch.tensor([[0.0, RS * (1 + 1e-8), 1.0, 0.5]], dtype=torch.float64)
    sv_s = equation.jacobian_singular_values(lambda q: equation.fronsdal_embedding(q, RS), xs)
    assert sv_s.min() < 1e-3  # d z / d t -> 0 at r = r_s in Schwarzschild coordinates


def test_fronsdal_embedding_runs_to_infinity_at_the_singularity():
    r = torch.tensor([1e-4, 1e-5, 1e-6], dtype=torch.float64)
    z2 = equation.fronsdal_z2(r, RS)
    assert torch.allclose(z2 * r.sqrt() / (-2 * RS**1.5), torch.ones(3, dtype=torch.float64), rtol=0.05)


def test_kretschmann_scalar_and_tidal_tensor():
    x = torch.cat([tensors.spacetime_points(RS, "exterior", CPU), tensors.spacetime_points(RS, "interior", CPU)])
    r = x[:, 1]
    K = equation.kretschmann(lambda q: equation.schwarzschild_metric(q, RS), x)
    assert torch.allclose(K, 12 * RS**2 / r**6, rtol=1e-8)
    xe = tensors.spacetime_points(RS, "exterior", CPU)
    tid = equation.static_tidal_eigenvalues(lambda q: equation.schwarzschild_metric(q, RS), xe)  # (B, 3) r, th, ph
    re = xe[:, 1:2]
    expected = torch.cat([RS / re**3, -RS / (2 * re**3), -RS / (2 * re**3)], 1)  # stretch, squeeze, squeeze
    assert torch.allclose(tid, expected, rtol=1e-8)


def test_fold_and_cusp_exponents_are_universal():
    n = torch.arange(2, 6)  # A_{n-1} catastrophes x = s^n: fold n = 2, cusp n = 3, ...
    alpha = equation.catastrophe_exponents(n, samples=2_000_000)
    assert torch.allclose(alpha, 1 / n.double() - 1, atol=0.02)
    assert abs(alpha[0].item() + 0.5) < 0.02  # fold: |x|^(-1/2)
    assert abs(alpha[1].item() + 2 / 3) < 0.02  # cusp: |x|^(-2/3)


def test_sphere_shadow_exponent_depends_only_on_the_codimension():
    gen = torch.Generator().manual_seed(4)
    for D in (6, 9):  # the ambient dimension D is open and does not enter
        kd = torch.tensor([0, 1, 2])  # sheet dimension k minus screen dimension d
        alpha = equation.sphere_rim_exponents(k=3 + kd, d=3, D=D, samples=400_000, generator=gen)
        assert torch.allclose(alpha, (kd.double() - 1) / 2, atol=0.05)  # fold -1/2 only when k = d


def test_rotating_sheet_folds_when_a_tan_alpha_exceeds_one():
    a = 2.0
    alpha = torch.linspace(0.05, 1.5, 30, dtype=torch.float64)
    n_fold = equation.sheet_fold_count(alpha, a)  # (A,) critical points of the projection per period
    assert ((n_fold > 0) == (a * torch.tan(alpha) > 1)).all()


def test_fold_shell_keeps_the_exterior_schwarzschild():
    R = 0.5 * RS
    r = torch.cat([torch.linspace(0.05, 0.499, 300), torch.linspace(0.51, 5.0, 100)]).double()
    toy = equation.fold_shell_field(r, R, RS)
    out = r > R
    assert torch.allclose(toy["mass"][out], torch.full_like(r[out], RS / 2), rtol=1e-9)  # M = r_s / 2 (G = c = 1)
    assert torch.allclose(toy["tidal_radial"][out], RS / r[out] ** 3, rtol=1e-9)  # GR / Newton exterior r_s / r^3
    assert toy["tidal_radial"][r < 0.1].abs().max() < 10  # finite at the centre, unlike r_s / r^3
    near = (R - r > 1e-3) & (R - r < 2e-2)
    slope = equation.loglog_slope(R - r[near], toy["density"][near])
    assert abs(slope + 0.5) < 0.02  # the projected density diverges like (R - r)^(-1/2) at the fold


@pytest.mark.slow
def test_run_accepts_parameters(tmp_path):
    from . import main

    params = {"fold_radius": 0.5, "sheet_amplitude": 2.0, "D": 6, "samples": 200_000}
    out = main.run(params, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_flamm_height_closed_form():
    assert math.isclose(equation.flamm_height(torch.tensor(2.0 * RS), RS).item(), 2.0, rel_tol=1e-12)
