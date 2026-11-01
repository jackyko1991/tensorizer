"""Physics checks, written first (TDD): what any correct path tracer code must satisfy.

Run with `pytest problems/graphics/path_tracing/test.py`. Each test is a mathematical invariant, not an implementation
detail: the sampling pdf is normalised, a furnace enclosure reaches the rendering equation's analytic radiance L_e/(1 −
ρ), and Monte Carlo noise falls as 1/√N.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_cosine_sampling_pdf_is_normalised_and_matches_samples():
    theta = torch.linspace(0, math.pi / 2, 2001, dtype=torch.float64)
    integrand = equation.cosine_pdf(torch.cos(theta)) * torch.sin(theta) * 2 * math.pi  # ∫ p dω over the hemisphere
    assert abs(torch.trapezoid(integrand, theta).item() - 1) < 1e-6
    gen = torch.Generator().manual_seed(0)
    n = torch.nn.functional.normalize(torch.randn(200_000, 3, generator=gen), dim=1)
    w = equation.cosine_sample(n, gen)
    cos = (w * n).sum(-1)
    assert torch.allclose(w.norm(dim=1), torch.ones(len(w)), atol=1e-5)
    assert (cos >= -1e-6).all()
    assert abs(cos.mean().item() - 2 / 3) < 0.005  # E[cos θ] = ∫ cos² θ / π dω = 2/3
    assert abs((cos**2).mean().item() - 1 / 2) < 0.005


def test_furnace_white_enclosure_reaches_the_analytic_radiance():
    rho, le = 0.5, 1.0
    scene = tensors.furnace(CPU, rho, le)
    cam = tensors.camera(12, eye=(0.1, -0.2, 0.3))
    fixed, _ = solve.render(scene, cam, 2, torch.Generator().manual_seed(0), 6, roulette=False)
    expected = le * sum(rho**k for k in range(7))  # every path picks up exactly ρ per bounce: zero variance
    assert torch.allclose(fixed, torch.full_like(fixed, expected), atol=1e-4)
    rr, _ = solve.render(scene, cam, 64, torch.Generator().manual_seed(1), 2, roulette=True)
    assert abs(rr.mean().item() - le / (1 - rho)) < 0.03  # Russian roulette is unbiased: L_e/(1 − ρ)


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_noise_falls_as_one_over_root_spp():
    scene = tensors.cornell_box(CPU)
    cam = tensors.camera(32)
    ref, _ = solve.render(scene, cam, 1024, torch.Generator().manual_seed(9), 3, True)
    errs = []
    for spp in (4, 16, 64):
        img, _ = solve.render(scene, cam, spp, torch.Generator().manual_seed(spp), 3, True)
        errs.append(solve.rmse(img[None], ref).item())
    slope = solve.noise_slope(torch.tensor([4.0, 16.0, 64.0]), torch.tensor(errs))
    assert -0.7 < slope < -0.38, (errs, slope)  # three noisy points: a loose window around −1/2


def test_direct_light_only_has_no_colour_bleeding():
    scene = tensors.cornell_box(CPU)
    cam = tensors.camera(32)
    gi, direct = solve.render(scene, cam, 64, torch.Generator().manual_seed(3), 3, True)
    r, c = tensors.pixel_of(cam, [-0.8, 0.3, -1.0])  # white back wall next to the red wall
    g, d = gi[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1)), direct[r - 1 : r + 2, c - 1 : c + 2].mean((0, 1))
    assert abs(d[0] / d[1] - 1) < 1e-4  # a white wall lit by a white light stays white
    assert g[0] / g[1] > 1.1  # light bounced off the red wall tints it
