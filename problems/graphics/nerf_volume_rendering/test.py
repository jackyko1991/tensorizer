"""Physics checks, written first (TDD): what any correct NeRF volume renderer code must satisfy.

Run with `pytest problems/graphics/nerf_volume_rendering/test.py`. Each test is a mathematical invariant, not an
implementation detail: a uniform slab absorbs 1 − e^{−σL} (Beer–Lambert), transmittance never increases, and positional
encoding lets the MLP fit high frequencies.
"""

import math

import torch

from . import equation, tensors

CPU = torch.device("cpu")


def test_volume_rendering_weights_for_uniform_slab():
    sigma0, a, b = 2.0, 1.0, 2.0
    t = torch.arange(0, 400, dtype=torch.float64)[None, :] * 0.01  # samples every 0.01 on [0, 4)
    sigma = torch.where((t >= a - 1e-9) & (t < b - 1e-9), sigma0, 0.0)  # density σ0 on the slab [a, b)
    w, T = equation.render_weights(sigma, t, far=4.0)
    assert math.isclose(w.sum().item(), 1 - math.exp(-sigma0 * (b - a)), rel_tol=1e-9)  # exact for piecewise σ
    expected_T = torch.exp(-sigma0 * (t - a).clamp(0, b - a))
    assert torch.allclose(T, expected_T, atol=1e-9)
    assert (T[:, 1:] <= T[:, :-1]).all()  # transmittance never increases
    # the expected depth inside the slab: E[t] = a + 1/σ0 - (b - a) e^{-σ0(b-a)} / (1 - e^{-σ0(b-a)}), up to O(δ)
    ea = math.exp(-sigma0 * (b - a))
    mean_t = (w * t).sum().item() / w.sum().item()
    assert abs(mean_t - (a + 1 / sigma0 - (b - a) * ea / (1 - ea))) < 0.01


def test_positional_encoding_improves_high_frequency_fit():
    x = torch.linspace(-1, 1, 256)[:, None]
    target = torch.sin(6 * math.pi * x) * torch.cos(2 * math.pi * x)  # (N, 1)
    pts = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)  # along the x axis
    errors = {}
    for L in (0, 6):
        layers = tensors.init_mlp(3 + 6 * L, 64, 3, 0, CPU)
        opt = torch.optim.Adam(tensors.parameters(layers), lr=3e-3)
        for _ in range(400):
            loss = ((equation.mlp(equation.positional_encoding(pts, L), layers)[:, :1] - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        errors[L] = loss.item()
    assert errors[6] < 0.1 * errors[0]


def test_positional_encoding_shape_and_values():
    x = torch.tensor([[0.25, -0.5, 1.0]])
    g = equation.positional_encoding(x, 3)
    assert g.shape == (1, 3 + 18)
    assert torch.allclose(g[0, :3], x[0])
    assert torch.allclose(g[0, 3:12].reshape(3, 3)[0], torch.sin(math.pi * torch.tensor([1.0, 2.0, 4.0]) * 0.25))
    assert torch.equal(equation.positional_encoding(x, 0), x)
