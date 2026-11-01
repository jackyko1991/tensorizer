"""Physics checks, written first (TDD): what any correct k-space imaging code must satisfy.

Run with `pytest problems/medical/mri_kspace/test.py`. Each test is a physical law: the encoding is a Fourier
transform, full sampling reconstructs exactly, skipping lines aliases by half the field of view, and keeping k = 0
preserves the total intensity.
"""

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def kspace():
    n = 32
    rho = tensors.shepp_logan(n, CPU)
    E = tensors.encoding_matrix(n, CPU)
    return n, rho, E, equation.encode(rho, E)


def test_encoding_einsum_equals_fft(kspace):
    _, rho, _, s = kspace
    assert (s - equation.encode_fft(rho)).abs().max() < 1e-9


def test_full_sampling_reconstructs_phantom(kspace):
    n, rho, _, s = kspace
    rec = equation.zero_filled(s, torch.ones(n, n, dtype=torch.float64))
    assert (rec.real - rho).abs().max() < 1e-6
    assert rec.imag.abs().max() < 1e-6


def test_skipping_lines_aliases_by_half_fov(kspace):
    n, rho, _, s = kspace
    mask = tensors.line_mask(tensors.scheme_lines("skip", n, 2), n, CPU)
    rec = equation.zero_filled(s, mask).real
    assert torch.allclose(rec, 0.5 * (rho + torch.roll(rho, n // 2, dims=0)), atol=1e-9)


def test_lowpass_keeps_total_intensity(kspace):
    n, rho, _, s = kspace
    mask = tensors.line_mask(tensors.scheme_lines("lowpass", n, 0.25), n, CPU)
    rec = equation.zero_filled(s, mask)
    assert rec.real.sum().item() == pytest.approx(rho.sum().item(), rel=1e-9)
    assert (rec.real - rho).abs().max() > 0.05  # but the image is blurred


def test_readout_k_is_gradient_integral():
    k, dwell = tensors.readout_k(16, fov=0.2, gradient=5e-3)
    assert torch.allclose(torch.diff(k), torch.full((15,), 1 / 0.2, dtype=torch.float64))
    assert k[0].item() == pytest.approx(-8 / 0.2)
    assert dwell == pytest.approx((1 / 0.2) / (42.58e6 * 5e-3))


def test_regularised_without_penalty_equals_full_data(kspace):
    n, rho, E, s = kspace
    rec = equation.regularised(s, torch.arange(n), E, torch.zeros(n, n, dtype=torch.float64), 0.0)
    assert (rec.real - rho).abs().max() < 1e-4
