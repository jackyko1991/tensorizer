"""Physics checks, written first (TDD): what any correct image filtering code must satisfy.

Run with `pytest problems/graphics/image_filtering/test.py`. Each test is a mathematical invariant, not an
implementation detail: a normalised blur preserves the mean (DC gain 1), the Gaussian is separable, and the convolution
theorem makes FFT filtering equal circular convolution.
"""

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def test_separable_gaussian_equals_2d_conv():
    img = tensors.gaussian_noise(tensors.test_image(64, CPU), 0.05, seed=0)
    for sigma in (0.8, 1.5, 3.0):
        g, g2 = tensors.gaussian_1d(sigma, CPU), tensors.gaussian_2d(sigma, CPU)
        assert g.sum().item() == pytest.approx(1.0)
        assert torch.allclose(equation.separable(img, g), equation.filter2d(img, g2), atol=1e-5)  # separable


def test_unfold_einsum_equals_conv2d():
    img = tensors.test_image(48, CPU)
    for k in (tensors.box(5, CPU), tensors.sobel(CPU)[0], tensors.laplacian_of_gaussian(1.2, CPU)):
        assert torch.allclose(equation.filter2d_unfold(img, k), equation.filter2d(img, k), atol=1e-5)


def test_sobel_response_to_vertical_edge():
    img = torch.zeros(16, 16)
    img[:, 8:] = 1.0  # dark left half, bright right half
    kx, ky = tensors.sobel(CPU)
    gx, gy, mag, angle = equation.gradient(img, kx, ky)
    assert torch.allclose(gy, torch.zeros_like(gy))  # nothing changes vertically
    assert torch.allclose(gx[:, 7], torch.full((16,), 4.0))  # (1 + 2 + 1) × step on both sides of the edge
    assert torch.allclose(gx[:, 8], torch.full((16,), 4.0))
    assert gx[:, :6].abs().max() == 0 and gx[:, 10:].abs().max() == 0  # flat regions give zero
    assert torch.allclose(angle[:, 7], torch.zeros(16))  # the gradient points along +x
    assert torch.equal(mag, gx.abs())


def test_fft_convolution_equals_circular_spatial_convolution():
    img = tensors.gaussian_noise(tensors.test_image(40, CPU), 0.1, seed=3)
    asym = torch.arange(15.0).reshape(3, 5)  # not symmetric: checks the flip and the centring
    for k in (tensors.gaussian_2d(1.5, CPU), tensors.sobel(CPU)[1], asym):
        assert torch.allclose(equation.filter_fft(img, k), equation.filter2d(img, k, mode="circular"), atol=1e-3)


def test_frequency_responses():
    g2 = tensors.gaussian_2d(1.5, CPU)
    h = equation.frequency_response(g2, 64)
    assert h[0].item() == pytest.approx(1.0, abs=1e-5)  # weights sum to 1: flat regions keep their value
    assert h[-1].item() < 0.01  # the finest checkerboard is removed
    assert equation.frequency_response(tensors.laplacian_of_gaussian(1.5, CPU), 64)[0].abs().item() < 1e-5


def test_median_removes_isolated_salt_and_pepper():
    clean = torch.full((20, 20), 0.5)
    noisy = clean.clone()
    noisy[5, 5], noisy[12, 7] = 1.0, 0.0
    assert torch.allclose(equation.median(noisy, 3), clean)
    blurred = equation.filter2d(noisy, tensors.gaussian_2d(1.0, CPU))
    assert (blurred - clean).abs().max() > 0.05  # the Gaussian only spreads the outlier
