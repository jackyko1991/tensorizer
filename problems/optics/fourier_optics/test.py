"""Physics checks, written first (TDD): what any correct Fourier-optics code must satisfy.

Run with `pytest problems/optics/fourier_optics/test.py`. The far field is the Fourier transform of the aperture: an
Airy pattern with its first dark ring at 1.22 λ/D for a circle and sinc² for a square.
"""

import math

import torch

from . import equation, solve, tensors


def test_fft2_equals_einsum_dft():
    a = tensors.aperture("circle", 10.0, n=48)
    u_fft = equation.far_field(a).to(torch.complex128)
    u_ein = equation.far_field_einsum(a)
    assert torch.allclose(u_fft, u_ein, atol=1e-3)


def test_circular_aperture_first_dark_ring():
    for diameter in (16.0, 24.0, 32.0):
        measured = solve.first_dark_ring(diameter)
        assert math.isclose(measured, equation.airy_radius(tensors.N, diameter), rel_tol=0.1)


def test_square_aperture_gives_sinc_squared():
    n, side = tensors.N, 16
    a = tensors.aperture("square", side)
    assert a.sum().item() == side**2
    i = equation.intensity(equation.far_field(a))
    row = i[n // 2, n // 2 : n // 2 + 40]  # along the l axis, from the centre outwards
    expected = torch.sinc(side * torch.arange(40.0) / n) ** 2
    assert torch.allclose(row / row[0], expected, atol=0.01)  # Fraunhofer: far field of a slit is sinc²
    assert (row[[16, 32]] / row[0] < 1e-6).all()  # zeros at multiples of N / D
