"""Physics checks, written first (TDD): what any correct FDTD Maxwell code must satisfy.

Run with `pytest problems/em/em_wave_fdtd/test.py`. The Courant condition is enforced, the curl is the right finite
difference, pulses travel at c in vacuum, reflect with the Fresnel coefficient (n₁ − n₂)/(n₁ + n₂), and the 2D
field stays bounded.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def run_1d(n2: float, n_steps: int, nx: int = 600):
    """A 1D run (Ny = 1): Gaussian pulse from x = 150, interface to refractive index n2 at x = 350."""
    eps_r = torch.ones(nx, 1)
    eps_r[350:] = n2**2
    mask = torch.zeros(nx, 1)
    mask[150] = 1.0
    dt = equation.courant_time_step(1.0, 0.5, dim=1)
    E, H = tensors.empty_fields(nx, 1, CPU)
    damping = tensors.damping(nx, 1, 40, 0.1, CPU)
    pulse = solve.gaussian_pulse(30.0, 8.0)
    times, Ez = solve.simulate(E, H, eps_r, damping, mask, pulse, tensors.levi_civita(CPU), dt, 1.0, n_steps)
    return times, Ez[:, :, 0]


def test_courant_condition_enforced():
    assert equation.courant_time_step(1.0, 1.0, dim=2) == 1 / math.sqrt(2)
    assert equation.courant_time_step(0.5, 0.7, dim=2) * math.sqrt(2) / 0.5 <= 1.0  # CFL: c dt ≤ dx/√d
    for bad in [1.01, 2.0, 0.0]:
        try:
            equation.courant_time_step(1.0, bad, dim=2)
        except ValueError:
            continue
        raise AssertionError(f"Courant number {bad} was accepted")


def test_curl_matches_hand_written_finite_differences():
    eps = tensors.levi_civita(CPU)
    F = torch.randn(3, 6, 5)
    curl = equation.curl(eps, equation.forward_diff(F, 1.0))
    assert torch.allclose(curl[0, :, :-1], F[2, :, 1:] - F[2, :, :-1])  # (∇×F)_x = ∂_y F_z  (∂_z = 0)
    assert torch.allclose(curl[1, :-1, :], -(F[2, 1:, :] - F[2, :-1, :]))  # (∇×F)_y = −∂_x F_z
    assert torch.allclose(curl[2, :-1, :-1], (F[1, 1:, :-1] - F[1, :-1, :-1]) - (F[0, :-1, 1:] - F[0, :-1, :-1]))


def test_wave_speed_in_vacuum_is_c():
    times, Ez = run_1d(1.0, 700)
    right = Ez[:, 160:]  # only the pulse travelling to the right of the source
    k1, k2 = 300, 600
    speed = (right[k2].argmax() - right[k1].argmax()).item() / (times[k2] - times[k1]).item()
    assert abs(speed - 1.0) < 0.03  # Maxwell in vacuum: light travels at c = 1


def test_reflection_coefficient_at_dielectric_interface():
    n1, n2 = 1.0, 2.0
    _, Ez = run_1d(n2, 2000)
    probe = Ez[:, 250]  # between the source and the interface
    incident = probe.max().item()
    reflected = probe.min().item()  # (n1 − n2)/(n1 + n2) < 0: the echo comes back upside down
    assert abs(reflected / incident - (n1 - n2) / (n1 + n2)) < 0.02  # Fresnel reflection at normal incidence
    _, Ez_vacuum = run_1d(1.0, 2000)
    assert Ez_vacuum[:, 250].min().item() > -0.01 * incident  # no interface, no echo: the border absorbs


def test_stays_bounded_in_2d():
    nx, ny = 60, 50
    E, H = tensors.empty_fields(nx, ny, CPU)
    eps_r = tensors.permittivity(nx, ny, 4.0, CPU)
    damping = tensors.damping(nx, ny, 10, 0.1, CPU)
    mask = solve.point_source(nx, ny, CPU)
    dt = equation.courant_time_step(1.0, 0.99, dim=2)
    wave = solve.continuous_wave(10.0)
    _, Ez = solve.simulate(E, H, eps_r, damping, mask, wave, tensors.levi_civita(CPU), dt, 1.0, 400)
    assert torch.isfinite(Ez).all() and Ez.abs().max() < 10  # below the Courant limit the scheme is stable
