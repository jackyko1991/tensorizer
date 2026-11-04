"""Checks written first (TDD): what any correct slow-roll inflation code must satisfy.

Run with `pytest problems/cosmology/inflation_slow_roll/test.py`. Units: reduced Planck mass M_pl = 1. Each test is a
known result: the slow-roll solution is an attractor, inflation ends where ε = 1 (ä changes sign), the potential
slow-roll formulas agree with the exact background to second order, m²φ² gives n_s ≈ 1 − 2/N and r ≈ 8/N, and the
Starobinsky model gives r ≈ 12/N².
"""

import numpy as np
import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


@pytest.fixture(scope="module")
def batch():
    """All five potentials integrated once, from their slow-roll velocities, shared by the tests below."""
    model = tensors.model(tensors.KINDS, f=7.0, mu=15.0, device=CPU)
    phi0 = tensors.start_field(model, 68.0)
    return model, solve.integrate(phi0, equation.slow_roll_velocity(phi0, model), model, dN=0.02, N_max=93.0)


def test_potentials_and_their_autograd_slopes():
    model = tensors.model(tensors.KINDS, f=7.0, mu=15.0, device=CPU)
    phi = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], dtype=F64)
    s = (2 / 3) ** 0.5
    v_exact = torch.tensor([9 / 2, 81 / 4, (1 - np.exp(-s * 3)) ** 2, 1 + np.cos(3 / 7), 1 - (3 / 15) ** 4], dtype=F64)
    assert torch.allclose(equation.potential(phi, model), v_exact, rtol=1e-12)
    dv = torch.tensor(
        [3, 27, 2 * s * np.exp(-3 * s) * (1 - np.exp(-3 * s)), -np.sin(3 / 7) / 7, -4 * 27 / 15**4], dtype=F64
    )  # V'(φ) by hand
    assert torch.allclose(equation.log_slope(phi, model), dv / v_exact, rtol=1e-10)
    assert torch.allclose(equation.autograd_slope(phi, model), dv / v_exact, rtol=1e-10)  # autograd agrees


def test_slow_roll_is_an_attractor():
    kinds = ["m2phi2"] * 4
    model = tensors.model(kinds, f=7.0, mu=15.0, device=CPU)
    phi0 = torch.full((4,), 16.0, dtype=F64)
    psi0 = torch.tensor([-2.0, -0.5, 0.0, 1.5], dtype=F64)  # very different initial velocities dφ/dN
    sol = solve.integrate(phi0, psi0, model, dN=0.04, N_max=72)
    phi, psi, N = sol["phi"].numpy(), sol["psi"].numpy(), sol["N"].numpy()
    late = np.linspace(12.0, 4.0, 50)  # compare dφ/dN at the same field value, after the transients
    curves = []
    for b in range(4):
        keep = (N > 3) & (N < sol["N_end"][b].item() - 1)  # φ falls monotonically there
        curves.append(np.interp(late, phi[keep, b][::-1], psi[keep, b][::-1]))
    for c in curves[1:]:
        assert (
            np.abs(c - curves[0]).max() < 1e-5
        )  # (interpolation error only) every trajectory has joined the same slow-roll curve
    # and it is the slow-roll velocity −V'/V = −2/φ, up to corrections of relative order ε_V = 2/φ²
    assert (np.abs(curves[0] * late / 2 + 1) < 2 / late**2).all()


def test_inflation_ends_when_epsilon_reaches_one(batch):
    model, sol = batch
    eps, N, N_end = sol["eps"], sol["N"], sol["N_end"]  # ε_H = −Ḣ/H² = (dφ/dN)²/2
    for b in range(len(tensors.KINDS)):
        before = N < N_end[b] - 0.02
        after = (N > N_end[b] + 0.02) & (N < N_end[b] + 0.1)
        assert (eps[before, b] < 1).all() and (eps[after, b] > 1).all()  # ä/a = H²(1 − ε) changes sign at N_end
    assert (N_end > 65).all()  # every model inflated long enough to set the pivot 50–60 e-folds before the end


def test_slow_roll_formulas_match_the_exact_background(batch):
    model, sol = batch
    pred = solve.predictions(sol, model, torch.tensor([50.0, 60.0], dtype=F64))  # each entry (S, B)
    eps_v, eta_v = pred["eps_V"], pred["eta_V"]
    small = torch.maximum(eps_v, eta_v.abs())
    # ε_H = ε_V (1 − 4ε_V/3 + 2η_V/3 + …): the same to first order
    assert ((pred["eps_H"] / eps_v - 1).abs() < 3 * small).all()
    # n_s and r from the potential (first order) against the Hubble-flow numbers on the numerical background
    assert ((pred["ns_V"] - pred["ns"]).abs() < 10 * small**2 + 2e-4).all()
    # r = 16ε₁(1 + Cε₂): the relative correction is of order ε₂ ≈ 4ε_V − 2η_V
    assert ((pred["r_V"] / pred["r"] - 1).abs() < 2 * (4 * eps_v - 2 * eta_v).abs() + 1e-3).all()


def test_quadratic_and_starobinsky_predictions(batch):
    model, sol = batch
    N = torch.tensor([50.0, 55.0, 60.0], dtype=F64)
    pred = solve.predictions(sol, model, N)
    ns, r = pred["ns"][:, [0, 2]], pred["r"][:, [0, 2]]  # columns: m²φ², Starobinsky
    assert torch.allclose(1 - ns[:, 0], 2 / N, rtol=0.03)  # m²φ²: n_s ≈ 1 − 2/N
    assert torch.allclose(r[:, 0], 8 / N, rtol=0.03)  # and r ≈ 8/N (≈ 0.15: ruled out by r < 0.036)
    assert torch.allclose(1 - ns[:, 1], 2 / N, rtol=0.08)  # Starobinsky: n_s ≈ 1 − 2/N too
    assert torch.allclose(r[:, 1], 12 / N**2, rtol=0.15)  # but r ≈ 12/N² (≈ 0.004)


def test_planck_bicep_comparison(batch):
    model, sol = batch
    pred = solve.predictions(sol, model, torch.tensor([55.0], dtype=F64))
    allowed = tensors.allowed(pred["ns"][0], pred["r"][0], sigmas=2)  # (B,) inside the 2σ n_s band and r < 0.036
    verdict = dict(zip(tensors.KINDS, allowed.tolist()))
    assert verdict["starobinsky"] and not verdict["m2phi2"] and not verdict["phi4"]
