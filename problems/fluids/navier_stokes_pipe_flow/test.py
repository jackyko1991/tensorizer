"""Physics checks, written first (TDD): what any correct pipe-flow (Navier–Stokes) code must satisfy.

Run with `pytest problems/fluids/navier_stokes_pipe_flow/test.py`. The flow reaches the Hagen–Poiseuille parabola
with Q = πGR⁴/8μ, the wall shear balances the pressure gradient, and start-up follows the exact Bessel series on the
viscous time scale R²/ν.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pipe_run():
    R, nu, G_p, rho = 0.005, 1e-6, 0.5, 1000.0
    r, dr = tensors.grid(40, R, CPU)
    L = tensors.radial_laplacian(r, dr)
    times, u = solve.simulate(tensors.at_rest(40, CPU), L, G_p, rho, nu, dr, 2.0 * R**2 / nu)
    return {"r": r, "dr": dr, "L": L, "times": times, "u": u, "R": R, "nu": nu, "G": G_p, "mu": rho * nu, "rho": rho}


def test_laplacian_is_exact_for_the_parabola(pipe_run):
    r, L = pipe_run["r"], pipe_run["L"]
    u = (pipe_run["R"] ** 2 - r**2)[:-1]  # unknowns: everything but the wall
    assert torch.allclose(torch.einsum("ij,j->i", L, u), torch.full_like(u, -4.0), rtol=1e-9)  # ∇²(R² - r²) = -4


def test_reaches_hagen_poiseuille(pipe_run):
    p = pipe_run
    u_s = equation.hagen_poiseuille(p["r"], p["G"], p["mu"], p["R"])
    assert torch.allclose(p["u"][-1], u_s, rtol=1e-4, atol=1e-9)
    Q = equation.flow_rate(p["u"][-1], p["r"]).item()
    assert Q == pytest.approx(math.pi * p["G"] * p["R"] ** 4 / (8 * p["mu"]), rel=2e-3)  # Hagen–Poiseuille law
    tau_w = equation.wall_shear_stress(p["u"][-1], p["dr"], p["mu"]).item()
    assert tau_w == pytest.approx(p["G"] * p["R"] / 2, rel=1e-3)  # force balance G πR² = τ_w 2πR


def test_startup_matches_bessel_series(pipe_run):
    p = pipe_run
    tau = p["R"] ** 2 / p["nu"]
    for frac in (0.05, 0.2):
        k = int((p["times"] - frac * tau).abs().argmin())
        exact = solve.series_solution(p["r"], p["times"][k].item(), p["G"], p["mu"], p["rho"], p["R"])
        assert (p["u"][k] - exact).abs().max() < 2e-3 * exact.max()  # exact transient (Bessel series)


def test_time_scale_is_R2_over_nu(pipe_run):
    p = pipe_run
    tau = p["R"] ** 2 / p["nu"]
    centre = p["u"][:, 0] / p["u"][-1, 0]
    t63 = p["times"][int((centre > 0.63).int().argmax())].item() / tau
    assert 0.1 < t63 < 0.3  # slowest mode e^{-λ₁² ν t/R²}, λ₁ = 2.405 ⇒ about 0.19 R²/ν


def test_bessel_zeros():
    lam = solve.bessel_zeros(3)
    assert lam.tolist() == pytest.approx([2.404826, 5.520078, 8.653728], rel=1e-6)
