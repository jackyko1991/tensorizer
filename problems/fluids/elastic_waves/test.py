"""Physics checks, written first (TDD): what any correct elastic-wave code must satisfy.

Run with `pytest problems/fluids/elastic_waves/test.py`. P and S fronts travel at c_p = √((λ+2μ)/ρ) and
c_s = √(μ/ρ), the CFL limit is enforced, and the Navier–Cauchy einsums give the P-wave dispersion on a plane wave.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_wave_speeds_from_front_positions():
    n, lam, mu, rho = 161, 2.0, 1.0, 1.0
    c_p, c_s = equation.wave_speeds(lam, mu, rho)
    dt = equation.stable_time_step(1.0, c_p, c_s, 0.6)
    u, u_prev = tensors.at_rest(n, CPU)
    force = tensors.point_force(n, 0.0, 1.5, CPU)
    result = solve.simulate(u, u_prev, force, tensors.sponge(n, 20, 0.08, CPU), 1.0, dt, lam, mu, rho, 0.08, 40.0)
    times = result["times"]
    a, b = int((times - 25).abs().argmin()), int((times - 38).abs().argmin())
    span = (times[b] - times[a]).item()
    cp_measured = (solve.front_radius(result["div"][b], "x") - solve.front_radius(result["div"][a], "x")) / span
    cs_measured = (solve.front_radius(result["curl"][b], "y") - solve.front_radius(result["curl"][a], "y")) / span
    assert cp_measured == pytest.approx(c_p, rel=0.1)  # compressional wave: c_p = √((λ+2μ)/ρ)
    assert cs_measured == pytest.approx(c_s, rel=0.1)  # shear wave: c_s = √(μ/ρ)


def test_cfl_is_enforced():
    with pytest.raises(ValueError):
        equation.stable_time_step(1.0, 2.0, 1.0, 1.2)
    assert equation.stable_time_step(1.0, 2.0, 1.0, 1.0) == pytest.approx(1 / math.sqrt(5))


def test_navier_cauchy_einsums_on_a_plane_wave():
    # u = (sin kx, 0): ∂_i∂_j u_j = (-k² sin kx, 0) and ∂_j∂_j u_i = (-k² sin kx, 0) ⇒ ü = -(λ+2μ)k² u/ρ (a P wave)
    n = 64
    k = 2 * math.pi / n
    x = torch.arange(n, dtype=torch.float32)
    u = torch.zeros(2, n, n)
    u[0] = torch.sin(k * x)[:, None]
    acc = equation.acceleration(u, 1.0, 2.0, 1.0, 1.0)
    k_eff2 = 2 - 2 * math.cos(k)  # second difference of sin is -k_eff² sin
    assert torch.allclose(acc[0], -4.0 * k_eff2 * u[0], atol=1e-5)  # longitudinal wave feels λ + 2μ
    assert acc[1].abs().max() < 1e-5  # no shear component excited
