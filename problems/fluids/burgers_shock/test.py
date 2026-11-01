"""Physics checks, written first (TDD): what any correct viscous Burgers code must satisfy.

Run with `pytest problems/fluids/burgers_shock/test.py`. The periodic scheme conserves mass, the solution matches the
exact Cole–Hopf transform, and a shock forms near the inviscid breaking time t* = 1/max(−u₀').
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def burgers_run():
    nu, n = 0.05, 256
    x, dx = tensors.grid(n, CPU)
    D, L = tensors.derivative_matrices(n, dx, CPU)
    u0 = tensors.initial_wave(x, 1.0, 0.0)
    dt = solve.time_step(dx, 1.0, nu)
    times, u = solve.simulate(u0, D, L, nu, dt, 2.0)
    return {"x": x, "dx": dx, "D": D, "u": u, "times": times, "nu": nu}


def test_conserves_mass(burgers_run):
    m = equation.mass(burgers_run["u"], burgers_run["dx"])
    assert (m - m[0]).abs().max() < 1e-10  # conservation law u_t + (u²/2)_x = ν u_xx on a periodic domain


def test_matches_cole_hopf(burgers_run):
    r = burgers_run
    for t in (0.5, 1.0, 2.0):
        k = int((r["times"] - t).abs().argmin())
        exact = solve.cole_hopf(r["x"], r["times"][k].item(), r["nu"], 1.0, 0.0)
        assert (r["u"][k] - exact).abs().max() < 0.02  # exact solution via the Cole–Hopf transform


def test_shock_forms_near_breaking_time():
    nu, n = 0.01, 512
    x, dx = tensors.grid(n, CPU)
    D, L = tensors.derivative_matrices(n, dx, CPU)
    times, u = solve.simulate(tensors.initial_wave(x, 1.0, 0.0), D, L, nu, solve.time_step(dx, 1.0, nu), 2.0)
    steep = (-equation.slope(u, D)).max(dim=-1).values
    t_half = times[int((steep > 0.5 * steep.max()).int().argmax())].item()
    assert 0.9 < t_half < 1.3  # t* = 1/max(-u0') = 1
    k = int((times - 0.5).abs().argmin())
    assert steep[k].item() == pytest.approx(1 / (1 - times[k].item()), rel=0.05)  # inviscid slope 1/(1 - t) early on


def test_cole_hopf_initial_condition():
    x, _ = tensors.grid(64, CPU)
    early = solve.cole_hopf(x, 1e-4, 0.05, 1.0, 0.3, n_y=40000)
    assert torch.allclose(early, 0.3 + torch.sin(x), atol=1e-2)  # t → 0 limit recovers u₀
