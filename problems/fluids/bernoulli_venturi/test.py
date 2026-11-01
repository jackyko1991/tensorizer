"""Physics checks, written first (TDD): what any correct Venturi-tube code must satisfy.

Run with `pytest problems/fluids/bernoulli_venturi/test.py`. The flow rate Q = A u is the same everywhere (mass
continuity), p + ½ρu² is constant along the tube (Bernoulli), and the meter inverts Δp back to Q.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_continuity_and_bernoulli():
    r1, ratio, u1, p1, rho = 0.05, 4.0, 1.5, 20_000.0, 1000.0
    x = tensors.axis(500, CPU)
    flow = solve.steady_flow(x, r1, r1 / math.sqrt(ratio), u1, p1, rho)
    Q = flow["A"] * flow["u"]
    assert torch.allclose(Q, torch.full_like(Q, flow["Q"]), rtol=1e-5)  # mass continuity: same Q everywhere
    u2 = flow["u"].max().item()
    assert u2 / u1 == pytest.approx(ratio, rel=1e-4)  # v2/v1 = A1/A2
    dp = p1 - flow["p"].min().item()
    assert dp == pytest.approx(0.5 * rho * (u2**2 - u1**2), rel=1e-4)
    energy = flow["p"] + 0.5 * rho * flow["u"] ** 2
    assert torch.allclose(energy, energy[0].expand_as(energy), rtol=1e-6)  # Bernoulli: p + ½ρu² is constant


def test_venturi_meter_recovers_flow_rate():
    A1, A2, rho, Q = 0.01, 0.0025, 1000.0, 0.02
    u1, u2 = Q / A1, Q / A2
    dp = 0.5 * rho * (u2**2 - u1**2)
    assert equation.venturi_flow_rate(dp, A1, A2, rho) == pytest.approx(Q, rel=1e-9)


def test_tracers_move_faster_in_the_throat():
    r1, r2, Q = 0.05, 0.025, 0.004
    x0 = torch.tensor([0.05, 0.45])  # inlet and throat
    times, traj = solve.move_tracers(x0, r1, r2, Q, t_max=0.01, dt=0.001)
    speed = (traj[1] - traj[0]) / (times[1] - times[0])
    assert (speed[1] / speed[0]).item() == pytest.approx((r1 / r2) ** 2, rel=1e-2)  # continuity: u ∝ 1/A
