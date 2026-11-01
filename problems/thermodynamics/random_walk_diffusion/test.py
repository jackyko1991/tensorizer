"""Physics checks, written first (TDD): what any correct random-walk diffusion code must satisfy.

Run with `pytest problems/thermodynamics/random_walk_diffusion/test.py`. The mean squared displacement grows as
2dDt (Einstein relation), each axis spreads to variance 2Dt, and unbiased kicks give no drift.
"""

import math

import torch

from . import solve, tensors

CPU = torch.device("cpu")


def test_msd_grows_as_2dDt():
    D, t_max, n_steps = 0.7, 5.0, 100
    t = tensors.times(t_max, n_steps, CPU)
    kicks = tensors.random_kicks(n_steps, 5000, D, t_max / n_steps, CPU, seed=3)
    r = solve.solve(t, kicks)
    assert math.isclose(r["D_fit"], D, rel_tol=0.05)  # Einstein relation: MSD slope = 2dD with d = 2

    var = r["x"][-1].var(dim=0)  # each axis spreads to variance 2Dt
    assert torch.allclose(var, torch.full_like(var, 2 * D * t_max), rtol=0.08)
    assert r["x"][-1].mean(dim=0).abs().max() < 0.1 * math.sqrt(2 * D * t_max)  # no drift
