"""Physics checks, written first (TDD): what any correct Oregonator (Belousov–Zhabotinsky) code must satisfy.

Run with `pytest problems/chemistry/belousov_zhabotinsky/test.py`. Each test is a physical law, not an implementation
detail: the steady state zeroes the rates, oscillations start at the Hopf points, and concentrations stay positive.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_steady_state():
    f = torch.linspace(0.2, 3.0, 50, dtype=torch.float64)
    xs = equation.steady_state(f, tensors.Q)
    dx, dz = equation.rhs(xs, xs, f, 0.04, tensors.Q)
    assert dx.abs().max() < 1e-9 and dz.abs().max() == 0  # fixed point: every rate vanishes
    assert (xs > tensors.Q).all()
    # production − loss form is the same right-hand side
    x, z = torch.rand(20, dtype=torch.float64), torch.rand(20, dtype=torch.float64)
    P, L = equation.production_loss(x, z, f[:20], tensors.Q)
    assert torch.allclose((P - L * x) / 0.04, equation.rhs(x, z, f[:20], 0.04, tensors.Q)[0])


def test_hopf_onset():
    eps = 0.04
    f_fine = torch.linspace(0.05, 4.0, 4000, dtype=torch.float64)
    hopf = solve.hopf_points(f_fine, eps, tensors.Q)
    assert len(hopf) == 2
    assert 0.45 < hopf[0] < 0.6 and 2.2 < hopf[1] < 2.45  # near 1/2 and 1 + √2 for small q
    # trace really vanishes there
    xs = equation.steady_state(hopf, tensors.Q)
    assert equation.trace(equation.jacobian(xs, xs, hopf, eps, tensors.Q)).abs().max() < 1e-2  # Hopf: tr J = 0
    # simulation: oscillation just inside the window, decay to the steady state just outside
    f = torch.tensor([hopf[0] - 0.1, 1.0, hopf[1] - 0.1, hopf[1] + 0.2], dtype=torch.float64)
    x0, z0 = tensors.initial_state(f)
    t, x, z = solve.integrate(x0, z0, f, eps, tensors.Q, tensors.DT, 12000)
    lo, hi = solve.amplitude(x)
    ratio = hi / lo
    assert ratio[0] < 1.05 and ratio[3] < 1.05
    assert ratio[1] > 100 and ratio[2] > 1.5
    assert torch.allclose(x[-1, [0, 3]], equation.steady_state(f[[0, 3]], tensors.Q), rtol=1e-2)
    assert (x > 0).all()  # concentrations stay positive
    period = solve.period(t, x[:, 1])
    assert period is not None and 4.5 < period < 7.0
