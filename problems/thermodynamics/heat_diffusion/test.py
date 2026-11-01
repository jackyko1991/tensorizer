"""Physics checks, written first (TDD): what any correct heat-diffusion code must satisfy.

Run with `pytest problems/thermodynamics/heat_diffusion/test.py`. A Fourier sine mode decays at the analytic rate
α k², insulated edges conserve heat (first law), and an unstable explicit time step is refused.
"""

import math

import pytest
import torch

from . import solve, tensors

CPU = torch.device("cpu")


def test_sine_mode_decays_at_analytic_rate():
    X, Y, dx = tensors.grid(48, CPU)
    T0 = tensors.sine_mode(X, Y).double()
    alpha, t_max = 2e-5, 20.0
    times, frames = solve.solve(T0, tensors.K, tensors.heat_capacity(alpha), dx, t_max, boundary="dirichlet")
    rate = alpha * 2 * (math.pi / tensors.PLATE) ** 2  # α k² with k² = kx² + ky²
    amplitude = frames[-1].max() / frames[0].max()
    assert math.isclose(amplitude.item(), math.exp(-rate * t_max), rel_tol=1e-2)  # heat equation: e^{-α k² t}
    # the shape is an eigenmode: it only shrinks, it does not deform
    assert torch.allclose(frames[-1] / amplitude, frames[0], atol=1e-6)


def test_heat_is_conserved_with_insulated_edges():
    X, Y, dx = tensors.grid(32, CPU)
    T0 = tensors.hot_spots(X, Y, 3).double()
    rho_c = tensors.heat_capacity(5e-5)
    _, frames = solve.solve(T0, tensors.K, rho_c, dx, 10.0, boundary="neumann", T_edge=tensors.T_EDGE)
    heat = solve.heat_content(frames, rho_c, dx, tensors.T_EDGE)
    assert torch.allclose(heat, heat[0].expand_as(heat), rtol=1e-10)  # first law: no flux through the edges
    assert frames[-1].max() < frames[0].max()  # yet the hot spots spread out (second law)


def test_rejects_unstable_time_step():
    with pytest.raises(ValueError):
        solve.stable_dt(1e-4, 1e-3, 0.3)  # explicit scheme: Fourier number above the stability limit
