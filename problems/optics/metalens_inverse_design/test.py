"""Physics checks, written first (TDD): what any correct metalens code must satisfy.

Run with `pytest problems/optics/metalens_inverse_design/test.py`. Angular-spectrum propagation conserves power, the
hyperbolic phase profile focuses at f (and is chromatic), and the optimised lens is at least as efficient.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _optics():
    return solve.Optics(torch.tensor([0.532, 0.633]), 0.532, 30.0, 40.0, CPU)


def test_angular_spectrum_propagation_conserves_energy():
    x, kx = tensors.grid(CPU)
    wl = torch.tensor([0.532, 0.633])
    beam = torch.exp(-(x**2) / (2 * 3.0**2)).to(torch.complex64)  # smooth: no evanescent components
    E0 = beam[None].expand(2, -1)
    E = equation.propagate(E0, equation.kz(kx, wl), torch.tensor([0.0, 10.0, 40.0]))  # (L, Z, X)
    power = (E.abs() ** 2).sum(dim=-1)
    assert torch.allclose(power, power[:, :1].expand_as(power), rtol=1e-4)
    assert not torch.allclose(E[:, 2].abs(), E[:, 0].abs(), atol=1e-2)  # but the beam did spread


def test_analytic_lens_focuses_at_f():
    optics = _optics()
    centres = tensors.atom_centres(40.0, CPU)
    theta = torch.remainder(equation.hyperbolic_theta(centres, 30.0, 0.532), 1.0)
    peak = solve.on_axis_peak(optics, theta, torch.linspace(10.0, 60.0, 501))
    assert abs(peak[0].item() - 30.0) < 0.03 * 30.0
    # the wrapped (diffractive) lens is chromatic: red light focuses closer, near f λ0/λ1
    assert abs(peak[1].item() - 30.0 * 0.532 / 0.633) < 0.08 * 30.0


def test_optimised_lens_at_least_as_good_as_analytic():
    optics = _optics()
    centres = tensors.atom_centres(40.0, CPU)
    theta_hyp = torch.remainder(equation.hyperbolic_theta(centres, 30.0, 0.532), 1.0)
    eta_hyp = optics.efficiency(theta_hyp)[0].item()
    weights = torch.tensor([1.0, 0.0])
    history, _, _ = solve.optimise(optics, torch.zeros(len(centres)), weights, 300, 0.02, n_snap=2)
    eta_opt = history[-1, 0].item()
    assert eta_hyp > 0.5
    assert eta_opt >= eta_hyp - 1e-3  # from a flat start it reaches (or beats) the analytic design
    assert history[0, 0].item() < 0.1  # the flat surface does not focus
