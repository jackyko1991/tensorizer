"""Physics checks, written first (TDD): what any correct diffusion-recovery code must satisfy.

Run with `pytest problems/biophysics/membrane_diffusion_frap/test.py`. Each test is a physical law: no-flux edges
conserve the dye, recovery is monotone, the half time matches Soumpasis' solution, and it scales as w² / D.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


def test_laplacian_stencil():
    x, h = tensors.grid(1.0, CPU)
    x = x.to(F64)
    w = tensors.stencil(CPU).to(F64)
    assert math.isclose(w.sum().item(), 0.0)
    const = torch.full((1, len(x), len(x)), 3.0, dtype=F64)
    assert equation.laplacian(const, w, h).abs().max() < 1e-12  # ∇² of a constant is zero, edges included
    quad = (x[:, None] ** 2 + x[None, :] ** 2)[None]  # ∇²(x² + y²) = 4
    lap = equation.laplacian(quad, w, h)[0, 1:-1, 1:-1]
    assert torch.allclose(lap, torch.full_like(lap, 4.0), atol=1e-9)


def _frap_run(w=1.0, t_end=12.0, D=None):
    x, h = tensors.grid(w, CPU)
    mask = tensors.bleach_mask(x, w)
    D = tensors.diffusion_coefficients(CPU)[:2] if D is None else D  # lipid and membrane protein
    c0 = tensors.initial_state(mask, 0.2, len(D))
    times, F, _, snaps = solve.solve(c0, D, tensors.stencil(CPU), h, mask, t_end, 3)
    F_inf = c0.mean(dim=(1, 2))
    f = equation.normalized_recovery(F, F[0], F_inf)
    return times, f, snaps, F_inf, D


def test_conserves_dye_and_recovers():
    times, f, snaps, F_inf, _ = _frap_run()
    assert (snaps.mean(dim=(2, 3)) - F_inf).abs().max() < 1e-5  # no-flux edges: total fluorescence constant
    assert math.isclose(f[0, 0].item(), 0.0, abs_tol=1e-6)
    assert (f[1:] - f[:-1]).min() > -1e-5  # recovery never goes backwards
    assert f[-1, 0] > 0.95  # the lipid has nearly fully recovered


def test_half_time_and_fit_match_soumpasis():
    w = 1.0
    times, f, _, _, D = _frap_run(w=w)
    t_half = solve.half_time(times, f)
    assert torch.allclose(t_half, equation.soumpasis_half_time(w, D), rtol=0.03)
    assert torch.allclose(solve.fit_diffusion(times, f, w, t_half), D, rtol=0.05)
    # the formula itself: f(t½) = ½ and f → 1 at long times
    tau = equation.diffusion_time(w, D)
    assert torch.allclose(
        equation.soumpasis(equation.soumpasis_half_time(w, D)[:1], tau[:1]), torch.tensor(0.5), atol=3e-3
    )
    assert equation.soumpasis(torch.tensor([1e4]), tau).min() > 0.99


def test_half_time_scales_as_w_squared():
    D = torch.tensor([1.0])
    t1 = solve.half_time(*_frap_run(w=1.0, t_end=1.0, D=D)[:2])
    t2 = solve.half_time(*_frap_run(w=2.0, t_end=4.0, D=D)[:2])
    assert math.isclose((t2 / t1).item(), 4.0, rel_tol=1e-3)  # doubling the spot: 4× slower, same D
