"""Physics checks, written first (TDD): what any correct Gō-model code must satisfy.

Run with `pytest problems/biophysics/coarse_grained_md/test.py`. Each test is a physical law: forces are minus the
energy gradient, the native fold is the energy minimum, and Langevin dynamics keeps the fold cold and melts it hot.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _go_model64():
    native = tensors.native_structure(6, CPU).double()
    return native, tensors.go_model(native)


def test_autograd_forces_match_finite_differences():
    native, model = _go_model64()
    x = (native + 0.3 * torch.randn(native.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64))[
        None
    ]
    F = equation.forces(x, model)
    h = 1e-6
    for i, a in [(0, 0), (3, 1), (7, 2), (12, 0), (len(native) - 1, 1)]:
        xp, xm = x.clone(), x.clone()
        xp[0, i, a] += h
        xm[0, i, a] -= h
        numeric = -(equation.total_energy(xp, model) - equation.total_energy(xm, model)) / (2 * h)
        assert abs(numeric.item() - F[0, i, a].item()) < 1e-5 * max(1.0, abs(numeric.item()))


def test_native_fold_is_the_energy_minimum():
    native, model = _go_model64()
    n_contacts = int(model["native"].sum())
    E = equation.total_energy(native[None], model).item()
    assert abs(E + n_contacts) < 0.05  # every contact sits at the bottom of its well (-ε); bonds and angles at rest
    assert equation.forces(native[None], model).abs().max().item() < 0.05
    assert equation.fraction_native(native[None], model).item() == 1.0
    # a contact's 12–10 well has its minimum −ε exactly at r⁰
    r0 = torch.tensor([6.0], dtype=torch.float64)
    r = torch.linspace(5.0, 8.0, 3001, dtype=torch.float64)[:, None]
    well = equation.pair_energy(r, r0, torch.ones(1, dtype=torch.float64))
    assert abs(r[well.argmin()].item() - 6.0) < 2e-3 and abs(well.min().item() + 1.0) < 1e-9


def test_langevin_keeps_the_fold_cold_and_melts_it_hot():
    native = tensors.native_structure(6, CPU)
    model = tensors.go_model(native)
    T = torch.tensor([0.1, 3.0])
    x0 = native[None].repeat(2, 1, 1)
    _, Q, _, _ = solve.langevin(x0, T, model, 0.02, 1.0, 1500)
    assert Q[-200:, 0].mean() > 0.95
    assert Q[-200:, 1].mean() < 0.6


def test_melting_temperature_interpolates_the_half_point():
    T = torch.tensor([0.2, 0.4, 0.6, 0.8])
    q = torch.tensor([0.9, 0.7, 0.3, 0.2])
    assert abs(solve.melting_temperature(T, q) - 0.5) < 1e-6
