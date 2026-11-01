"""Physics checks, written first (TDD): what any correct proton-therapy code must satisfy.

Run with `pytest problems/medical/proton_bragg_peak/test.py`. Each test is a physical law: CSDA ranges match the
PSTAR table and the Bragg–Kleeman rule, dose peaks just before the end of range, and a weighted spread-out Bragg
peak is flat over the target.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _ranges(E0, dz=0.04):
    z, E = solve.slow_down(E0, 75e-6, 40.0, dz)
    return z, E, solve.csda_range(z, E)


def test_range_150_mev_matches_pstar():
    _, _, R = _ranges(torch.tensor([150.0]))
    assert abs(float(R[0]) - 15.8) / 15.8 < 0.05
    e_ref, r_ref = tensors.reference_ranges(CPU)
    _, _, R = _ranges(e_ref)
    assert torch.allclose(R, r_ref, rtol=0.03)


def test_range_follows_bragg_kleeman():
    E0 = torch.linspace(70.0, 230.0, 9)
    _, _, R = _ranges(E0)
    alpha, p = solve.fit_bragg_kleeman(E0, R)
    assert 1.70 < p < 1.82
    assert torch.allclose(equation.bragg_kleeman(E0, alpha, p), R, rtol=0.03)


def test_dose_peaks_near_end_of_range():
    E0 = torch.tensor([70.0, 120.0, 180.0, 230.0])
    z, E, R = _ranges(E0)
    z_mid, D = solve.depth_dose(z, E, R, energy_spread=0.01)
    peak = z_mid[D.argmax(1)]
    assert torch.all(peak < R) and torch.all(peak > 0.95 * R)
    assert torch.all(D.max(1).values > 3 * D[:, 2])  # peak well above the entrance dose


def test_sobp_plateau_is_flat_over_target():
    E0 = torch.linspace(70.0, 230.0, 7)
    _, _, R = _ranges(E0)
    alpha, p = solve.fit_bragg_kleeman(E0, R)
    El = tensors.layer_energies(tensors.layer_ranges(10.0, 15.0, 16, CPU), alpha, p)
    z, E, Rl = _ranges(El)
    z_mid, layers = solve.depth_dose(z, E, Rl, energy_spread=0.01)
    w = solve.sobp_weights(layers, z_mid, 10.0, 15.0)
    dose = equation.sobp(layers, w)
    assert torch.all(w >= 0)
    assert solve.flatness(dose, z_mid, 10.0, 15.0) < 0.03
    plateau = dose[(z_mid >= 10) & (z_mid <= 15)].mean()
    assert dose[z_mid > 17.0].max() < 0.05 * plateau  # protons spare tissue beyond the target
