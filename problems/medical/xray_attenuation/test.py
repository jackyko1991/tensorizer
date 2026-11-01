"""Physics checks, written first (TDD): what any correct X-ray attenuation code must satisfy.

Run with `pytest problems/medical/xray_attenuation/test.py`. Each test is a physical law: monochromatic beams obey
Beer–Lambert, the half-value layer is ln 2 / μ, line integrals add in log space, and polychromatic beams harden with
depth.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_monochromatic_follows_exponential_law():
    mu = tensors.attenuation_table(10.0, CPU)  # (E, M)
    thickness = torch.linspace(0.0, 20.0, 11)
    lengths = torch.zeros(mu.shape[1], len(thickness))
    lengths[2] = thickness  # soft tissue slabs of growing thickness
    T = equation.transmission(mu, lengths)  # (E, X)
    expected = torch.exp(-mu[:, 2:3] * thickness[None, :])
    assert torch.allclose(T, expected, rtol=1e-5)


def test_half_value_layer_halves_the_beam():
    mu = tensors.attenuation_table(0.0, CPU)
    for m in (1, 2, 3):
        for e in range(mu.shape[0]):
            hvl = solve.half_value_layer(float(mu[e, m]))
            assert math.isclose(hvl, math.log(2) / float(mu[e, m]), rel_tol=1e-6)
            lengths = torch.zeros(mu.shape[1], 1)
            lengths[m] = hvl
            assert math.isclose(float(equation.transmission(mu[e : e + 1], lengths)), 0.5, rel_tol=1e-4)


def test_beam_hardening_lowers_effective_mu():
    e = tensors.energies(CPU)
    mu = tensors.attenuation_table(0.0, CPU)
    spectrum = tensors.tube_spectrum(120.0, e)
    _, _, mu_eff, mean_energy = solve.beam_hardening(mu[:, 2], spectrum, e)
    assert torch.all(torch.diff(mu_eff) < 0)  # μ_eff strictly decreases with depth
    assert torch.all(torch.diff(mean_energy) > 0)  # the beam gets harder
    assert mu_eff[0] > mu_eff[-1] * 1.1


def test_projection_is_additive_in_log_space():
    e = tensors.energies(CPU)
    mu = tensors.attenuation_table(10.0, CPU)
    x, y, masks = tensors.phantom(64, CPU)
    dl = tensors.path_weights(y)
    _, mono, _ = solve.projections(mu, mu[3], masks, dl, tensors.tube_spectrum(120.0, e))
    log_proj = -torch.log(mono[:-1])  # (E, X)
    # Splitting the phantom into two halves along the ray: the line integrals add up.
    top, bottom = masks.clone(), masks.clone()
    half = masks.shape[1] // 2
    top[:, half:], bottom[:, :half] = 0, 0
    parts = sum(equation.line_integral(mu, equation.path_lengths(m, dl)) for m in (top, bottom))
    assert torch.allclose(log_proj, parts, rtol=1e-4, atol=1e-5)
    # ...and log-attenuation is linear in the material lengths: doubling every length doubles it.
    L = equation.path_lengths(masks, dl)
    assert torch.allclose(equation.line_integral(mu, 2 * L), 2 * log_proj, rtol=1e-4, atol=1e-5)


def test_bone_contrast_falls_with_energy():
    c = solve.contrast_vs_energy(tensors.attenuation_table(10.0, CPU))
    assert torch.all(torch.diff(c[:, 1]) < 0)  # bone
    assert c[1, 2] > c[0, 2]  # iodine K-edge: more contrast at 40 keV than at 30 keV
