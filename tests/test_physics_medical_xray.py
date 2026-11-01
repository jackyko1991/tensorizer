"""Physics checks for the X-ray, CT and proton-therapy problems of the medical topic."""

import math

import torch

from problems.medical.proton_bragg_peak import equation as pb_eq
from problems.medical.proton_bragg_peak import solve as pb_solve
from problems.medical.proton_bragg_peak import tensors as pb_t
from problems.medical.radon_ct import equation as ct_eq
from problems.medical.radon_ct import solve as ct_solve
from problems.medical.radon_ct import tensors as ct_t
from problems.medical.xray_attenuation import equation as xr_eq
from problems.medical.xray_attenuation import solve as xr_solve
from problems.medical.xray_attenuation import tensors as xr_t

CPU = torch.device("cpu")


# --- X-ray attenuation ------------------------------------------------------------------------------


def test_xray_monochromatic_follows_exponential_law():
    mu = xr_t.attenuation_table(10.0, CPU)  # (E, M)
    thickness = torch.linspace(0.0, 20.0, 11)
    lengths = torch.zeros(mu.shape[1], len(thickness))
    lengths[2] = thickness  # soft tissue slabs of growing thickness
    T = xr_eq.transmission(mu, lengths)  # (E, X)
    expected = torch.exp(-mu[:, 2:3] * thickness[None, :])
    assert torch.allclose(T, expected, rtol=1e-5)


def test_xray_half_value_layer_halves_the_beam():
    mu = xr_t.attenuation_table(0.0, CPU)
    for m in (1, 2, 3):
        for e in range(mu.shape[0]):
            hvl = xr_solve.half_value_layer(float(mu[e, m]))
            assert math.isclose(hvl, math.log(2) / float(mu[e, m]), rel_tol=1e-6)
            lengths = torch.zeros(mu.shape[1], 1)
            lengths[m] = hvl
            assert math.isclose(float(xr_eq.transmission(mu[e : e + 1], lengths)), 0.5, rel_tol=1e-4)


def test_xray_beam_hardening_lowers_effective_mu():
    e = xr_t.energies(CPU)
    mu = xr_t.attenuation_table(0.0, CPU)
    spectrum = xr_t.tube_spectrum(120.0, e)
    _, _, mu_eff, mean_energy = xr_solve.beam_hardening(mu[:, 2], spectrum, e)
    assert torch.all(torch.diff(mu_eff) < 0)  # μ_eff strictly decreases with depth
    assert torch.all(torch.diff(mean_energy) > 0)  # the beam gets harder
    assert mu_eff[0] > mu_eff[-1] * 1.1


def test_xray_projection_is_additive_in_log_space():
    e = xr_t.energies(CPU)
    mu = xr_t.attenuation_table(10.0, CPU)
    x, y, masks = xr_t.phantom(64, CPU)
    dl = xr_t.path_weights(y)
    _, mono, _ = xr_solve.projections(mu, mu[3], masks, dl, xr_t.tube_spectrum(120.0, e))
    log_proj = -torch.log(mono[:-1])  # (E, X)
    # Splitting the phantom into two halves along the ray: the line integrals add up.
    top, bottom = masks.clone(), masks.clone()
    half = masks.shape[1] // 2
    top[:, half:], bottom[:, :half] = 0, 0
    parts = sum(xr_eq.line_integral(mu, xr_eq.path_lengths(m, dl)) for m in (top, bottom))
    assert torch.allclose(log_proj, parts, rtol=1e-4, atol=1e-5)
    # ...and log-attenuation is linear in the material lengths: doubling every length doubles it.
    L = xr_eq.path_lengths(masks, dl)
    assert torch.allclose(xr_eq.line_integral(mu, 2 * L), 2 * log_proj, rtol=1e-4, atol=1e-5)


def test_xray_bone_contrast_falls_with_energy():
    c = xr_solve.contrast_vs_energy(xr_t.attenuation_table(10.0, CPU))
    assert torch.all(torch.diff(c[:, 1]) < 0)  # bone
    assert c[1, 2] > c[0, 2]  # iodine K-edge: more contrast at 40 keV than at 30 keV


# --- Radon transform and CT -------------------------------------------------------------------------


def test_radon_of_centred_disk_matches_chord_length():
    x = ct_t.grid(256, CPU)
    R = 0.5
    theta = ct_t.angles(12, CPU)
    p = ct_solve.scan(ct_t.disk(x, R), theta, x)
    chord = 2 * torch.sqrt((R**2 - x**2).clamp_min(0.0))
    assert (p - chord).abs().max() < 0.02  # ~ one pixel width (2/256) of edge error
    assert (p - chord).abs().mean() < 0.003


def test_radon_conserves_mass_for_every_angle():
    x = ct_t.grid(128, CPU)
    f = ct_t.shepp_logan(x, ct_t.ellipse_params(CPU))
    dx = float(x[1] - x[0])
    p = ct_solve.scan(f, ct_t.angles(45, CPU), x)
    mass = p.sum(1) * dx
    assert torch.allclose(mass, (f.sum() * dx * dx).expand_as(mass), rtol=5e-3)


def test_analytic_ellipse_projection_matches_sampled_one():
    x = ct_t.grid(128, CPU)
    E = ct_t.ellipse_params(CPU)
    theta = ct_t.angles(30, CPU)
    analytic = sum(ct_eq.ellipse_projection(theta, x, *e) for e in E)
    sampled = ct_solve.scan(ct_t.shepp_logan(x, E), theta, x)
    assert (analytic - sampled).abs().mean() < 0.01


def test_fbp_error_small_and_decreasing_with_angles():
    x = ct_t.grid(128, CPU)
    f = ct_t.shepp_logan(x, ct_t.ellipse_params(CPU))
    errors = ct_solve.error_vs_angles(f, x, [8, 16, 32, 64, 128], "ram-lak")
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < 0.3  # limited by the sharp skull edges on a 128² grid
    assert errors[0] > 3 * errors[-1]


def test_plain_backprojection_of_point_falls_like_one_over_r():
    n = 129  # odd: a pixel sits exactly at the centre
    x = ct_t.grid(n, CPU)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    point = torch.exp(-(xx**2 + yy**2) / (2 * (1.5 * 2 / n) ** 2))
    theta = ct_t.angles(720, CPU)
    b = ct_eq.backproject(ct_eq.smear(ct_solve.scan(point, theta, x), theta, x, x), theta)
    c = n // 2
    r_times_b = torch.tensor([float(b[c, c + k] * x[c + k]) for k in (10, 20, 40, 60)])
    assert (r_times_b.max() - r_times_b.min()) / r_times_b.mean() < 0.05


# --- Proton Bragg peak ------------------------------------------------------------------------------


def _ranges(E0, dz=0.04):
    z, E = pb_solve.slow_down(E0, 75e-6, 40.0, dz)
    return z, E, pb_solve.csda_range(z, E)


def test_proton_range_150_mev_matches_pstar():
    _, _, R = _ranges(torch.tensor([150.0]))
    assert abs(float(R[0]) - 15.8) / 15.8 < 0.05
    e_ref, r_ref = pb_t.reference_ranges(CPU)
    _, _, R = _ranges(e_ref)
    assert torch.allclose(R, r_ref, rtol=0.03)


def test_proton_range_follows_bragg_kleeman():
    E0 = torch.linspace(70.0, 230.0, 9)
    _, _, R = _ranges(E0)
    alpha, p = pb_solve.fit_bragg_kleeman(E0, R)
    assert 1.70 < p < 1.82
    assert torch.allclose(pb_eq.bragg_kleeman(E0, alpha, p), R, rtol=0.03)


def test_proton_dose_peaks_near_end_of_range():
    E0 = torch.tensor([70.0, 120.0, 180.0, 230.0])
    z, E, R = _ranges(E0)
    z_mid, D = pb_solve.depth_dose(z, E, R, energy_spread=0.01)
    peak = z_mid[D.argmax(1)]
    assert torch.all(peak < R) and torch.all(peak > 0.95 * R)
    assert torch.all(D.max(1).values > 3 * D[:, 2])  # peak well above the entrance dose


def test_sobp_plateau_is_flat_over_target():
    E0 = torch.linspace(70.0, 230.0, 7)
    _, _, R = _ranges(E0)
    alpha, p = pb_solve.fit_bragg_kleeman(E0, R)
    El = pb_t.layer_energies(pb_t.layer_ranges(10.0, 15.0, 16, CPU), alpha, p)
    z, E, Rl = _ranges(El)
    z_mid, layers = pb_solve.depth_dose(z, E, Rl, energy_spread=0.01)
    w = pb_solve.sobp_weights(layers, z_mid, 10.0, 15.0)
    dose = pb_eq.sobp(layers, w)
    assert torch.all(w >= 0)
    assert pb_solve.flatness(dose, z_mid, 10.0, 15.0) < 0.03
    plateau = dose[(z_mid >= 10) & (z_mid <= 15)].mean()
    assert dose[z_mid > 17.0].max() < 0.05 * plateau  # protons spare tissue beyond the target
