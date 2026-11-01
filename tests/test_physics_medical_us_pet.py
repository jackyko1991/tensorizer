import math

import torch

from problems.medical.doppler_ultrasound import equation as us_eq
from problems.medical.doppler_ultrasound import solve as us_solve
from problems.medical.doppler_ultrasound import tensors as us_t
from problems.medical.pet_mlem import equation as pet_eq
from problems.medical.pet_mlem import solve as pet_solve
from problems.medical.pet_mlem import tensors as pet_t

CPU = torch.device("cpu")
F0, PRF, R = 5e6, 10e3, 3e-3


# --- Doppler ultrasound ---------------------------------------------------------------------------------


def _steady_spectrum(v_max: float, theta: float, n: int, prf: float = PRF, duration: float = 1.0):
    """Average spectrum of n scatterers in steady flow with centreline velocity v_max (parabolic if n > 1)."""
    t = torch.arange(int(duration * prf)) / prf
    scat = us_t.scatterers(n, R, 20.0, CPU, seed=3)
    if n == 1:
        scat["r"] = torch.zeros(1)  # on the axis: moves at v_max
    _, s = us_solve.simulate(torch.full_like(t, v_max), scat, R, F0, theta, prf)
    _, freqs, power = us_solve.spectrogram(s, prf, n_fft=256, hop=64)
    return freqs, power.mean(1)


def test_single_scatterer_shift_matches_formula():
    theta = math.radians(45)
    for v in (0.3, 0.8, -0.5):
        freqs, spec = _steady_spectrum(v, theta, 1)
        expected = 2 * F0 * v * math.cos(theta) / us_eq.C_TISSUE
        assert abs(us_eq.doppler_shift(torch.tensor(v), F0, theta).item() - expected) < 1e-6 * abs(expected) + 1e-3
        assert abs(freqs[spec.argmax()].item() - expected) <= (freqs[1] - freqs[0]).item()
        # the inverse relation recovers the speed
        assert abs(us_eq.velocity_from_frequency(torch.tensor(expected), F0, theta).item() - v) < 1e-5


def test_perpendicular_beam_gives_zero_shift():
    assert abs(us_eq.doppler_shift(torch.tensor(1.0), F0, math.pi / 2).item()) < 1e-9
    freqs, spec = _steady_spectrum(1.0, math.pi / 2, 1)
    assert abs(freqs[spec.argmax()].item()) < 1e-9


def test_max_spectral_frequency_is_centreline_velocity():
    theta, v_max = math.radians(60), 1.0
    freqs, spec = _steady_spectrum(v_max, theta, 2000)
    f_max = us_eq.doppler_shift(torch.tensor(v_max), F0, theta).item()  # 3247 Hz
    plateau = spec[(freqs > 0.2 * f_max) & (freqs < 0.8 * f_max)].mean()
    edge = freqs[spec > 0.1 * plateau].max().item()  # highest frequency with real signal
    assert abs(edge - f_max) <= 2 * (freqs[1] - freqs[0]).item()


def test_parabolic_profile_mean_frequency_is_half_max():
    theta, v_max = math.radians(60), 1.0
    freqs, spec = _steady_spectrum(v_max, theta, 2000)
    f_max = us_eq.doppler_shift(torch.tensor(v_max), F0, theta).item()
    f_mean = us_solve.mean_frequency(freqs, spec[:, None]).item()
    assert abs(f_mean / f_max - 0.5) < 0.04
    # the same from the velocities themselves: uniform density over the disc ⇒ <1 - r²/R²> = 1/2
    v = us_eq.velocity_field(torch.ones(1), us_t.scatterers(20000, R, 20.0, CPU)["r"], R)
    assert abs(v.mean().item() - 0.5) < 0.01


def test_aliasing_wraps_above_nyquist():
    theta, prf = 0.0, 4e3
    v = 0.8  # f_D = 2 · 5 MHz · 0.8 / 1540 ≈ 5.19 kHz > PRF/2 = 2 kHz
    f_true = us_eq.doppler_shift(torch.tensor(v), F0, theta).item()
    f_alias = us_eq.alias(torch.tensor(f_true), prf).item()
    assert -prf / 2 <= f_alias < prf / 2
    assert abs((f_true - f_alias) / prf - round((f_true - f_alias) / prf)) < 1e-6  # differs by a multiple of PRF
    freqs, spec = _steady_spectrum(v, theta, 1, prf=prf)
    assert abs(freqs[spec.argmax()].item() - f_alias) <= (freqs[1] - freqs[0]).item()
    # below the Nyquist limit nothing changes
    assert us_eq.alias(torch.tensor(1500.0), prf).item() == 1500.0


def test_cardiac_waveform_has_systolic_peak():
    t = us_t.sample_times(1000.0, 60.0, 1, CPU)
    v = us_t.cardiac_waveform(t, 1.0, 60.0)
    assert abs(v.max().item() - 1.0) < 0.05 and 0.08 < t[v.argmax()].item() < 0.16
    assert v.min().item() < 0  # brief reverse flow


# --- PET / MLEM -----------------------------------------------------------------------------------------


def test_annihilation_photons_back_to_back_511_kev():
    g = torch.Generator().manual_seed(0)
    phi = 2 * math.pi * torch.rand(1000, generator=g)
    u, energy = pet_eq.annihilation_photons(phi)
    cos = torch.einsum("ei,ei->e", u[:, 0], u[:, 1])
    assert torch.allclose(cos, torch.full_like(cos, -1.0))  # 180° apart
    assert torch.allclose(u.norm(dim=-1), torch.ones(1000, 2))
    assert torch.all(energy == 511.0)
    # both photons reach the ring, on a straight line through the annihilation point
    a = 10 * (torch.rand(1000, 2, generator=g) - 0.5)
    h = pet_eq.ring_hits(a, u, 40.0)
    assert torch.allclose(h.norm(dim=-1), torch.full((1000, 2), 40.0), atol=1e-3)
    d1, d2 = h[:, 0] - a, h[:, 1] - a
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    assert cross.abs().max() < 1e-2 and (torch.einsum("ei,ei->e", d1, d2) < 0).all()


def test_positron_range_blurs_point_source():
    g = torch.Generator().manual_seed(1)
    sigma = 0.3  # cm
    a = pet_eq.positron_range(torch.zeros(200000, 2), sigma, g)
    assert a.mean(0).abs().max() < 0.01
    assert torch.allclose(a.std(0), torch.full((2,), sigma), rtol=0.02)
    rms_radius = a.norm(dim=1).pow(2).mean().sqrt().item()  # 2D Gaussian: √2 σ
    assert abs(rms_radius - math.sqrt(2) * sigma) < 0.01


def _small_problem(n: int = 32):
    activity, mu = pet_t.phantom(n, CPU)
    A = pet_t.system_matrix(n, CPU)
    truth = activity.flatten() * 1e5 / activity.sum()
    return A, truth


def test_system_matrix_columns_are_probabilities():
    A, truth = _small_problem()
    n = 32
    centre = pet_t.pixel_centres(n, CPU).view(-1, 2).norm(dim=1) < 12  # well inside the sinogram range
    col = A.sum(0)[centre]
    assert abs(col.mean().item() - 1.0) < 0.05


def test_mlem_conserves_counts_and_likelihood_increases():
    A, truth = _small_problem()
    g = torch.Generator().manual_seed(2)
    y = torch.poisson(A @ truth, generator=g)
    r = torch.zeros_like(y)
    images, ll, totals = pet_solve.mlem(A, y, r, 30)
    assert torch.allclose(totals, torch.full_like(totals, y.sum().item()), rtol=1e-4)
    assert (images >= 0).all()
    assert (ll.diff() >= -1e-6 * ll.abs()[1:]).all()
    assert ll[-1] > ll[1] > ll[0]


def test_mlem_noiseless_converges_to_truth():
    A, truth = _small_problem()
    y = A @ truth  # exact, noise-free data
    images, ll, _ = pet_solve.mlem(A, y, torch.zeros_like(y), 150)
    err = pet_solve.relative_error(images, truth)
    assert err[150] < err[50] < err[10] < err[1]
    assert err[150] < 0.25 * err[0]


def test_lor_coordinates_of_a_known_line():
    # horizontal line y = 5 → normal along +y (θ = 90°), s = 5
    theta, s = pet_eq.lor_coordinates(torch.tensor([[-40.0, 5.0]]), torch.tensor([[40.0, 5.0]]))
    assert abs(theta.item() - math.pi / 2) < 1e-6 and abs(s.item() - 5.0) < 1e-5
