"""Physics checks for the MRI problems: relaxation contrast, Bloch spin echo, k-space imaging and DTI."""

import math

import pytest
import torch

from problems.medical.bloch_spin_echo import equation as be_eq
from problems.medical.bloch_spin_echo import solve as be_solve
from problems.medical.bloch_spin_echo import tensors as be_t
from problems.medical.diffusion_tensor_imaging import equation as dti_eq
from problems.medical.diffusion_tensor_imaging import solve as dti_solve
from problems.medical.diffusion_tensor_imaging import tensors as dti_t
from problems.medical.mri_kspace import equation as ks_eq
from problems.medical.mri_kspace import tensors as ks_t
from problems.medical.mri_relaxation import equation as rx_eq
from problems.medical.mri_relaxation import tensors as rx_t

CPU = torch.device("cpu")


# --- T1 / T2 relaxation and contrast -------------------------------------------------------------------


def test_relaxation_curves_match_exponentials():
    T1, T2, PD = rx_t.tissue_parameters(CPU)
    t = torch.linspace(0, 3000, 31)
    Mz = rx_eq.longitudinal(t, T1, PD)
    Mxy = rx_eq.transverse(t, T2, PD)
    for s in range(len(T1)):
        for k in (0, 7, 30):
            tk = t[k].item()
            assert Mz[s, k].item() == pytest.approx(
                PD[s].item() * (1 - math.exp(-tk / T1[s].item())), rel=1e-5, abs=1e-7
            )
            assert Mxy[s, k].item() == pytest.approx(PD[s].item() * math.exp(-tk / T2[s].item()), rel=1e-5, abs=1e-7)
    # after one T1 the magnetisation has recovered 1 - 1/e of the way
    one_T1 = rx_eq.longitudinal(T1[:1], T1[:1], torch.ones(1))
    assert one_T1.item() == pytest.approx(1 - math.exp(-1), rel=1e-6)


def test_inversion_recovery_null_at_T1_ln2():
    T1, _, PD = rx_t.tissue_parameters(CPU)
    TI = rx_eq.null_point(T1)
    Mz = rx_eq.inversion_recovery(TI, T1, PD)  # (S, S): row s evaluated at every tissue's TI
    assert torch.diagonal(Mz).abs().max() < 1e-5
    assert rx_eq.inversion_recovery(torch.zeros(1), T1, PD)[:, 0].tolist() == pytest.approx((-PD).tolist())


def test_weighting_contrast():
    T1, T2, PD = rx_t.tissue_parameters(CPU)
    names = rx_t.TISSUES
    wm, csf = names.index("white matter"), names.index("CSF")
    TR = torch.tensor([500.0, 4500.0])
    TE = torch.tensor([15.0, 100.0])
    S = rx_eq.spin_echo_signal(PD, T1, T2, TR, TE)  # (S, 2, 2)
    t1w, t2w = S[:, 0, 0], S[:, 1, 1]
    assert t1w[wm] > t1w[csf]  # T1-weighted: white matter brighter than CSF
    assert int(t2w.argmax()) == csf  # T2-weighted: CSF brightest
    # the einsum equals the product written out
    expected = PD * (1 - torch.exp(-TR[1] / T1)) * torch.exp(-TE[1] / T2)
    assert torch.allclose(t2w, expected)


def test_phantom_image_is_label_contraction():
    T1, T2, PD = rx_t.tissue_parameters(CPU)
    L = rx_t.phantom(32, CPU)
    assert torch.all(L.sum(-1) <= 1) and L.sum() > 0
    S = rx_eq.spin_echo_signal(PD, T1, T2, torch.tensor([500.0]), torch.tensor([15.0]))
    img = rx_eq.image(L, S)[..., 0, 0]
    label = L.argmax(-1)
    inside = L.sum(-1) > 0
    assert torch.allclose(img[inside], S[label[inside], 0, 0])
    assert torch.all(img[~inside] == 0)


# --- Bloch equation and spin echo ----------------------------------------------------------------------


def _bloch(M_init, dw, T1, T2, pulses, t_max, dt=0.02):
    eps = be_t.levi_civita(CPU)
    R, M0 = be_t.relaxation(T1, T2, CPU)
    return be_solve.solve(M_init, dw, eps, R, M0, pulses, t_max, dt)


def test_single_isochromat_free_precession():
    M_init = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    dw = torch.tensor([0.5], dtype=torch.float64)
    times, M = _bloch(M_init, dw, 1e12, 1e12, [], 20.0)
    norm = M[:, 0].norm(dim=-1)
    assert (norm - 1).abs().max() < 1e-6  # |M| = 1 without relaxation
    phase = torch.atan2(M[:, 0, 1], M[:, 0, 0])
    expected = torch.atan2(torch.sin(-0.5 * times), torch.cos(-0.5 * times))  # precesses at Δω (clockwise)
    assert torch.allclose(phase, expected, atol=1e-6)


def test_ninety_pulse_tips_into_transverse_plane():
    M_init = be_t.equilibrium_state(1, CPU)
    pulses = be_t.pulse_list(TE=20.0, n_echoes=0, duration=0.2)
    _, M = _bloch(M_init, torch.zeros(1, dtype=torch.float64), 1e12, 1e12, pulses, 0.2)
    final = M[-1, 0]
    assert abs(final[2].item()) < 1e-4  # RK4 rotation error only
    assert final[:2].norm().item() == pytest.approx(1.0, abs=1e-5)


def test_echo_at_TE_with_true_T2_amplitude():
    T2, TE, n = 80.0, 24.0, 300
    dw = be_t.off_resonance(n, 4.0, "lorentzian", CPU)
    pulses = be_t.pulse_list(TE, 1, 0.2)
    t90 = 0.1
    times, M = _bloch(be_t.equilibrium_state(n, CPU), dw, 800.0, T2, pulses, t90 + 1.3 * TE, dt=0.05)
    mag = be_eq.signal(M).abs()
    # FID decays much faster than T2: T2* < T2
    t2_star = be_solve.fid_decay_time(times, M, t90, TE)
    assert t2_star < 0.2 * T2
    i_half = int(((times - (t90 + TE / 4)).abs()).argmin())
    assert mag[i_half] < math.exp(-(TE / 4) / T2) * 0.5
    # echo peak at TE with amplitude ≈ exp(-TE/T2)
    t_echo, height = be_solve.echo_peaks(times, M, TE, 1, t90)
    assert t_echo[0].item() - t90 == pytest.approx(TE, abs=0.3)
    assert height[0].item() == pytest.approx(math.exp(-TE / T2), rel=0.03)


# --- k-space -------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kspace():
    n = 32
    rho = ks_t.shepp_logan(n, CPU)
    E = ks_t.encoding_matrix(n, CPU)
    return n, rho, E, ks_eq.encode(rho, E)


def test_encoding_einsum_equals_fft(kspace):
    _, rho, _, s = kspace
    assert (s - ks_eq.encode_fft(rho)).abs().max() < 1e-9


def test_full_sampling_reconstructs_phantom(kspace):
    n, rho, _, s = kspace
    rec = ks_eq.zero_filled(s, torch.ones(n, n, dtype=torch.float64))
    assert (rec.real - rho).abs().max() < 1e-6
    assert rec.imag.abs().max() < 1e-6


def test_skipping_lines_aliases_by_half_fov(kspace):
    n, rho, _, s = kspace
    mask = ks_t.line_mask(ks_t.scheme_lines("skip", n, 2), n, CPU)
    rec = ks_eq.zero_filled(s, mask).real
    assert torch.allclose(rec, 0.5 * (rho + torch.roll(rho, n // 2, dims=0)), atol=1e-9)


def test_lowpass_keeps_total_intensity(kspace):
    n, rho, _, s = kspace
    mask = ks_t.line_mask(ks_t.scheme_lines("lowpass", n, 0.25), n, CPU)
    rec = ks_eq.zero_filled(s, mask)
    assert rec.real.sum().item() == pytest.approx(rho.sum().item(), rel=1e-9)
    assert (rec.real - rho).abs().max() > 0.05  # but the image is blurred


def test_readout_k_is_gradient_integral():
    k, dwell = ks_t.readout_k(16, fov=0.2, gradient=5e-3)
    assert torch.allclose(torch.diff(k), torch.full((15,), 1 / 0.2, dtype=torch.float64))
    assert k[0].item() == pytest.approx(-8 / 0.2)
    assert dwell == pytest.approx((1 / 0.2) / (42.58e6 * 5e-3))


def test_regularised_without_penalty_equals_full_data(kspace):
    n, rho, E, s = kspace
    rec = ks_eq.regularised(s, torch.arange(n), E, torch.zeros(n, n, dtype=torch.float64), 0.0)
    assert (rec.real - rho).abs().max() < 1e-4


# --- Diffusion tensor imaging --------------------------------------------------------------------------


def test_dti_fit_recovers_true_tensor_without_noise():
    g = dti_t.gradient_directions(30, CPU)
    D, fibre, label = dti_t.phantom(16, CPU)
    S0, S = dti_solve.measure(D, g, 1.0, noise=0.0)
    D_fit, lam, e1, md, fa = dti_solve.fit(S0, S, dti_t.design_matrix(g), 1.0)
    assert (D_fit - D).abs().max() < 1e-8
    single = (label == 1) | (label == 2)
    align = (e1[single] * fibre[single]).sum(-1).abs()
    assert align.min() > 1 - 1e-8  # principal eigenvector along the fibre
    assert torch.allclose(md, torch.einsum("vii->v", D) / 3)  # MD = trace / 3
    assert torch.allclose(md, lam.mean(-1))


def test_fa_limits():
    iso = 2.0 * torch.eye(3, dtype=torch.float64)[None]
    linear = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))[None]
    nearly = torch.diag(torch.tensor([1.0, 1e-4, 1e-4], dtype=torch.float64))[None]
    assert dti_eq.invariants(iso)[3].item() == pytest.approx(0.0, abs=1e-12)
    assert dti_eq.invariants(linear)[3].item() == pytest.approx(1.0, abs=1e-12)
    assert dti_eq.invariants(nearly)[3].item() > 0.999
    bundle = dti_t.fibre_tensor(torch.tensor([[1.0, 0, 0]], dtype=torch.float64), 1.7, 0.3)
    lam = torch.tensor([1.7, 0.3, 0.3], dtype=torch.float64)
    fa = math.sqrt(1.5) * (lam - lam.mean()).norm().item() / lam.norm().item()
    assert dti_eq.invariants(bundle)[3].item() == pytest.approx(fa, rel=1e-9)


def test_signal_attenuation_along_direction():
    e = torch.tensor([[0.6, 0.8, 0.0]], dtype=torch.float64)
    D = dti_t.fibre_tensor(e, 1.7, 0.3)
    g = torch.tensor([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0], [0.8, -0.6, 0.0]], dtype=torch.float64)
    b = 1.0  # 1000 s/mm² in units of (10⁻³ mm²/s)⁻¹
    S = dti_eq.signal(D, g, b)[0]
    assert S.tolist() == pytest.approx([math.exp(-1.7), math.exp(-0.3), math.exp(-0.3)], rel=1e-12)


def test_tractography_follows_straight_bundle():
    n = 12
    D = dti_t.fibre_tensor(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), 1.7, 0.3).expand(n, n, 3, 3)
    lines = dti_solve.track(D, torch.tensor([[5.0, 5.0]], dtype=torch.float64), h=0.5, n_steps=30)
    pts = lines[:, 0][~lines[:, 0, 0].isnan()]
    assert (pts[:, 0] - 5.0).abs().max() < 1e-9  # stays on its column
    assert pts[:, 1].max() > 10.5  # and runs along the fibre
