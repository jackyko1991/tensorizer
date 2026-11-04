"""Physics checks, written first (TDD): what any correct gravitational-wave search code must satisfy.

Run with `pytest problems/astro/gw150914_chirp/test.py`. Each test is a law, not an implementation detail: the
Newtonian chirp obeys its own frequency equation, Parseval holds, whitening flattens coloured noise, every template
is normalised, a matched filter finds an injected chirp at the right SNR, time and chirp mass, and the real LIGO
data give a chirp mass near the published one.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

FS = 2048.0


def test_newtonian_chirp_matches_its_frequency_equation():
    mc = torch.tensor([10.0, 30.0, 60.0], dtype=torch.float64)[:, None] * equation.T_SUN  # chirp mass in seconds
    tau = torch.linspace(0.02, 2.0, 400, dtype=torch.float64).requires_grad_(True)  # time to coalescence
    f = equation.chirp_frequency(tau, mc)  # (3, T)
    (dfdtau,) = torch.autograd.grad(f.sum(), tau)  # each row depends on its own τ only
    # df/dt = −df/dτ must equal (96/5) π^{8/3} M_c^{5/3} f^{11/3}
    rate = equation.chirp_rate(f.detach(), mc)
    assert torch.allclose(-dfdtau, rate.sum(0), rtol=1e-10)  # summed over the 3 masses on both sides
    rows = torch.autograd.functional.jacobian(lambda t: equation.chirp_frequency(t, mc[1]), tau.detach()).diagonal()
    assert torch.allclose(-rows, equation.chirp_rate(f[1].detach(), mc[1]), rtol=1e-10)
    # integrating the rate equation with RK4 from 30 Hz reproduces the closed form
    f0 = equation.chirp_frequency(torch.tensor(1.0, dtype=torch.float64), mc[1])
    y, dt = f0.clone(), 1e-4
    for _ in range(9000):  # 0.9 s forward in time: τ goes 1.0 → 0.1
        k1 = equation.chirp_rate(y, mc[1])
        k2 = equation.chirp_rate(y + dt / 2 * k1, mc[1])
        k3 = equation.chirp_rate(y + dt / 2 * k2, mc[1])
        k4 = equation.chirp_rate(y + dt * k3, mc[1])
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    exact = equation.chirp_frequency(torch.tensor(0.1, dtype=torch.float64), mc[1])
    assert y.item() == pytest.approx(exact.item(), rel=1e-6)
    # a 30 M☉ chirp sweeps through 35 Hz about 0.2 s before merger (GW150914: 0.2 s in band)
    assert 0.1 < equation.time_to_coalescence(torch.tensor(35.0), 30 * equation.T_SUN).item() < 0.3


def test_parseval_and_welch_normalisation():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8192, generator=gen, dtype=torch.float64)
    big_x = torch.fft.fft(x)
    assert torch.allclose(x.pow(2).sum(-1), big_x.abs().pow(2).sum(-1) / x.shape[-1], rtol=1e-12)  # Parseval
    # one-sided PSD of white noise with variance σ² sampled at fs is flat at 2σ²/fs; it integrates to σ²
    psd, f = solve.welch_psd(2.0 * x, FS, 1024)
    band = (f > 10) & (f < FS / 2 - 10)
    assert psd[:, band].mean().item() == pytest.approx(2 * 4.0 / FS, rel=0.03)
    assert torch.allclose(torch.trapezoid(psd, f), torch.full((3,), 4.0, dtype=torch.float64), rtol=0.04)


def test_whitening_flattens_coloured_noise():
    gen = torch.Generator().manual_seed(1)
    n = 16 * 2048
    f = torch.fft.rfftfreq(n, 1 / FS).double()
    model = tensors.model_psd(f)
    x = tensors.coloured_noise(model, n, FS, 2, gen)  # (2, n)
    psd, fw = solve.welch_psd(x, FS, 2048)
    band = (fw > 30) & (fw < 400)
    raw = psd[:, band].log10()
    assert (raw.amax(-1) - raw.amin(-1)).min().item() > 1.0  # more than a factor 10 across the band
    assert torch.allclose(psd[:, band].mean(-1), tensors.model_psd(fw[band]).mean(), rtol=0.15)
    white = solve.whiten(x, solve.interpolate(f, fw, psd), FS)
    psd_w, _ = solve.welch_psd(white[:, 2048:-2048], FS, 2048)
    flat = psd_w[:, band] / psd_w[:, band].mean(-1, keepdim=True)
    assert flat.std(-1).max().item() < 0.15  # flat to within Welch scatter
    assert torch.allclose(white[:, 2048:-2048].std(-1), torch.ones(2, dtype=torch.float64), rtol=0.1)  # unit variance


def test_templates_are_normalised():
    n = 16 * 2048
    f = torch.fft.rfftfreq(n, 1 / FS).double()
    psd = tensors.model_psd(f)
    mc = tensors.chirp_mass_grid(10.0, 60.0, 26)
    for pn in (0, 2):
        h = solve.template_bank(f, mc, psd, pn_order=pn)  # (K, F)
        norm = equation.inner(h, h, psd, f[1] - f[0])
        assert torch.allclose(norm, torch.ones_like(norm), rtol=1e-10)  # ⟨h|h⟩ = 1
        assert torch.isfinite(h).all()
    overlap = equation.inner(h[3], h[20], psd, f[1] - f[0]).abs()
    assert overlap < 0.9  # different chirp masses really are different templates


def test_matched_filter_recovers_injection_in_coloured_noise():
    gen = torch.Generator().manual_seed(2)
    n = 16 * 2048
    f = torch.fft.rfftfreq(n, 1 / FS).double()
    psd = tensors.model_psd(f)
    noise = tensors.coloured_noise(psd, n, FS, 1, gen)[0]
    mc_true, t_c, snr_true = 30.0, 9.0, 12.0
    signal = solve.injection(f, mc_true, t_c, snr_true, psd, n, FS)  # time series with ⟨s|s⟩^½ = 12
    data = noise + signal
    bank = tensors.chirp_mass_grid(20.0, 40.0, 41)
    res = solve.matched_filter(data[None], bank, psd[None], FS, pn_order=2)
    k = int(res["snr_peak"][0].argmax())
    assert res["snr_peak"][0, k].item() == pytest.approx(snr_true, abs=2.0)  # noise moves ρ by ~1
    assert bank[k].item() == pytest.approx(mc_true, abs=1.0)
    assert res["t_peak"][0, k].item() == pytest.approx(t_c, abs=2e-3)
    pure = solve.matched_filter(noise[None], bank, psd[None], FS, pn_order=2)
    assert pure["snr_peak"].max().item() < 6.5  # noise alone rarely exceeds ~5–6 in 16 s


@pytest.mark.slow
def test_real_gw150914_chirp_mass_and_offset():
    res = solve.analyse(tensors.load_strain(), pn_order=2, f_low=20.0, cutoff="peak")
    assert 26.0 < res["mc_bank"] < 34.0  # published detector-frame chirp mass ≈ 30 M☉ (source frame 28.3)
    # the Newtonian fit to the ridge reads heavier: PN corrections slow the real chirp at a given M_c
    assert 28.0 < res["mc_track"] < 42.0
    assert 5.0 < res["dt_ms"] < 9.5  # L1 sees it first, H1 ≈ 7 ms later
    assert res["snr"][0] > 10.0 and res["snr"][1] > 7.0
    assert math.isclose(res["t_event"], 7.42, abs_tol=0.03)  # GPS 1126259462.42 is 7.42 s into the file
