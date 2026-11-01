"""Physics checks, written first (TDD): what any correct FCS and lifetime code must satisfy.

Run with `pytest problems/biophysics/point_fcs_flim/test.py`. Each test is a physical law: G(0) = 1/N for a 3D
Gaussian volume, the FFT autocorrelation equals the direct sum, fits recover D and the lifetimes, and the lifetime
filters are dual to the decay patterns.
"""

from pathlib import Path

import pytest
import torch

from problems._common.runtime import load_meta, resolve_params

from . import equation, main, solve, tensors

CPU = torch.device("cpu")
META = load_meta(Path(__file__).parent)


@pytest.fixture(scope="module")
def mixture():
    """One seeded run of the default sample (free dye + labelled protein) with a 2 s trace."""
    return main.measure(resolve_params(META, {"trace": 2.0}), CPU)


def test_g0_equals_one_over_N_for_the_3d_gaussian():
    """var(I)/⟨I⟩² over random configurations = 1/N - 1/M (N = c V_eff in the volume, M molecules in the box)."""
    w0, z0 = 0.25, 1.25
    A = equation.shape_matrix(w0, z0, CPU).double()
    box = tensors.box_size(w0, z0, CPU).double()
    gen = torch.Generator().manual_seed(3)
    M = 250
    x = (torch.rand(20000, M, 3, generator=gen, dtype=torch.float64) - 0.5) * box  # 20000 snapshots
    intensity = equation.detection(x, A).sum(-1)  # (20000,)
    N = M * equation.effective_volume(w0, z0) / box.prod().item()
    G0 = intensity.var() / intensity.mean() ** 2
    assert G0.item() == pytest.approx(1 / N - 1 / M, rel=0.04)
    # the model is normalised so that G(0) = 1/N exactly
    g = equation.diffusion_model(torch.zeros(1, dtype=torch.float64), torch.tensor([1e-4], dtype=torch.float64), 5.0)
    assert g.item() == 1.0
    # a constant background B dilutes G(0) by ((I - B)/I)²; the correction restores it
    B = 0.3 * intensity.mean()
    G0_bg = intensity.var() / (intensity.mean() + B) ** 2
    corr = equation.background_correction((intensity.mean() + B).item(), B.item())
    assert (G0_bg * corr).item() == pytest.approx(G0.item(), rel=1e-10)


def test_fft_autocorrelation_matches_direct_sum():
    gen = torch.Generator().manual_seed(0)
    T = 400
    trace = torch.poisson(3 + 2 * torch.rand(2, T, generator=gen, dtype=torch.float64), generator=gen)
    G = equation.autocorrelation(trace)
    mean = trace.mean(-1, keepdim=True)
    d = trace - mean
    direct = torch.stack([(d[:, : T - k] * d[:, k:]).sum(-1) / (T - k) for k in range(T)], -1) / mean**2
    assert torch.allclose(G, direct, atol=1e-10)
    # the log-binned version is the mean of G over each lag window
    edges = solve.lag_edges(100)
    lags, G_bin = solve.binned_correlation(trace, edges)
    for i in range(len(edges) - 1):
        lo, hi = edges[i].item(), edges[i + 1].item()
        assert torch.allclose(G_bin[:, i], direct[:, lo:hi].mean(-1), atol=1e-10)
        assert lags[i].item() == pytest.approx((lo + hi - 1) / 2)


def test_single_species_fcs_recovers_D_and_N():
    r = main.measure(resolve_params(META, {"species": "dye", "trace": 1.0}), CPU)
    D = equation.diffusion_coefficient(r["w0"], r["tauD_all"])
    assert D.item() == pytest.approx(400.0, rel=0.2)
    assert 1 / r["a_all"].item() == pytest.approx(1.0, rel=0.2)  # background-corrected 1/G(0)


def test_flcs_separates_fast_and_slow_diffusers(mixture):
    D = equation.diffusion_coefficient(mixture["w0"], mixture["tauD_flcs"])
    assert D[0].item() == pytest.approx(400.0, rel=0.2)  # free dye
    assert D[1].item() == pytest.approx(40.0, rel=0.2)  # labelled protein
    assert torch.allclose(mixture["N_flcs"], torch.ones(2, dtype=torch.float64), rtol=0.25)


def test_lifetimes_and_fractions_recovered(mixture):
    assert torch.allclose(mixture["tau_fl"], torch.tensor([4.0, 2.5], dtype=torch.float64), rtol=0.1)
    fit = mixture["photons"] / mixture["photons"].sum()
    true = mixture["true_photons"].double() / mixture["true_photons"].sum()
    assert torch.allclose(fit, true, atol=0.05)


def test_mono_exponential_reconvolution_fit():
    gen = torch.Generator().manual_seed(5)
    tcspc = (tensors.IRF_T0, tensors.IRF_SIGMA, tensors.PERIOD)
    tau = torch.tensor([3.2], dtype=torch.float64)
    h = solve.reference_histograms(tau, 50_000, 2_000, gen, *tcspc, tensors.N_CHANNELS)
    tau_fit, photons, bg = solve.fit_flim(tensors.channels(CPU), h, 1, *tcspc)
    assert tau_fit.item() == pytest.approx(3.2, rel=0.03)
    assert photons.item() == pytest.approx(50_000, rel=0.03)
    assert bg.item() == pytest.approx(2_000, rel=0.25)


def test_flcs_filters_are_dual_to_the_patterns():
    t = tensors.channels(CPU)
    tcspc = (tensors.IRF_T0, tensors.IRF_SIGMA, tensors.PERIOD)
    p = equation.decay_pattern(t, torch.tensor([4.0, 2.5], dtype=torch.float64), *tcspc)  # (C, 2)
    assert torch.allclose(p.sum(0), torch.ones(2, dtype=torch.float64))  # normalised decays
    M = torch.cat([p, torch.full((len(t), 1), 1 / len(t), dtype=torch.float64)], 1)  # + flat background
    h = M @ torch.tensor([6e4, 4e4, 1e3], dtype=torch.float64)  # an expected mixture histogram
    F = equation.flcs_filters(M, h)
    assert torch.allclose(torch.einsum("kc,cj->kj", F, M), torch.eye(3, dtype=torch.float64), atol=1e-9)
    # applied to the mixture, each filter counts exactly the photons of its own component
    assert torch.allclose(F @ h, torch.tensor([6e4, 4e4, 1e3], dtype=torch.float64), rtol=1e-8)
