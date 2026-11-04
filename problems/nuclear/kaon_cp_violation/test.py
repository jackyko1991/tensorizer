"""Physics checks, written first (TDD): what any correct neutral-kaon code must satisfy.

Run with `pytest problems/nuclear/kaon_cp_violation/test.py`. Each test is a physical law, not an implementation
detail: CPT gives K⁰ and K̄⁰ equal masses and lifetimes, the decay matrix obeys unitarity (Bell–Steinberger), the
strangeness oscillates at the frequency Δm, the T-violating Kabir asymmetry is 4 Re ε and every asymmetry vanishes
when ε → 0.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
C128 = torch.complex128
DM, GS, GL = tensors.DM, tensors.GAMMA_S, tensors.GAMMA_L  # in units of ħ/τ_S
EPS = tensors.epsilon(tensors.EPS_ABS, tensors.EPS_PHASE)  # PDG ε (complex)


def scenarios(eps_list, delta_list):
    eps = torch.tensor(eps_list, dtype=torch.complex128)
    delta = torch.tensor(delta_list, dtype=torch.complex128)
    return equation.hamiltonian(DM, GS, GL, eps, delta)  # (B, 2, 2)


def test_cpt_gives_equal_masses_and_lifetimes():
    h = scenarios([EPS, EPS], [0.0, 3e-4])
    # CPT conserved (δ = 0): H₁₁ = H₂₂, so m(K⁰) = m(K̄⁰) = Re H₁₁ and Γ(K⁰) = Γ(K̄⁰) = −2 Im H₁₁
    assert torch.allclose(h[0, 0, 0], h[0, 1, 1], atol=1e-14)
    assert (h[1, 0, 0] - h[1, 1, 1]).abs() > 1e-4  # CPT violated (δ ≠ 0): they differ
    t = tensors.times(20.0, 400, CPU)
    p = equation.probabilities(equation.evolution(h, t))  # (B, T, 2, 2), P[b, t, final, initial]
    assert torch.allclose(p[0, :, 0, 0], p[0, :, 1, 1], atol=1e-14)  # P(K⁰→K⁰) = P(K̄⁰→K̄⁰)
    a_cpt = equation.cpt_asymmetry(p)
    assert a_cpt[0].abs().max() < 1e-12
    assert a_cpt[1, -1].item() == pytest.approx(4 * 3e-4, rel=0.01)  # long-time plateau 4 Re δ (CPLEAR's CPT test)
    a_t = equation.kabir_asymmetry(p[:, 1:])
    assert torch.allclose(a_t[0], a_t[1], atol=1e-12)  # mixing-only δ leaves the T asymmetry untouched


def test_eigenvalues_are_the_physical_states():
    h = scenarios([EPS], [0.0])[0]
    lam = torch.linalg.eigvals(h)
    lam = lam[lam.imag.argsort()]  # K_S decays fastest (most negative imaginary part)
    assert lam[0].real.item() == pytest.approx(-DM / 2, abs=1e-12)  # m_S = m − Δm/2
    assert lam[1].real.item() == pytest.approx(DM / 2, abs=1e-12)  # m_L = m + Δm/2
    assert -2 * lam[0].imag.item() == pytest.approx(GS, rel=1e-12)  # Γ_S
    assert -2 * lam[1].imag.item() == pytest.approx(GL, rel=1e-9)  # Γ_L


def test_evolution_matches_matrix_exponential():
    h = scenarios([EPS, 0.0], [1e-3, 0.0])
    t = tensors.times(15.0, 31, CPU)
    u = equation.evolution(h, t)  # (B, T, 2, 2)
    ref = torch.linalg.matrix_exp(-1j * t[None, :, None, None] * h[:, None])  # ψ(t) = e^{−iHt} ψ(0)
    assert torch.allclose(u, ref, atol=1e-12)


def test_unitarity_and_bell_steinberger():
    h = scenarios([EPS], [0.0])[0]
    gamma = equation.decay_matrix(h)  # Γ = i(H − H†)
    assert torch.allclose(gamma, gamma.conj().T, atol=1e-15)
    assert torch.linalg.eigvalsh(gamma).min() >= 0  # decays only remove probability: Γ ≥ 0
    v = equation.eigenvectors(
        torch.tensor(EPS, dtype=C128), torch.tensor(0j, dtype=C128)
    )  # columns K_S, K_L (normalised)
    overlap = torch.vdot(v[:, 0], v[:, 1])  # ⟨K_S|K_L⟩
    assert overlap.real.item() == pytest.approx(2 * EPS.real / (1 + abs(EPS) ** 2), rel=1e-12)
    # Bell–Steinberger: (i Δm + (Γ_S + Γ_L)/2) ⟨K_S|K_L⟩ = ⟨K_S|Γ|K_L⟩ = Σ_f A*_S(f) A_L(f)
    lhs = (1j * DM + (GS + GL) / 2) * overlap
    rhs = torch.vdot(v[:, 0], gamma @ v[:, 1])
    assert abs(lhs - rhs) < 1e-14
    assert abs(overlap) <= math.sqrt(GS * GL) / abs(1j * DM + (GS + GL) / 2)  # Cauchy–Schwarz unitarity bound
    # ππ dominance: Σ_f A*_S A_L ≈ Γ_S η₊₋, so the measured η₊₋ predicts Re ε
    eta = tensors.epsilon(tensors.ETA_PM_ABS, tensors.ETA_PM_PHASE)
    re_eps = (GS * eta / (2 * (1j * DM + (GS + GL) / 2))).real
    assert re_eps == pytest.approx(EPS.real, rel=0.02)
    t = tensors.times(20.0, 2001, CPU)
    norm = equation.probabilities(equation.evolution(h[None], t)).sum(-2)[0]  # (T, 2) total probability
    assert (norm.diff(dim=0) <= 1e-15).all()  # never increases


def test_strangeness_oscillates_at_delta_m():
    h = scenarios([0.0], [0.0])
    t = tensors.times(12.0, 1201, CPU)
    p = equation.probabilities(equation.evolution(h, t))[0]  # (T, 2, 2)
    closed = (torch.exp(-GS * t) + torch.exp(-GL * t) - 2 * torch.exp(-(GS + GL) / 2 * t) * torch.cos(DM * t)) / 4
    assert torch.allclose(p[:, 1, 0], closed, atol=1e-14)  # P(K⁰ → K̄⁰)
    a = equation.flavour_asymmetry(p)  # (P_same − P_opp)/(sum) ∝ cos Δm t
    crossing = t[:-1][(a[:-1] > 0) & (a[1:] <= 0)][0]
    assert crossing.item() == pytest.approx(math.pi / 2 / DM, abs=0.02)  # first zero at Δm t = π/2


def test_kabir_asymmetry_is_four_re_epsilon():
    h = scenarios([EPS], [0.0])
    t = tensors.times(20.0, 200, CPU)[1:]
    a_t = equation.kabir_asymmetry(equation.probabilities(equation.evolution(h, t)))[0]  # (T,)
    assert torch.allclose(a_t, a_t.mean().expand_as(a_t), atol=1e-12)  # a time-independent plateau
    assert a_t.mean().item() == pytest.approx(4 * EPS.real, rel=0.01)  # ≈ 4 Re ε ≈ 6.5 × 10⁻³
    value, error = tensors.CPLEAR_AT
    assert abs(a_t.mean().item() - value) < error  # agrees with CPLEAR (6.6 ± 1.6) × 10⁻³


def test_pipi_rates_match_the_closed_form():
    eta = tensors.epsilon(tensors.ETA_PM_ABS, tensors.ETA_PM_PHASE)
    h = scenarios([EPS], [0.0])
    v = equation.eigenvectors(torch.tensor([EPS], dtype=C128), torch.tensor([0j], dtype=C128))
    t = tensors.times(20.0, 400, CPU)
    rate = solve.pipi_rate(h, v, torch.tensor([eta], dtype=C128), t)[0]  # (T, 2) for K⁰ and K̄⁰ at t = 0
    # K⁰ = N/(2(1+ε)) (K_S + K_L), K̄⁰ = N/(2(1−ε)) (K_S − K_L) with N² = 2(1 + |ε|²)
    for col, sign, one_pm_eps in ((0, 1.0, 1 + EPS), (1, -1.0, 1 - EPS)):
        phase = math.atan2(eta.imag, eta.real)
        interference = 2 * abs(eta) * torch.exp(-(GS + GL) / 2 * t) * torch.cos(DM * t - phase)
        closed = torch.exp(-GS * t) + abs(eta) ** 2 * torch.exp(-GL * t) + sign * interference
        closed = closed * (1 + abs(EPS) ** 2) / (2 * abs(one_pm_eps) ** 2)
        assert torch.allclose(rate[:, col], closed, rtol=1e-10, atol=1e-16)


def test_no_cp_violation_removes_all_asymmetries():
    h = scenarios([0.0], [0.0])
    t = tensors.times(20.0, 300, CPU)[1:]
    p = equation.probabilities(equation.evolution(h, t))
    assert equation.kabir_asymmetry(p).abs().max() < 1e-12
    assert equation.cpt_asymmetry(p).abs().max() < 1e-12
    assert torch.allclose(p[..., 1, 0], p[..., 0, 1], atol=1e-15)  # P(K⁰→K̄⁰) = P(K̄⁰→K⁰): time-reversal symmetric
    v = equation.eigenvectors(torch.tensor([0j], dtype=C128), torch.tensor([0j], dtype=C128))
    rate = solve.pipi_rate(h, v, torch.tensor([0j], dtype=C128), t)[0]
    assert torch.allclose(rate[:, 1], rate[:, 0], rtol=1e-9, atol=1e-20)  # K⁰ and K̄⁰ decay to ππ alike
