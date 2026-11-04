"""Physics checks, written first (TDD): what any correct Maxwell's-demon simulation must satisfy.

Run with `pytest problems/thermodynamics/maxwell_demon/test.py`. Overdamped Langevin particles relax to the Boltzmann
distribution; without feedback Jarzynski's equality holds; with a demon's (noisy) measurement the Sagawa–Ueda
equality holds once the information is counted; the extracted work never beats kT times the mutual information
(at most kT ln 2 per bit); erasing a bit costs at least kT ln 2 and tends to it for slow protocols; Crooks' theorem
relates forward and reversed work; and the log ratio of forward and reversed path probabilities is (W - ΔF)/kT.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
LN2 = math.log(2.0)


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_relaxes_to_the_boltzmann_distribution():
    n, tilt = 4000, 2.0
    x0 = 2 * tensors.A * torch.rand(n, generator=gen(1), dtype=torch.float64)  # uniform: far from equilibrium
    side = torch.zeros(n, dtype=torch.float64)
    p = torch.full((n,), tensors.A, dtype=torch.float64)
    out = solve.run_protocol(x0, side, p, p, tau=torch.ones(n, dtype=torch.float64), t_total=2.5, gen=gen(2), tilt=tilt)
    x = out["x"].sort().values
    grid = tensors.x_grid(CPU)
    rho = torch.exp(-equation.potential(grid, torch.tensor(tensors.A), torch.tensor(0.0), tilt))
    cdf = torch.cumsum(rho, 0) / rho.sum()
    theory = _interp(x, grid, cdf)
    empirical = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    assert (theory - empirical).abs().max() < 0.03  # Kolmogorov–Smirnov distance to e^{-V/kT}/Z
    assert (x.mean() - (grid * rho).sum() / rho.sum()).abs() < 0.03


def _interp(x, xp, fp):
    i = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    w = (x - xp[i - 1]) / (xp[i] - xp[i - 1])
    return fp[i - 1] + w * (fp[i] - fp[i - 1])


def test_jarzynski_without_feedback():
    res = solve.crooks(n=6000, tau=0.15, gen=gen(3), device=CPU)
    dF = res["dF"]
    delta = math.sqrt(math.pi / (2 * tensors.KAPPA))  # each soft wall adds ∫e^{-κu²/2}du to the free length
    z_big, z_small = 2 * tensors.A + delta + delta / math.sqrt(2), tensors.A + 2 * delta  # piston on the wall: 2κ
    assert dF == pytest.approx(math.log(z_big / z_small), rel=2e-3)  # 2a → a: ΔF → kT ln 2 for hard walls
    W = res["W_forward"]
    assert torch.logsumexp(-W, 0).item() - math.log(len(W)) == pytest.approx(-dF, abs=0.04)  # ⟨e^{-βW}⟩ = e^{-βΔF}
    assert W.mean() > dF + 0.1  # a fast protocol dissipates: ⟨W⟩ > ΔF (second law)


def test_sagawa_ueda_with_feedback():
    eps = torch.tensor([0.2], dtype=torch.float64)
    res = solve.szilard(n_per=8000, eps=eps, tau=0.2, gen=gen(4), device=CPU)
    W, dF, info = res["W_on"], res["dF_m"], res["info"]
    su = torch.exp(-(W - dF) - info).mean()
    assert su.item() == pytest.approx(1.0, abs=0.05)  # ⟨e^{-β(W-ΔF)-I}⟩ = 1 (Sagawa–Ueda 2010)
    plain = torch.exp(-(W - dF)).mean()
    assert plain.item() > 1.15  # without the information term Jarzynski is violated: the demon's measurement matters
    I_mean = info.mean().item()
    assert I_mean == pytest.approx(equation.mutual_information(eps).item(), abs=0.02)  # ⟨I⟩ = ln 2 − H(ε)


def test_mutual_information_of_a_noisy_bit():
    eps = torch.tensor([0.0, 0.1, 0.5], dtype=torch.float64)
    info_th = equation.mutual_information(eps)
    assert info_th[0].item() == pytest.approx(LN2)  # a perfect measurement gives one bit
    assert info_th[1].item() == pytest.approx(LN2 + 0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert info_th[2].item() == pytest.approx(0.0, abs=1e-12)  # a coin flip tells nothing


@pytest.mark.slow
def test_extracted_work_is_bounded_by_kT_times_information():
    eps = torch.tensor([0.0, 0.1, 0.25], dtype=torch.float64)
    res = solve.szilard(n_per=900, eps=eps, tau=3.0, gen=gen(5), device=CPU)
    W_ext = -res["W_on"].view(len(eps), -1)
    bound = (res["info"] - res["dF_m"]).view(len(eps), -1).mean(1)  # ⟨I⟩ − ⟨ΔF_m⟩ (≈ I for hard walls)
    se = W_ext.std(1) / math.sqrt(W_ext.shape[1])
    mean = W_ext.mean(1)
    assert (mean <= bound + 3 * se).all()  # second law with information: ⟨W_ext⟩ ≤ kT I
    assert (mean <= LN2 + 3 * se).all()  # Szilard: at most kT ln 2 per bit
    info_th = equation.mutual_information(eps)
    assert (mean >= 0.7 * info_th).all()  # a slow protocol gets close to the bound


@pytest.mark.slow
def test_landauer_erasure_cost_tends_to_kT_ln2():
    taus = torch.tensor([0.25, 1.0, 4.0], dtype=torch.float64)
    res = solve.erasure(n_per=700, taus=taus, gen=gen(6), device=CPU)
    assert res["dF"] == pytest.approx(LN2, abs=1e-9)  # partitioned memory → confined bit: ΔF = kT ln 2 exactly
    W = res["W"].view(len(taus), -1)
    mean, se = W.mean(1), W.std(1) / math.sqrt(W.shape[1])
    assert (mean >= LN2 - 3 * se).all()  # Landauer: erasing a bit costs at least kT ln 2
    assert mean[0] > mean[1] > mean[2]  # slower is cheaper
    assert mean[2].item() == pytest.approx(LN2, abs=0.12)  # and tends to kT ln 2
    assert res["success"].min() > 0.99  # the bit really is reset: the particle ends in the left half (or its wall)


def test_crooks_fluctuation_theorem():
    res = solve.crooks(n=20000, tau=0.1, gen=gen(7), device=CPU)
    wf, minus_wr, dF = res["W_forward"], -res["W_reverse"], res["dF"]
    # P_F(W) e^{-β(W-ΔF)} = P_R(-W), integrated over W > w (robust to the atom at W = 0 of untouched particles)
    for w in (-0.2, 0.3, 0.6, 1.0):
        lhs = ((wf > w) * torch.exp(-(wf - dF))).mean()
        rhs = (minus_wr > w).to(wf.dtype).mean()
        assert lhs.item() == pytest.approx(rhs.item(), rel=0.08)
    slope, intercept = equation.crooks_line(wf, minus_wr, bins=24, min_count=40)
    assert slope == pytest.approx(1.0, abs=0.2)  # ln P_F(W)/P_R(-W) = β(W - ΔF): unit slope


def test_path_probability_ratio_is_entropy_production():
    res = solve.crooks(n=3000, tau=0.15, gen=gen(8), device=CPU)
    sigma = res["sigma_path"]  # ln P_F[x] / P_R[x̃] from the transition probabilities + boundary terms
    target = res["W_forward"] - res["dF"]
    assert (sigma - target).abs().max() < 1e-8  # every single trajectory: σ = β(W − ΔF), from detailed balance
    assert sigma.mean() > 0.1  # entropy is produced on average (second law) ...
    assert torch.exp(-sigma).mean().item() == pytest.approx(1.0, abs=0.08)  # integral fluctuation theorem
