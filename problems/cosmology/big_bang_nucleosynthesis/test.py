"""Physics checks, written first (TDD): what any correct Big Bang nucleosynthesis code must satisfy.

Run with `pytest problems/cosmology/big_bang_nucleosynthesis/test.py`. The checks are physical laws and classic
numbers: detailed balance, the radiation-era clock t ∝ 1/T², n/p freezing out near 1/6 and decaying to about 1/7,
Y_p ≈ 2(n/p)/(1 + n/p), baryon number conserved, deuterium falling with η, helium almost flat, and the yields at the
Planck η compared with the observed abundances (lithium is about 3× too high: the cosmological lithium problem).
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64


@pytest.fixture(scope="module")
def run():
    """One batched network run: 6 values of η across 1–10 × 10⁻¹⁰ plus the Planck value, all at once."""
    etas = torch.tensor([1.0, 2.0, 3.5, 5.0, 6.1, 8.0, 10.0], dtype=F64) * 1e-10
    return solve.run_bbn(etas, tau_n=878.4, n_nu=3.0, n_steps=900, device=CPU)


def test_reactions_conserve_baryon_number_and_charge():
    R, P = tensors.stoichiometry(CPU)  # (Rx, N) reactant and product counts
    A, Z = tensors.MASS_NUMBER.to(CPU), tensors.CHARGE.to(CPU)
    assert torch.equal((P - R) @ A, torch.zeros(len(R), dtype=A.dtype))  # every reaction keeps the baryon number
    assert torch.equal(((P - R) @ Z)[1:], torch.zeros(len(R) - 1, dtype=Z.dtype))  # nuclear reactions keep charge
    Q = tensors.q_values(CPU)
    assert float(Q[1]) == pytest.approx(2.2246, abs=1e-3)  # deuteron binding energy (MeV)
    assert float(Q[11]) == pytest.approx(17.347, abs=1e-2)  # ⁷Li + p → 2 ⁴He


def test_detailed_balance_reproduces_the_published_reverse_coefficients():
    """Reverse rates from the Saha equation give the coefficients of the Wagoner/Kawano code (Kawano 1992)."""
    T9 = torch.tensor([1.0], dtype=F64)
    coef = equation.reverse_coefficient(T9 * tensors.MEV_PER_T9, CPU)  # (1, Rx) λ⁻ / (λ⁺ e^{−Q/T}) up to n_b
    # d + γ → p + n per deuteron: λ = 4.71e9 T9^{3/2} e^{-25.82/T9} N_A⟨σv⟩, and here λ = ⟨σv⟩ · coef · e^{-Q/T}
    assert float(coef[0, 1] / tensors.N_AVOGADRO) == pytest.approx(4.71e9, rel=5e-3)
    assert float(coef[0, 2] / tensors.N_AVOGADRO) == pytest.approx(1.63e10, rel=5e-3)  # ³He + γ → d + p
    assert float(coef[0, 3]) == pytest.approx(1.73 * 2, rel=5e-3)  # ³He + n → d + d (×2: the ½ for identical d)


def test_radiation_era_clock():
    T = torch.tensor([5.0, 0.01], dtype=F64)  # MeV
    t = equation.time_of(T, n_nu=3.0)
    g_star = 10.75  # photons + e± + 3 neutrino species, all relativistic
    t_hot = 0.301 / math.sqrt(g_star) * tensors.M_PLANCK / T[0] ** 2 * tensors.HBAR  # t = 0.301 g*^{-1/2} M_P / T²
    assert float(t[0]) == pytest.approx(float(t_hot), rel=0.02)
    Tnu = equation.neutrino_temperature(T)
    assert float(Tnu[0] / T[0]) == pytest.approx(1.0, abs=1e-3)  # before e± annihilation
    assert float(Tnu[1] / T[1]) == pytest.approx((4 / 11) ** (1 / 3), rel=1e-3)  # after: entropy went to photons


def test_weak_rates_limits_and_detailed_balance():
    T = torch.tensor([1e-3, 0.5, 1.0, 3.0], dtype=F64)
    lam_np, lam_pn = equation.weak_rates(T, T, tau_n=878.4)  # T_ν = T here
    assert float(lam_np[0]) == pytest.approx(1 / 878.4, rel=1e-3)  # cold universe: free neutron decay
    ratio = lam_pn[1:] / lam_np[1:]
    assert torch.allclose(ratio, torch.exp(-tensors.Q_NP / T[1:]), rtol=2e-3)  # detailed balance: e^{-Q/T}


def test_neutron_to_proton_freeze_out_then_decay(run):
    T = run["T"]
    n_over_p = run["Y"][..., 0] / run["Y"][..., 1]  # (S, B)
    k_fo = int(torch.argmin((T - 0.25).abs()))  # T = 0.25 MeV: weak interactions frozen, nuclei not yet formed
    k_bn = int(torch.argmin((T - 0.08).abs()))  # T = 0.08 MeV: just before the deuterium bottleneck opens
    assert 1 / 6.5 < float(n_over_p[k_fo, 4]) < 1 / 4.8  # frozen out at 1/5–1/6 (the textbook "about 1/6")
    assert 1 / 7.8 < float(n_over_p[k_bn, 4]) < 1 / 6.4  # free decay brings it to about 1/7


def test_helium_is_the_neutrons_locked_up(run):
    Y = run["Y"]  # (S, B, N)
    A, Z = tensors.MASS_NUMBER.to(F64), tensors.CHARGE.to(F64)
    neutrons = torch.einsum("sbi,i->sb", Y, A - Z)  # free + bound: only weak reactions change it
    protons = torch.einsum("sbi,i->sb", Y, Z)
    opened = Y[..., 0] < 0.5 * neutrons  # the bottleneck has opened: half of the neutrons are in nuclei
    k = opened.to(torch.int8).argmax(0)  # (B,) first such step for every η
    x = (neutrons / protons).gather(0, k[None])[0]  # n/p when nucleosynthesis happens
    estimate = 2 * x / (1 + x)  # every neutron ends in ⁴He (2 n per 4 nucleons)
    assert torch.allclose(run["Yp"][2:], estimate[2:], rtol=0.015)
    assert torch.allclose(run["Yp"], estimate, rtol=0.045)  # at low η more neutrons stay behind in d and ³He


def test_baryon_number_is_conserved_at_every_step(run):
    total = torch.einsum("sbi,i->sb", run["Y"], tensors.MASS_NUMBER.to(F64))
    assert torch.allclose(total, torch.ones_like(total), atol=1e-9)


def test_deuterium_falls_and_helium_barely_moves_with_eta(run):
    dh, yp = run["DH"], run["Yp"]
    assert bool((dh.diff() < 0).all())  # more baryons ⇒ deuterium burnt more completely
    assert float(dh[0] / dh[-1]) > 20
    assert bool((yp.diff() > 0).all()) and float(yp[-1] - yp[0]) < 0.03  # only logarithmic in η


def test_planck_eta_matches_observations_except_lithium(run):
    i = 4  # η = 6.1 × 10⁻¹⁰
    assert float(run["Yp"][i]) == pytest.approx(0.247, rel=0.05)  # PDG observed 0.245 ± 0.003
    assert 2.527e-5 / 1.5 < float(run["DH"][i]) < 2.527e-5 * 1.5  # Cooke et al. 2018
    assert 0.5e-5 < float(run["He3H"][i]) < 2e-5
    li = float(run["LiH"][i])
    assert 2.0 < li / 1.6e-10 < 5.0  # the cosmological lithium problem: 2–5× the Spite plateau


@pytest.mark.slow
def test_step_size_converged():
    etas = torch.tensor([6.1e-10], dtype=F64)
    a = solve.run_bbn(etas, tau_n=878.4, n_nu=3.0, n_steps=900, device=CPU)
    b = solve.run_bbn(etas, tau_n=878.4, n_nu=3.0, n_steps=1800, device=CPU)
    assert float(a["Yp"][0]) == pytest.approx(float(b["Yp"][0]), rel=2e-3)
    assert float(a["DH"][0]) == pytest.approx(float(b["DH"][0]), rel=2e-2)


def test_extra_neutrinos_raise_helium():
    etas = torch.tensor([6.1e-10], dtype=F64)
    y3 = solve.run_bbn(etas, tau_n=878.4, n_nu=3.0, n_steps=500, device=CPU)["Yp"]
    y4 = solve.run_bbn(etas, tau_n=878.4, n_nu=4.0, n_steps=500, device=CPU)["Yp"]
    assert 0.008 < float(y4 - y3) < 0.02  # faster expansion ⇒ earlier freeze-out ⇒ ΔY_p ≈ 0.013 per species
