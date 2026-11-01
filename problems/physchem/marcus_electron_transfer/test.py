"""Physics checks, written first (TDD): what any correct Marcus electron-transfer code must satisfy.

Run with `pytest problems/physchem/marcus_electron_transfer/test.py`. Each test is a physical law, not an implementation
detail: the rate peaks at ΔG° = −λ with an inverted region, the barrier is the parabola crossing, hops follow the rate.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_rate_peaks_at_minus_lambda_and_inverted_region_falls():
    dG = tensors.driving_forces(-2.0, 0.5, 2501, CPU).double()
    lams = torch.tensor([0.3, 0.6, 1.0], dtype=torch.float64)
    kT = tensors.KB * torch.tensor([250.0, 300.0], dtype=torch.float64)
    k = solve.rate_grid(dG, lams, kT, 0.01)  # (L, T, G)
    for i, lam in enumerate(lams.tolist()):
        for t in range(len(kT)):
            g = int(k[i, t].argmax())
            assert abs(dG[g].item() + lam) < 2e-3  # activationless at −ΔG° = λ
            inverted = k[i, t, :g]  # more negative ΔG° than -λ
            assert (inverted[1:] - inverted[:-1] >= 0).all()  # Marcus inverted region: rate falls with driving force
            normal = k[i, t, g:]
            assert (normal[1:] - normal[:-1] <= 0).all()
    # symmetric about -λ, and the activationless rate is the prefactor
    lam, kT0 = 0.6, tensors.KB * 300
    a = equation.marcus_rate(torch.tensor(-lam - 0.2), torch.tensor(lam), torch.tensor(kT0), 0.01)
    b = equation.marcus_rate(torch.tensor(-lam + 0.2), torch.tensor(lam), torch.tensor(kT0), 0.01)
    assert torch.allclose(a, b)
    top = equation.marcus_rate(torch.tensor(-lam), torch.tensor(lam), torch.tensor(kT0), 0.01)
    expected = 2 * math.pi / tensors.HBAR * 1e-4 / math.sqrt(4 * math.pi * lam * kT0)  # Fermi's golden rule
    assert math.isclose(top.item(), expected, rel_tol=1e-5)


def test_crossing_point_on_both_parabolas():
    lam, dG = 0.7, torch.linspace(-1.5, 0.3, 7)
    x = equation.crossing_point(lam, dG)
    assert torch.allclose(equation.reactant_surface(x, lam), equation.product_surface(x, lam, dG), atol=1e-6)
    # barrier ΔG‡ = (λ + ΔG°)² / 4λ is the energy at the crossing
    assert torch.allclose(equation.reactant_surface(x, lam), equation.activation_energy(lam, dG), atol=1e-6)


def test_surface_hopping_follows_the_rate():
    lam, kT, H, omega = 0.5, tensors.KB * 300, 0.005, 20.0
    dG = torch.tensor([-0.25, -0.5, -0.75])
    x, v = tensors.thermal_ensemble(3, 1000, lam, kT, omega, CPU)
    res = solve.hopping_dynamics(x, v, dG, lam, kT, H, omega, 10.0, 0.1 / omega, 1500, [1500])
    k_sim, _ = solve.simulated_rates(res)
    k_th = equation.marcus_rate(dG, torch.tensor(lam), torch.tensor(kT), H)
    assert ((k_sim / k_th).log10().abs() < 0.15).all()
    assert k_sim[1] > k_sim[0] and k_sim[1] > k_sim[2]  # fastest at -ΔG° = λ, slower in the inverted region
