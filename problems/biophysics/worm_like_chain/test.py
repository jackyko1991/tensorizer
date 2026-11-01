"""Physics checks, written first (TDD): what any correct worm-like chain code must satisfy.

Run with `pytest problems/biophysics/worm_like_chain/test.py`. Each test is a physical law: Marko–Siggia has the
right low- and high-force limits, free chains have the exact exponential tangent correlation, local and total energy
changes agree, and Monte Carlo gives back the persistence length.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_force_extension_limits():
    P = 50.0
    b = 2 * P
    # low force: both laws are linear with the same slope z/L = 2 P F / 3 kT
    F = torch.tensor([1e-4, 3e-4], dtype=torch.float64)
    slope = 2 * P / (3 * equation.KT)
    assert torch.allclose(equation.marko_siggia_extension(F, P) / F, torch.full_like(F, slope), rtol=2e-3)
    assert torch.allclose(equation.fjc_extension(F, b) / F, torch.full_like(F, slope), rtol=2e-3)
    # high force: 1 - z/L → ½ √(kT / F P) for the WLC, kT / (F b) for the FJC
    F = torch.tensor([200.0, 1000.0], dtype=torch.float64)
    wlc = 1 - equation.marko_siggia_extension(F, P)
    assert torch.allclose(wlc, 0.5 * torch.sqrt(equation.KT / (F * P)), rtol=0.03)
    assert torch.allclose(1 - equation.fjc_extension(F, b), equation.KT / (F * b), rtol=1e-6)
    # the inversion is exact: F(z(F)) = F
    F = torch.logspace(-2, 1, 9, dtype=torch.float64)
    assert torch.allclose(equation.marko_siggia_force(equation.marko_siggia_extension(F, P), P), F, rtol=1e-8)


def test_stiffness_gives_requested_persistence_length():
    for P, a in [(50.0, 2.5), (20.0, 1.0), (100.0, 5.0)]:
        K = tensors.stiffness_for(P, a)
        assert math.isclose(equation.persistence_from_stiffness(K, a), P, rel_tol=1e-9)
        assert math.isclose(K, P / a + 0.5, rel_tol=0.01)


def test_force_free_chains_have_exact_tangent_correlation():
    P, a = 50.0, 2.5
    K = tensors.stiffness_for(P, a)
    t = tensors.relaxed_chains(4, 64, 200, K, seed=0, device=CPU)
    assert torch.allclose(t.norm(dim=-1), torch.ones(()), atol=1e-5)
    c = equation.tangent_correlation(t, 20)
    n = torch.arange(21, dtype=c.dtype)
    assert torch.allclose(c, torch.exp(-n * a / P), atol=0.03)
    assert math.isclose(equation.persistence_from_correlation(c, a), P, rel_tol=0.08)


def test_local_energy_change_matches_total_energy():
    K = 10.0
    f = torch.tensor([0.0, 1.5])
    t = tensors.relaxed_chains(2, 3, 30, K, seed=1, device=CPU).double()
    i = 7
    t_new = t.clone()
    t_new[:, :, i] = torch.nn.functional.normalize(t[:, :, i] + 0.3 * torch.randn(2, 3, 3, dtype=t.dtype), dim=-1)
    dE_local = equation.local_energy_change(equation.local_field(t, K, f.double()), t, t_new)[:, :, i]
    dE_total = equation.bending_energy(t_new, K, f.double()) - equation.bending_energy(t, K, f.double())
    assert torch.allclose(dE_local, dE_total, atol=1e-10)


def test_monte_carlo_recovers_persistence_length():
    P, a, L = 50.0, tensors.SEGMENT, 600.0
    K = tensors.stiffness_for(P, a)
    F = torch.tensor([0.0, 0.6, 1.2, 2.5])
    f = tensors.reduced_forces(F, a)
    t0 = tensors.relaxed_chains(len(F), 6, int(L / a), K, seed=1, device=CPU)
    t0[1:] = torch.tensor([0.0, 0.0, 1.0])  # pulled chains start straight: they relax fast at these forces
    out = solve.simulate(t0, K, f, sweeps=400, n_records=5, show=[1], seed=0)
    assert torch.allclose(out["t"].norm(dim=-1), torch.ones(()), atol=1e-4)  # tangents stay unit vectors
    # the free chain keeps the right correlation
    assert math.isclose(out["correlation"][1].item(), equation.segment_correlation(K), rel_tol=2e-3)
    assert math.isclose(equation.persistence_from_correlation(out["correlation"][:20], a), P, rel_tol=0.1)
    # the stretched chains follow Marko–Siggia, and a fit gives back P
    z = out["z_mean"][1:]
    assert torch.allclose(z, equation.marko_siggia_extension(F[1:], P), atol=0.02)
    P_fit = torch.exp(torch.log(equation.persistence_from_force(F[1:], z)).mean()).item()
    assert math.isclose(P_fit, P, rel_tol=0.12)
    assert torch.all(z[1:] > z[:-1])  # more force, more extension
