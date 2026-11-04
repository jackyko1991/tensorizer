"""Checks written first (TDD): what any correct loop-quantum-cosmology (LQC) bounce code must satisfy.

Run with `pytest problems/cosmology/quantum_gravity_toy/test.py`. Each test is a property of the effective LQC
equations for a flat FRW universe with a massless scalar field (Ashtekar–Pawlowski–Singh; Taveras): the Hamiltonian
constraint is preserved, the effective Friedmann equation H² = (8πG/3)ρ(1 − ρ/ρ_c) holds, the bounce happens at
ρ = ρ_c for every p_φ, the exact solution V(t) = √(V_B² + 12πG p_φ² t²) is reproduced, small λ gives back classical
general relativity (which hits V = 0), and the Wheeler–DeWitt wave packet follows the classical singular path.
"""

import functools
import math

import pytest
import torch

from . import equation, solve, tensors

G = equation.G


@functools.lru_cache
def _run(p_phi=(50.0, 100.0, 200.0), scales=(1.0, 1.0, 1.0), rho_start=1e-3, t_span=None, dt=2e-3):
    """Universes starting to contract at density rho_start ρ_c (ρ_c of the full area gap), integrated with RK4."""
    p = torch.tensor(p_phi, dtype=torch.float64)
    lam = tensors.LAMBDA * torch.tensor(scales, dtype=torch.float64)
    y0 = tensors.initial_state(p, lam, rho_start * tensors.RHO_C)
    t_span = t_span or 2 * float(equation.time_to_bounce(y0, p, lam).max())
    sol = solve.evolve(y0, p, lam, dt, t_span)
    sol["t_bounce"] = equation.time_to_bounce(y0, p, lam)
    return p, lam, sol


def _vertex(f: torch.Tensor) -> torch.Tensor:
    """Extremum of a sampled curve (T, B) from the parabola through the extreme sample and its neighbours."""
    k = f.argmax(0) if (f[f.shape[0] // 2] > f[0]).all() else f.argmin(0)
    cols = torch.arange(f.shape[1])
    a, b, c = f[k - 1, cols], f[k, cols], f[k + 1, cols]
    return b - (c - a) ** 2 / (8 * (a - 2 * b + c))


def test_critical_density_from_the_area_gap():
    rho_c = equation.critical_density(torch.tensor(tensors.LAMBDA, dtype=torch.float64))
    # ρ_c = 3/(8πGγ²λ²) with λ² = 4√3πγ ℓ_P²: about 0.41 Planck densities for γ = 0.2375
    assert rho_c.item() == pytest.approx(0.41, abs=0.005)


def test_hamiltonian_constraint_is_preserved():
    p, lam, sol = _run()
    C = equation.constraint(sol["y"], p, lam)  # (T, B)
    scale = equation.matter_energy(sol["y"], p)  # p²/(2V) sets the size of each term
    assert (C / scale).abs().max() < 1e-8  # constraint surface C = 0 is kept by the flow


def test_effective_friedmann_equation_holds_along_the_trajectory():
    p, lam, sol = _run()
    H = equation.hubble(sol["y"], lam)
    rho = equation.density(sol["y"], p)
    rho_c = equation.critical_density(lam)
    lhs, rhs = H**2, 8 * math.pi * G / 3 * rho * (1 - rho / rho_c)
    assert ((lhs - rhs) / (8 * math.pi * G / 3 * rho)).abs().max() < 1e-8  # relative to the classical H²


def test_bounce_happens_at_the_critical_density_for_every_p_phi():
    p, lam, sol = _run()
    rho = equation.density(sol["y"], p)  # (T, B)
    rho_c = equation.critical_density(lam)
    assert torch.allclose(_vertex(rho), rho_c, rtol=1e-6)  # ρ_max = ρ_c, independent of p_φ
    V = sol["y"][..., 0]
    V_min = _vertex(V)
    assert torch.allclose(V_min, p / torch.sqrt(2 * rho_c), rtol=1e-6)  # V_B = p_φ/√(2ρ_c), never zero
    H = equation.hubble(sol["y"], lam)
    assert (H[0] < 0).all() and (H[-1] > 0).all()  # contraction → bounce → expansion


def test_matches_the_exact_bounce_solution():
    p, lam, sol = _run()
    V = sol["y"][..., 0]
    t_b = sol["t_bounce"]  # predicted bounce time of each member
    exact = equation.exact_volume(sol["t"][:, None] - t_b, p, lam)
    assert torch.allclose(V, exact, rtol=2e-5)  # V(t) = √(V_B² + 12πG p_φ² t²)


def test_small_lambda_recovers_classical_friedmann():
    # λ → 0 sends ρ_c → ∞: far from the bounce the volume shrinks linearly, V = √(12πG) p_φ |t − t_s| (GR)
    p, lam, sol = _run(p_phi=(100.0,), scales=(1e-3,), rho_start=1e-3, t_span=3.0)
    V = sol["y"][:, 0, 0]
    t = sol["t"]
    V_gr = V[0] - math.sqrt(12 * math.pi * G) * p * (t - t[0])
    assert torch.allclose(V, V_gr, rtol=1e-6)
    H = equation.hubble(sol["y"], lam)[:, 0]
    rho = equation.density(sol["y"], p)[:, 0]
    assert torch.allclose(H**2, 8 * math.pi * G / 3 * rho, rtol=1e-6)  # classical H² = (8πG/3)ρ
    # the classical solution reaches V = 0 (infinite density) in finite time: the Big Bang singularity
    t_sing = equation.classical_singularity_time(V[0], p)
    assert 0 < t_sing < float("inf")


def test_wheeler_dewitt_packet_follows_the_classical_path_into_the_singularity():
    p_phi = 100.0
    phi = torch.linspace(-0.6, 0.2, 9, dtype=torch.float64)
    x = torch.linspace(-2.0, 12.0, 2000, dtype=torch.float64)
    x_b = math.log(p_phi / math.sqrt(2 * tensors.RHO_C))
    psi = equation.wdw_packet(x, phi, p_phi, x_b, spread=0.15)  # (P, X)
    prob = psi.abs() ** 2
    norm = torch.trapezoid(prob, x, dim=1)
    assert torch.allclose(norm, norm[0].expand_as(norm), rtol=1e-8)  # unitary evolution in the clock φ
    mean_x = torch.trapezoid(prob * x, x, dim=1) / norm
    classical = x_b - math.log(2) - math.sqrt(12 * math.pi * G) * phi  # ln V of contracting GR, no bounce
    assert torch.allclose(mean_x, classical, atol=1e-6)
