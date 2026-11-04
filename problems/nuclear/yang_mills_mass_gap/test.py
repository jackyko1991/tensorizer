"""Physics checks, written first (TDD): what any correct lattice gauge theory code must satisfy.

Run with `pytest problems/nuclear/yang_mills_mass_gap/test.py`. Each test is a property of the Wilson action, not an
implementation detail: the action is gauge invariant, the links stay in the group, 2D lattice gauge theory is exactly
solvable (⟨plaquette⟩ = I₁(β)/I₀(β) for U(1), I₂(β)/I₁(β) for SU(2), and Wilson loops obey an exact area law), and at
strong coupling ⟨plaquette⟩ approaches its leading character-expansion value in any dimension.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_quaternions_form_a_group():
    g = torch.Generator().manual_seed(0)
    a, b, c = (tensors.random_su2((50,), g, CPU) for _ in range(3))
    assert torch.allclose(equation.qmul(equation.qmul(a, b), c), equation.qmul(a, equation.qmul(b, c)), atol=1e-12)
    ab = equation.qmul(a, b)
    assert torch.allclose(ab.norm(dim=-1), torch.ones(50, dtype=ab.dtype), atol=1e-12)  # closed: |ab| = 1
    one = equation.qmul(a, equation.qconj(a))
    assert torch.allclose(one, torch.tensor([1.0, 0, 0, 0], dtype=one.dtype).expand_as(one), atol=1e-12)


@pytest.mark.parametrize("group", ["SU(2)", "U(1)"])
def test_plaquettes_and_wilson_loops_are_gauge_invariant(group):
    g = torch.Generator().manual_seed(1)
    links = tensors.hot_start(2, 3, 6, group, g, CPU)  # (B, D, L, L, L, 4)
    gauge = tensors.random_gauge((2, 6, 6, 6), group, g, CPU)  # g(x)
    moved = equation.gauge_transform(links, gauge)  # U_μ(x) → g(x) U_μ(x) g(x+μ)†
    assert not torch.allclose(moved, links)
    assert torch.allclose(equation.plaquettes(moved), equation.plaquettes(links), atol=1e-12)
    assert torch.allclose(equation.wilson_loops(moved, 3), equation.wilson_loops(links, 3), atol=1e-12)


def test_links_stay_in_the_group_under_updates():
    g = torch.Generator().manual_seed(2)
    links = tensors.hot_start(2, 3, 4, "SU(2)", g, CPU)
    beta = torch.tensor([1.0, 3.0], dtype=torch.float64)
    for _ in range(5):
        links, _ = solve.sweep(links, beta, "SU(2)", g, hits=3)
    assert torch.allclose(links.norm(dim=-1), torch.ones_like(links[..., 0]), atol=1e-12)
    u1 = tensors.hot_start(1, 3, 4, "U(1)", g, CPU)
    u1, _ = solve.sweep(u1, beta[:1], "U(1)", g, hits=3)
    assert u1[..., 1:3].abs().max() == 0  # U(1) links stay on the circle (a₀, 0, 0, a₃)


@pytest.mark.slow
@pytest.mark.parametrize("group", ["U(1)", "SU(2)"])
def test_2d_plaquette_is_the_exact_bessel_ratio(group):
    betas = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    out = solve.simulate(group, dim=2, size=12, betas=betas, n_sweeps=500, n_therm=100, seed=3, measure_every=10)
    exact = equation.strong_coupling_u(betas, group)  # I₁/I₀ (U(1)) or I₂/I₁ (SU(2)): exact in 2D
    measured = out["plaquette"].mean(dim=0)  # (B,)
    assert torch.allclose(measured, exact, atol=6e-3)


@pytest.mark.slow
def test_2d_wilson_loops_obey_the_exact_area_law():
    betas = torch.full((4,), 1.0, dtype=torch.float64)  # four independent replicas in one batch
    out = solve.simulate("U(1)", dim=2, size=12, betas=betas, n_sweeps=600, n_therm=100, seed=4, measure_every=1)
    w = out["wilson"].mean(dim=(0, 1))  # (R, T) for R, T = 1 … 3, averaged over measurements and replicas
    u = equation.strong_coupling_u(betas[:1], "U(1)").item()
    for r, t in [(1, 1), (1, 2), (2, 2), (1, 3)]:
        assert abs(w[r - 1, t - 1].item() - u ** (r * t)) < 6e-3  # W(R, T) = u^{RT}: string tension σ = −ln u


@pytest.mark.slow
def test_3d_strong_coupling_limit():
    betas = torch.tensor([0.2, 0.4], dtype=torch.float64)
    out = solve.simulate("SU(2)", dim=3, size=6, betas=betas, n_sweeps=300, n_therm=50, seed=5, measure_every=10)
    measured = out["plaquette"].mean(dim=0)
    leading = equation.strong_coupling_u(betas, "SU(2)")  # ≈ β/4; next correction is O(u⁵)
    assert torch.allclose(measured, leading, atol=4e-3)
    assert abs(leading[0].item() - 0.05) < 1e-3  # I₂(β)/I₁(β) ≈ β/4 for small β
