"""Physics checks, written first (TDD): what any correct field-tensor boost code must satisfy.

Run with `pytest problems/em/field_tensor_boost/test.py`. F is antisymmetric, the Lorentz invariants B² − E² and
E·B do not change under a boost, a boosted pure E field gains B = −γ β × E, and the boost preserves the metric.
"""

import torch

from . import equation, solve, tensors


def make_F(E, B):
    E = torch.tensor(E, dtype=tensors.DTYPE)
    B = torch.tensor(B, dtype=tensors.DTYPE)
    return E, B, equation.field_tensor(E, B, tensors.levi_civita(3))


def test_field_tensor_round_trip_and_antisymmetry():
    E, B, F = make_F([0.3, -1.0, 0.7], [0.2, 0.5, -0.4])
    assert torch.allclose(F, -F.T)  # F^{μν} = −F^{νμ}
    assert torch.allclose(equation.electric(F), E)
    assert torch.allclose(equation.magnetic(F, tensors.levi_civita(3)), B)


def test_invariants_unchanged_by_boost():
    E, B, F = make_F([0.3, 1.0, 0.2], [0.4, -0.3, 0.5])
    sweep = solve.solve(F, beta_max=0.95, n_steps=20)
    assert torch.allclose(sweep["FF"], 2 * (B @ B - E @ E).expand(20))  # Lorentz invariant F_{μν}F^{μν}
    assert torch.allclose(sweep["FFdual"], -4 * (E @ B).expand(20))  # Lorentz invariant F_{μν}F̃^{μν}
    E_dot_B = torch.einsum("si,si->s", sweep["E"], sweep["B"])
    assert torch.allclose(E_dot_B, (E @ B).expand(20))


def test_boost_of_pure_electric_field_creates_magnetic_field():
    E, _, F = make_F([0.5, 1.0, -0.3], [0.0, 0.0, 0.0])
    sweep = solve.solve(F, beta_max=0.9, n_steps=10)
    beta_vec = torch.zeros(10, 3, dtype=tensors.DTYPE)
    beta_vec[:, 0] = sweep["beta"]  # β along x
    gamma = 1 / torch.sqrt(1 - sweep["beta"] ** 2)
    expected_B = -gamma[:, None] * torch.linalg.cross(beta_vec, E.expand(10, 3))
    assert torch.allclose(sweep["B"], expected_B)  # B' = −γ β × E
    # parallel E unchanged, transverse E stretched by γ
    assert torch.allclose(sweep["E"][:, 0], E[0].expand(10))
    assert torch.allclose(sweep["E"][:, 1:], gamma[:, None] * E[1:])


def test_boost_matrix_preserves_metric():
    Lam = tensors.boost(torch.tensor([0.0, 0.5, 0.99], dtype=tensors.DTYPE))
    eta = tensors.metric()
    # Lorentz transformation: Λᵀ η Λ = η
    assert torch.allclose(torch.einsum("sma,snb,mn->sab", Lam, Lam, eta), eta.expand(3, 4, 4))
