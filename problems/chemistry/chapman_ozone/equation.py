r"""Physics → tensor translation: the Chapman mechanism as a chemical reaction network dc/dt = S·r(c).

Indices: a altitude level, s species (O, O₂, O₃), r reaction (1–4), e element (oxygen).

    (1) O₂ + hν → 2 O             r₁ = j₁ [O₂]
    (2) O + O₂ + M → O₃ + M       r₂ = k₂ [M] [O] [O₂]
    (3) O₃ + hν → O + O₂          r₃ = j₃ [O₃]
    (4) O + O₃ → 2 O₂             r₄ = k₄ [O] [O₃]

    dc_as/dt = S_sr r_ar(c_a)                         (r summed)
The stoichiometry matrix S (species × reactions) holds the net number of molecules of s made by reaction r.
Atoms are conserved because the composition row a_e = (1, 2, 3) oxygen atoms per (O, O₂, O₃) is a left null
vector of S:  a_s S_sr = 0 for every r (s summed).

Backward Euler, c⁺ = c + Δt S r(c⁺), is solved by Newton's method with the Jacobian
    J_asc = S_sr ∂r_ar/∂c_ac                          (r summed),
a 3 × 3 linear solve per altitude. It is stable for any step, which matters because the fast O ⇄ O₃ cycling
(milliseconds to hours) and the slow build-up of odd oxygen Oₓ = O + O₃ (days to centuries) differ by up to 10¹⁴.
"""

import torch

SPECIES = ["O", "O2", "O3"]
S = torch.tensor(
    [
        # r1  r2  r3  r4
        [2.0, -1.0, 1.0, -1.0],  # O
        [-1.0, -1.0, 1.0, 2.0],  # O2
        [0.0, 1.0, -1.0, -1.0],  # O3
    ],
    dtype=torch.float64,
)
ATOMS = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)  # oxygen atoms in O, O2, O3


def rates(c: torch.Tensor, j1, j3, k2, k4, M) -> torch.Tensor:
    """r_ar of the four reactions at every altitude.   c (A, 3), coefficients (A,) → (A, 4), cm⁻³ s⁻¹"""
    o, o2, o3 = c[:, 0], c[:, 1], c[:, 2]  # [O], [O₂], [O₃]
    return torch.stack([j1 * o2, k2 * M * o * o2, j3 * o3, k4 * o * o3], -1)


def rate_jacobian(c: torch.Tensor, j1, j3, k2, k4, M) -> torch.Tensor:
    """∂r_ar/∂c_ac.   (A, 4, 3)"""
    o, o2, o3 = c[:, 0], c[:, 1], c[:, 2]  # [O], [O₂], [O₃]
    z = torch.zeros_like(o)
    return torch.stack(
        [
            torch.stack([z, j1, z], -1),  # r1 = j1 [O2]
            torch.stack([k2 * M * o2, k2 * M * o, z], -1),  # r2 = k2 [M][O][O2]
            torch.stack([z, z, j3], -1),  # r3 = j3 [O3]
            torch.stack([k4 * o3, z, k4 * o], -1),  # r4 = k4 [O][O3]
        ],
        -2,
    )


def dcdt(c: torch.Tensor, *coeffs) -> torch.Tensor:
    """dc_as/dt = S_sr r_ar, r summed.   (A, 3)"""
    return torch.einsum("sr,ar->as", S.to(c), rates(c, *coeffs))


def jacobian(c: torch.Tensor, *coeffs) -> torch.Tensor:
    """J_asc = S_sr ∂r_ar/∂c_ac, r summed.   (A, 3, 3)"""
    return torch.einsum("sr,arc->asc", S.to(c), rate_jacobian(c, *coeffs))


def k2_rate(T: torch.Tensor) -> torch.Tensor:
    """O + O₂ + M → O₃ + M, low-pressure limit 6.0×10⁻³⁴ (T/300)^-2.4 cm⁶ s⁻¹ (JPL evaluation, approximate)."""
    return 6.0e-34 * (T / 300) ** -2.4


def k4_rate(T: torch.Tensor) -> torch.Tensor:
    """O + O₃ → 2 O₂, 8.0×10⁻¹² exp(-2060 / T) cm³ s⁻¹ (JPL evaluation, approximate)."""
    return 8.0e-12 * torch.exp(-2060 / T)


def steady_state(j1, j3, k2, k4, M, O2) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact Chapman steady state with [O₂] fixed.

    d[Oₓ]/dt = 0 gives k₄[O][O₃] = j₁[O₂]; d[O₃]/dt = 0 gives k₂[M][O][O₂] = [O₃](j₃ + k₄[O]).
    Eliminating [O₃]:  k₂k₄[M] [O]² - j₁k₄ [O] - j₁j₃ = 0, so
        [O] = (j₁k₄ + √(j₁²k₄² + 4 k₂k₄[M] j₁j₃)) / (2 k₂k₄[M]),   [O₃] = j₁[O₂] / (k₄[O])."""
    o = (j1 * k4 + torch.sqrt((j1 * k4) ** 2 + 4 * k2 * k4 * M * j1 * j3)) / (2 * k2 * k4 * M)
    return o, j1 * O2 / (k4 * o)


def chapman_approx(j1, j3, k2, k4, M, O2) -> torch.Tensor:
    """The textbook form [O₃] = [O₂] √(j₁ k₂ [M] / (j₃ k₄)), valid when k₄[O] ≪ j₃."""
    return O2 * torch.sqrt(j1 * k2 * M / (j3 * k4))
