r"""Physics → tensor translation.

The van der Waals equation corrects the ideal gas for molecular attraction (a) and molecular size (b):

    \Big(p + \frac{a n^2}{V^2}\Big)(V - n b) = n R T     ⇔     p = \frac{nRT}{V - nb} - \frac{a n^2}{V^2}

For a family of isotherms T_t sampled at volumes V_v (no index summed — pure broadcasting):

    p_{tv} = \frac{n R T_t}{V_v - n b} - \frac{a n^2}{V_v^2}

The critical point is where an isotherm has a flat inflection, ∂p/∂V = ∂²p/∂V² = 0:

    T_c = \frac{8a}{27Rb},     V_c = 3 n b,     p_c = \frac{a}{27 b^2}

Below T_c the isotherm has an unphysical loop. Maxwell's equal-area rule replaces it by a flat line at the
coexistence pressure p^*, chosen so the areas above and below the line cancel:

    \int_{V_l}^{V_g} (p(V) - p^*) \, dV = 0,      where p(V_l) = p(V_g) = p^*

The integral of p is known in closed form: ∫p dV = nRT ln(V - nb) + a n² / V. The volumes where p(V) = p^*
are the roots of the cubic p^* V³ − (p^* n b + nRT) V² + a n² V − a n³ b = 0, found for every temperature
at once as eigenvalues of its companion matrix. p^* itself comes from bisection on a batch of temperatures.
"""

import torch

R = 8.314  # gas constant (J / (mol K))


def pressure(V: torch.Tensor, T: torch.Tensor, n: float, a: float, b: float) -> torch.Tensor:
    """p_{tv} = nRT_t / (V_v − nb) − a n² / V_v².   V: (V,), T: (T,) → (T, V)"""
    return n * R * T[:, None] / (V[None, :] - n * b) - a * n**2 / V[None, :] ** 2


def critical_point(n: float, a: float, b: float) -> tuple[float, float, float]:
    """(T_c, V_c, p_c) = (8a / 27Rb, 3nb, a / 27b²)."""
    return 8 * a / (27 * R * b), 3 * n * b, a / (27 * b**2)


def volume_roots(p: torch.Tensor, T: torch.Tensor, n: float, a: float, b: float) -> torch.Tensor:
    """The three volumes where the isotherm crosses pressure p, sorted.   p, T: (B,) → (B, 3)

    Divide the cubic by p to get V³ + c₂V² + c₁V + c₀ = 0. Its companion matrix has the roots as eigenvalues.
    """
    c2 = -(n * b + n * R * T / p)
    c1 = a * n**2 / p
    c0 = -a * n**3 * b / p
    companion = torch.zeros(len(p), 3, 3, dtype=p.dtype, device=p.device)  # (B, 3, 3)
    companion[:, 0] = torch.stack([-c2, -c1, -c0], dim=-1)
    companion[:, 1, 0] = 1.0
    companion[:, 2, 1] = 1.0
    return torch.linalg.eigvals(companion).real.sort(dim=-1).values  # three real roots inside the loop


def work_integral(V1: torch.Tensor, V2: torch.Tensor, T: torch.Tensor, n: float, a: float, b: float) -> torch.Tensor:
    """∫ p dV from V1 to V2 along an isotherm = nRT ln((V2 − nb) / (V1 − nb)) + a n² (1/V2 − 1/V1)."""
    return n * R * T * torch.log((V2 - n * b) / (V1 - n * b)) + a * n**2 * (1 / V2 - 1 / V1)


def area_mismatch(p_star: torch.Tensor, T: torch.Tensor, n: float, a: float, b: float) -> torch.Tensor:
    """∫_{V_l}^{V_g} (p − p*) dV: positive when the line p* sits too low.   (B,) → (B,)"""
    roots = volume_roots(p_star, T, n, a, b)
    V_l, V_g = roots[:, 0], roots[:, 2]  # liquid and gas ends of the flat line
    return work_integral(V_l, V_g, T, n, a, b) - p_star * (V_g - V_l)
