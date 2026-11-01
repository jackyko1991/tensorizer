r"""Physics → tensor translation.

A gas reaction  Σ_i ν_i X_i = 0  (ν_i < 0 for reactants, > 0 for products) has equilibrium constant

    ΔG°(T) = ΔH° − T ΔS°,        K(T) = exp(−ΔG° / (R T))

For a batch of conditions b (temperature T_b, pressure P_b, starting moles n⁰_bi) and an extent of reaction ξ_b:

    n_bi = n⁰_bi + ν_i ξ_b                     moles of species i
    a_bi = (n_bi / n_b) (P_b / P°)             ideal-gas activity = partial pressure in bar, n_b = Σ_j n_bj
    ln Q_b = ν_i ln a_bi                        reaction quotient (i summed)

Equilibrium is reached when Q_b = K_b. The sum over species is an einsum; the batch index b is never summed.
"""

import torch

R_GAS = 8.314462618  # J/(mol K)
P_STANDARD = 1.0  # bar


def standard_gibbs(T: torch.Tensor, dH: float, dS: float) -> torch.Tensor:
    """ΔG°_b = ΔH° − T_b ΔS°  (J/mol), with ΔH° and ΔS° taken as temperature-independent.   (B,)"""
    return dH - T * dS


def log_K(T: torch.Tensor, dH: float, dS: float) -> torch.Tensor:
    """ln K_b = −ΔG°_b / (R T_b) = −ΔH°/(R T_b) + ΔS°/R : a straight line in 1/T (van 't Hoff).   (B,)"""
    return -standard_gibbs(T, dH, dS) / (R_GAS * T)


def moles(n0: torch.Tensor, nu: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """n_bi = n⁰_bi + ν_i ξ_b.   n0: (B, S), nu: (S,), xi: (B,) → (B, S).  Nothing summed (outer product)."""
    return n0 + torch.einsum("i,b->bi", nu, xi)


def log_Q(n: torch.Tensor, P: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """ln Q_b = ν_i ln a_bi with a_bi = x_bi P_b / P°.   n: (B, S), P: (B,) → (B,).  i (species) is summed."""
    x = n / n.sum(-1, keepdim=True)  # mole fractions x_bi
    ln_a = torch.log(x) + torch.log(P / P_STANDARD)[:, None]
    return torch.einsum("i,bi->b", nu, ln_a)


def extent_bounds(n0: torch.Tensor, nu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Allowed range of ξ_b: no amount may go negative.   n0: (B, S) → (ξ_min (B,), ξ_max (B,)).

    reactants (ν < 0) give ξ ≤ n⁰_i / |ν_i|;  products (ν > 0) give ξ ≥ −n⁰_i / ν_i.
    """
    limit = n0 / nu.abs()  # (B, S)
    product = nu > 0
    xi_min = torch.where(product, -limit, -torch.inf).amax(-1)
    xi_max = torch.where(product, torch.inf, limit).amin(-1)
    return xi_min, xi_max


def van_t_hoff_slope(dH: float) -> float:
    """d ln K / d(1/T) = −ΔH° / R : exothermic (ΔH° < 0) ⇒ K falls as T rises, endothermic ⇒ K rises."""
    return -dH / R_GAS
