r"""Physics → tensor translation.

The membrane is a capacitor charged by an injected current I and discharged through three ion channels:

    C dV/dt = I − ḡ_Na m³h (V − E_Na) − ḡ_K n⁴ (V − E_K) − ḡ_L (V − E_L)

Each gate x ∈ {m, h, n} opens with rate α_x(V) and closes with rate β_x(V):  dx/dt = φ [α_x (1 − x) − β_x x].

For a batch of axons b (one per stimulus current) and the channels k = (Na, K, leak):

    Γ_{kb} = (m_b³ h_b,  n_b⁴,  1)              open fraction of channel k
    I^{ion}_b = ḡ_k Γ_{kb} (V_b − E_k)          k summed
    C \dot V_b = I_b − I^{ion}_b
    \dot x_{gb} = φ [α_g(V_b)(1 − x_{gb}) − β_g(V_b) x_{gb}]     g = (m, h, n), no sum
"""

import torch


def _x_over_1_minus_exp(x: torch.Tensor) -> torch.Tensor:
    """x / (1 − e^{−x}), which tends to 1 at x = 0 (where the formula itself would read 0/0)."""
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 + x / 2, safe / -torch.expm1(-safe))


def rates(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Opening rates α_gb and closing rates β_gb in 1/ms for the gates g = (m, h, n).   V: (B,) → (3, B), (3, B)."""
    alpha_m = _x_over_1_minus_exp((V + 40.0) / 10.0)  # = 0.1 (V+40) / (1 − e^{−(V+40)/10})
    beta_m = 4.0 * torch.exp(-(V + 65.0) / 18.0)
    alpha_h = 0.07 * torch.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
    alpha_n = 0.1 * _x_over_1_minus_exp((V + 55.0) / 10.0)  # = 0.01 (V+55) / (1 − e^{−(V+55)/10})
    beta_n = 0.125 * torch.exp(-(V + 65.0) / 80.0)
    return torch.stack([alpha_m, alpha_h, alpha_n]), torch.stack([beta_m, beta_h, beta_n])


def steady_state(V: torch.Tensor) -> torch.Tensor:
    """x_∞ = α / (α + β): where each gate settles if V is held fixed.   (B,) → (3, B)"""
    alpha, beta = rates(V)
    return alpha / (alpha + beta)


def open_fraction(gates: torch.Tensor) -> torch.Tensor:
    """Γ_kb = (m³h, n⁴, 1) for k = Na, K, leak.   gates (3, B) = (m, h, n) → (3, B)"""
    m, h, n = gates
    return torch.stack([m**3 * h, n**4, torch.ones_like(m)])


def ionic_current(V: torch.Tensor, gates: torch.Tensor, g: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """I^ion_b = ḡ_k Γ_kb (V_b − E_k),  k summed over the three channels.   (B,)"""
    driving = V[None, :] - E[:, None]  # (3, B): how far each ion is from its equilibrium
    return torch.einsum("k,kb,kb->b", g, open_fraction(gates), driving)


def rhs(y: torch.Tensor, I_ext: torch.Tensor, g, E, C_m: float, phi: float) -> torch.Tensor:
    """dy/dt for y of shape (4, B) = (V, m, h, n)."""
    V, gates = y[0], y[1:]
    dV = (I_ext - ionic_current(V, gates, g, E)) / C_m
    alpha, beta = rates(V)
    dgates = phi * (alpha * (1.0 - gates) - beta * gates)  # (3, B), every gate on its own
    return torch.cat([dV[None], dgates])
