r"""Physics → tensor translation.

An ideal gas obeys pV = nRT. Four classic processes start from the same state (V_1, p_1):

    isothermal  (T fixed):   p = nRT / V
    isobaric    (p fixed):   p = p_1
    isochoric   (V fixed):   V = V_1
    adiabatic   (Q = 0):     p V^\gamma = p_1 V_1^\gamma

Sample every path c at K points k and store them as one tensor of shape (4, K, 2). The work done by the
gas is the area under the curve, a sum over the repeated index k (the trapezoid rule):

    W_c = \int p \, dV \approx \sum_k \tfrac12 (p_{c,k} + p_{c,k+1}) (V_{c,k+1} - V_{c,k})

which is torch.trapezoid(p, V, dim=-1). The first law then gives the heat:

    \Delta U_c = n C_V (T_{c,K} - T_{c,0}),     Q_c = \Delta U_c + W_c,     C_V = R / (\gamma - 1)
"""

import torch

R = 8.314  # gas constant (J / (mol K))


def temperature(V: torch.Tensor, p: torch.Tensor, n: float) -> torch.Tensor:
    """T = pV / (nR), element-wise.   (..., K) → (..., K)"""
    return p * V / (n * R)


def work(V: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""W_c = Σ_k ½(p_k + p_{k+1})(V_{k+1} − V_k) along the last dimension.   (4, K) → (4,)"""
    return torch.trapezoid(p, V, dim=-1)


def running_work(V: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Work done so far after each sample (the area traced up to point k).   (4, K) → (4, K)"""
    W = torch.cumulative_trapezoid(p, V, dim=-1)  # (4, K-1)
    return torch.cat([torch.zeros_like(W[:, :1]), W], dim=-1)  # zero work at the start


def internal_energy_change(T: torch.Tensor, n: float, gamma: float) -> torch.Tensor:
    """ΔU = n C_V (T − T_start) with C_V = R / (γ − 1).   (4, K) → (4, K)"""
    c_v = R / (gamma - 1)
    return n * c_v * (T - T[:, :1])


def heat(dU: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """First law: Q = ΔU + W  (heat added = energy stored + work done by the gas)."""
    return dU + W
