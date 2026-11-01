r"""Physics → tensor translation.

An ideal gas obeys pV = nRT. A Carnot cycle is made of two isotherms and two adiabats:

    isotherm (T fixed):     p = nRT / V
    adiabat (no heat):      p V^\gamma = \text{const}    ⇔    T V^{\gamma - 1} = \text{const}

The work done by the gas over one cycle is the area enclosed in the p–V diagram,

    W = \oint p \, dV  \approx  \sum_k \tfrac12 (p_k + p_{k+1}) (V_{k+1} - V_k) = \bar p_k \, \Delta V_k

On a tensor of sample points the sum over the repeated index k is exactly torch.trapezoid(p, V).
Heat only flows on the isotherms, where the internal energy is constant so Q = W = nRT ln(V_b / V_a).
Entropy of the ideal gas (up to a constant):

    S = n C_V \ln T + n R \ln V,     C_V = R / (\gamma - 1)

so the isotherms are horizontal and the adiabats vertical lines in the T–S diagram.
"""

import torch

R = 8.314  # gas constant (J / (mol K))


def isotherm_pressure(V: torch.Tensor, n: float, T: float) -> torch.Tensor:
    """p = nRT / V, element-wise on a tensor of volumes."""
    return n * R * T / V


def adiabat_pressure(V: torch.Tensor, p0: float, V0: float, gamma: float) -> torch.Tensor:
    """p = p0 (V0 / V)^γ: the adiabat through the point (V0, p0)."""
    return p0 * (V0 / V) ** gamma


def adiabat_volume(V0: float, T0: float, T1: float, gamma: float) -> float:
    """Volume reached along an adiabat from (V0, T0) to temperature T1:  T V^(γ−1) = const."""
    return V0 * (T0 / T1) ** (1 / (gamma - 1))


def temperature(V: torch.Tensor, p: torch.Tensor, n: float) -> torch.Tensor:
    """T = pV / (nR)."""
    return p * V / (n * R)


def entropy(V: torch.Tensor, T: torch.Tensor, n: float, gamma: float) -> torch.Tensor:
    """S = n C_V ln T + n R ln V  (J/K, up to an additive constant)."""
    c_v = R / (gamma - 1)
    return n * c_v * torch.log(T) + n * R * torch.log(V)


def work(V: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    r"""W = ∫ p dV ≈ Σ_k ½(p_k + p_{k+1})(V_{k+1} − V_k)  along the last dimension.  (..., K) → (...)"""
    return torch.trapezoid(p, V, dim=-1)


def isothermal_heat(n: float, T: float, V_start: float, V_end: float) -> float:
    """Heat absorbed on an isotherm: Q = nRT ln(V_end / V_start)  (negative when compressed)."""
    return n * R * T * torch.log(torch.tensor(V_end / V_start)).item()
